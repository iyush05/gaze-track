"""Export a trained GazeNet to ONNX (deployment format) and verify it against PyTorch."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from .model import GazeNet, reparameterize


def load_checkpoint(ckpt: str | Path) -> tuple[GazeNet, dict]:
    blob = torch.load(ckpt, map_location="cpu")
    m = GazeNet(blob["config"]["backbone"], pretrained=False)
    m.load_state_dict(blob["state_dict"])
    return reparameterize(m.eval()), blob["config"]


def export(model: GazeNet, roi: int, path: str | Path) -> dict:
    """Input: (N, roi, roi, 3) uint8 BGR, output: (N, 4). Returns verification + latency info."""
    import onnxruntime as ort

    path = Path(path)
    x = torch.randint(0, 255, (2, roi, roi, 3), dtype=torch.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (x,), str(path), input_names=["patches"], output_names=["out"],
                          dynamic_axes={"patches": {0: "n"}, "out": {0: "n"}}, opset_version=18, dynamo=False)
    sess = session(path)
    with torch.no_grad():
        ref = model(x).numpy()
    diff = float(np.abs(sess.run(None, {"patches": x.numpy()})[0] - ref).max())
    x1 = x[:1].numpy()
    for _ in range(20):
        sess.run(None, {"patches": x1})
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        sess.run(None, {"patches": x1})
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"onnx": str(path), "max_abs_diff_vs_torch": diff,
            "cpu_latency_ms": {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
                               "runtime": f"onnxruntime {ort.__version__}", "threads": 4}}


def session(path: str | Path, threads: int = 4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
