"""INT8 post-training quantization of the gaze ONNX model (ONNX Runtime static QDQ, per-channel),
calibrated on real normalized face patches, then checked for accuracy and latency."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..data.index import load_index
from .export_onnx import session


class _PatchReader:
    def __init__(self, patches: np.ndarray, batch: int = 16):
        self.batches = [patches[i : i + batch] for i in range(0, len(patches), batch)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        b = self.batches[self.i]
        self.i += 1
        return {"patches": np.ascontiguousarray(b, dtype=np.uint8)}

    def rewind(self):
        self.i = 0


def _val_error_deg(sess, patches: np.ndarray, gaze: np.ndarray) -> float:
    from ..data.normalize import from_eth_pitchyaw

    out = np.concatenate([sess.run(None, {"patches": patches[i : i + 256]})[0][:, :2]
                          for i in range(0, len(patches), 256)])
    a = np.array([from_eth_pitchyaw(*p) for p in out])
    b = np.array([from_eth_pitchyaw(*p) for p in gaze])
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())


def _latency(sess, roi: int, n: int = 300) -> dict:
    x = np.random.default_rng(0).integers(0, 255, (1, roi, roi, 3), dtype=np.uint8)
    for _ in range(20):
        sess.run(None, {"patches": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"patches": x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95))}


def quantize(fp32_onnx: str | Path, val_subjects: list[str], data: list[str], n_calib: int = 512,
             n_val: int = 2000, seed: int = 0) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_onnx = Path(fp32_onnx)
    idx = load_index(data)
    rng = np.random.default_rng(seed)
    is_val = np.isin(idx.subject, val_subjects) & idx.has_gaze()
    train_ids = np.nonzero(~is_val & idx.has_gaze())[0]
    val_ids = np.nonzero(is_val)[0]
    calib = idx.patches(np.sort(rng.choice(train_ids, min(n_calib, len(train_ids)), replace=False)))
    vsel = np.sort(rng.choice(val_ids, min(n_val, len(val_ids)), replace=False))
    vpatch, vgaze = idx.patches(vsel), idx.labels["gaze"][vsel]

    pre = fp32_onnx.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_onnx), str(pre))
    out = fp32_onnx.with_name(fp32_onnx.stem + ".int8.onnx")
    quantize_static(str(pre), str(out), _PatchReader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax)
    pre.unlink(missing_ok=True)

    s32, s8 = session(fp32_onnx), session(out)
    roi = vpatch.shape[1]
    return {
        "int8_onnx": str(out),
        "val_subjects": val_subjects,
        "fp32": {"val_deg": _val_error_deg(s32, vpatch, vgaze), "cpu_ms": _latency(s32, roi),
                 "size_mb": fp32_onnx.stat().st_size / 1e6},
        "int8": {"val_deg": _val_error_deg(s8, vpatch, vgaze), "cpu_ms": _latency(s8, roi),
                 "size_mb": out.stat().st_size / 1e6},
    }
