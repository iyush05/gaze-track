"""Train a compact gaze model on normalized patches (MPIIFaceGaze ± own sessions).

Default split: hold out MPIIFaceGaze persons p00–p02 (never seen in training) for validation.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ..data.index import load_index  # noqa: E402
from .data import angular_error_deg, augment, batches, load_arrays  # noqa: E402
from .model import GazeNet  # noqa: E402

HEAD_LOSS_W = 0.2


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_arrays(model: GazeNet, data, device, batch_size: int = 256) -> dict:
    model.eval()
    errs, subj = [], []
    for b, x, g, _ in batches(data, batch_size, shuffle=False):
        out = model(torch.from_numpy(x).to(device))
        errs.append(angular_error_deg(out[:, :2].float().cpu(), torch.from_numpy(g)).numpy())
        subj.append(data.subject[b])
    e, s = np.concatenate(errs), np.concatenate(subj)
    per = {k: float(e[s == k].mean()) for k in sorted(set(s.tolist()))}
    return {"mean_deg": float(e.mean()), "median_deg": float(np.median(e)), "per_subject": per}


def train(args) -> Path:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    idx = load_index(args.data)
    holdout = getattr(args, "holdout", None) or []
    if holdout:
        # Validate on whole held-out recordings (own sessions share one subject id, so subject-wise
        # splitting can't separate them); every MPIIFaceGaze person is used for training.
        stems = np.array([f.stem for f in idx.files])
        is_val = np.isin(stems[idx.file_idx], holdout)
        val_subj = [f"files:{','.join(holdout)}"]
    else:
        val_subj = [f"mpii_{p}" if not p.startswith("mpii_") and p.startswith("p") else p for p in args.val]
        is_val = np.isin(idx.subject, val_subj)
    print(f"device {device} | {len(idx)} samples, {len(idx.subjects)} subjects | val {val_subj}")
    t0 = time.time()
    tr = load_arrays(idx, ~is_val, limit=args.limit, seed=args.seed)
    va = load_arrays(idx, is_val, limit=args.limit and max(256, args.limit // 4), seed=args.seed)
    own_repeat = int(getattr(args, "own_repeat", 1) or 1)
    if own_repeat > 1:
        # Oversample own (non-MPII) frames so ~1k personal samples aren't drowned by 37k MPII ones.
        own = np.nonzero(~np.char.startswith(tr.subject.astype(str), "mpii_"))[0]
        if len(own):
            rep = np.concatenate([np.arange(len(tr))] + [own] * (own_repeat - 1))
            tr = type(tr)(tr.patches[rep], tr.gaze[rep], tr.head[rep], tr.subject[rep])
            print(f"oversampled {len(own)} own samples ×{own_repeat} → {len(tr)} training samples")
    roi = tr.patches.shape[1]
    print(f"loaded train {len(tr)} / val {len(va)} patches ({roi}px) in {time.time() - t0:.0f}s, "
          f"{(tr.patches.nbytes + va.patches.nbytes) / 1e9:.2f} GB")

    init = getattr(args, "init", None)
    if init:
        blob = torch.load(init, map_location="cpu")
        args.backbone = blob["config"]["backbone"]
        model = GazeNet(args.backbone, pretrained=False)
        model.load_state_dict(blob["state_dict"])
        model = model.to(device)
        print(f"fine-tuning from {init}")
    else:
        model = GazeNet(args.backbone, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(tr) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    mean = model.mean.to(device)
    std = model.std.to(device)

    out = Path(args.out) / f"{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"backbone": args.backbone, "roi": roi, "val_subjects": val_subj, "epochs": args.epochs,
           "init": init, "holdout": holdout, "own_repeat": own_repeat,
           "batch": args.batch, "lr": args.lr, "n_train": len(tr), "n_val": len(va), "data": args.data,
           "device": str(device), "seed": args.seed}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log, best = [], float("inf")
    for ep in range(args.epochs):
        model.train()
        t_ep, losses = time.time(), []
        for _, x, g, h in batches(tr, args.batch, shuffle=True, rng=rng):
            x = torch.from_numpy(x).to(device)
            g = torch.from_numpy(g).to(device)
            h = torch.from_numpy(h).to(device)
            img, g, h = augment(x, g, h)
            pred = model.forward_float((img - mean) / std)
            loss = F.l1_loss(pred[:, :2], g) + HEAD_LOSS_W * F.l1_loss(pred[:, 2:], h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = evaluate_arrays(model, va, device)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_deg": v["mean_deg"],
               "val_per_subject": v["per_subject"], "sec": round(time.time() - t_ep, 1)}
        log.append(rec)
        per = " ".join(f"{k.replace('mpii_', '')}:{e:.2f}" for k, e in v["per_subject"].items())
        print(f"epoch {ep + 1:2d}/{args.epochs}  loss {rec['train_loss']:.4f}  val {v['mean_deg']:.2f}°  [{per}]  {rec['sec']}s",
              flush=True)
        keep_last = getattr(args, "keep", "best") == "last"
        if keep_last or v["mean_deg"] < best:
            # keep=last: when all personal data is in training, the only validation left is other
            # people (MPIIFaceGaze), which is the wrong criterion for picking a personal model.
            best = v["mean_deg"]
            torch.save({"state_dict": model.state_dict(), "config": cfg}, out / "best.pt")
        (out / "log.json").write_text(json.dumps(log, indent=2))

    from .export_onnx import export, load_checkpoint

    fused, _ = load_checkpoint(out / "best.pt")
    onnx_info = export(fused, roi, out / "best.onnx")
    lat = onnx_info["cpu_latency_ms"]
    summary = {"best_val_deg": best, "cpu_latency_ms": lat, "onnx": onnx_info, **cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best val {best:.2f}° (unseen persons)  |  ONNX Runtime CPU batch-1 p50 {lat['p50']:.2f} ms "
          f"(max diff vs torch {onnx_info['max_abs_diff_vs_torch']:.1e})  →  {out}")
    return out
