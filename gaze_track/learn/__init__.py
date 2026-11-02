"""Learned gaze models (Phase 2+). Parsers live here so the CLI doesn't import torch at startup."""


def _train(args):
    from .train import train

    return train(args)


def add_parser(sub) -> None:
    c = sub.add_parser("train", help="train a compact gaze model on normalized patches")
    c.add_argument("--data", nargs="+", default=["data/normalized/mpiifacegaze"])
    c.add_argument("--backbone", default="mobileone_s0")
    c.add_argument("--val", nargs="+", default=["p00", "p01", "p02"], help="held-out subjects")
    c.add_argument("--epochs", type=int, default=12)
    c.add_argument("--batch", type=int, default=128)
    c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--limit", type=int, default=None, help="max training samples (smoke tests)")
    c.add_argument("--device", default="auto")
    c.add_argument("--no-pretrained", action="store_true")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="runs/train")
    c.add_argument("--init", default=None, help="checkpoint to fine-tune from (Phase 4 personalization)")
    c.add_argument("--holdout", nargs="*", default=[], help="validate on these whole .npz files (by stem)")
    c.add_argument("--own-repeat", type=int, default=1, help="oversample non-MPII samples this many times")
    c.add_argument("--keep", choices=["best", "last"], default="best", help="checkpoint to keep")
    c.set_defaults(fn=_train)


def _export(args):
    import json
    from pathlib import Path

    from .export_onnx import export, load_checkpoint

    model, cfg = load_checkpoint(args.checkpoint)
    info = export(model, int(cfg["roi"]), Path(args.checkpoint).with_suffix(".onnx"))
    print(json.dumps(info, indent=2))


def add_export_parser(sub) -> None:
    c = sub.add_parser("export-onnx", help="export a trained checkpoint to ONNX next to it (+ verify, latency)")
    c.add_argument("checkpoint")
    c.set_defaults(fn=_export)


def _quantize(args):
    import json
    from pathlib import Path

    from .quantize import quantize

    ckpt = Path(args.checkpoint)
    onnx = ckpt.with_suffix(".onnx")
    if not onnx.exists():
        raise SystemExit(f"{onnx} missing; run export-onnx first")
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    res = quantize(onnx, cfg["val_subjects"], cfg["data"] if isinstance(cfg["data"], list) else [cfg["data"]])
    (ckpt.parent / "quantize.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def add_quantize_parser(sub) -> None:
    c = sub.add_parser("quantize-onnx", help="INT8 static quantization of an exported model (+ accuracy/latency check)")
    c.add_argument("checkpoint", help="best.pt whose best.onnx should be quantized")
    c.set_defaults(fn=_quantize)


def _personalize(args):
    """Export this subject's recorded sessions and fine-tune the base model on them + MPIIFaceGaze."""
    from pathlib import Path
    from types import SimpleNamespace

    from ..data.export import export_session
    from ..data.session import list_sessions, load_session
    from .train import train

    base = args.base
    if base == "latest":
        runs = sorted(Path("runs/train").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit("No base model in runs/train; train one first.")
        base = str(runs[-1])
    sessions = [p for p in list_sessions() if load_session(p).subject == args.subject]
    if not sessions:
        raise SystemExit(f"No recorded sessions for subject {args.subject} in data/own (use `collect`).")
    files = []
    for p in sessions:
        f = export_session(p, "data/normalized")
        files.append(str(f))
        print(f"exported {p.name} → {f}")
    out = train(SimpleNamespace(
        seed=0, device="auto", data=["data/normalized/mpiifacegaze", *files], val=["p00", "p01", "p02"],
        limit=None, backbone=None, no_pretrained=True, lr=args.lr, batch=128, epochs=args.epochs,
        out=f"runs/personal/{args.subject}", init=base, holdout=[], own_repeat=args.own_repeat, keep="last"))
    print(f"\nPersonal model: {out / 'best.pt'}\nUse it:  uv run python -m gaze_track calibrate --profile {args.profile} "
          f"--model {out / 'best.pt'}")


def add_personalize_parser(sub) -> None:
    c = sub.add_parser("personalize", help="fine-tune the gaze model on all of a subject's recorded sessions")
    c.add_argument("--subject", required=True)
    c.add_argument("--base", default="latest", help="base checkpoint (default: newest in runs/train)")
    c.add_argument("--epochs", type=int, default=3)
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--own-repeat", type=int, default=8)
    c.add_argument("--profile", default="me", help="(only used in the printed next-step command)")
    c.set_defaults(fn=_personalize)
