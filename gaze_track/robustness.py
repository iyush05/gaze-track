"""Robustness stress test: re-run the full pipeline on recorded guided sessions with realistic image
degradations and measure what breaks.

Everything is recomputed from the degraded image (MediaPipe landmarks, head pose, features, the
learned model's input crop), so clean-frame information can't leak into the degraded results.
Degradations are deterministic per frame (seeded) so runs are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .data.session import load_session
from .evaluate import _session_cfg, evaluate_protocol
from .pipeline import Pipeline


def _noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _exposure(img: np.ndarray, gain: float, gamma: float = 1.0) -> np.ndarray:
    x = (img.astype(np.float32) / 255.0) ** gamma * gain
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _backlight(img: np.ndarray, rng) -> np.ndarray:
    """Bright window behind the user: auto-exposure darkens the face, background blows out."""
    h, w = img.shape[:2]
    dark = _exposure(img, 0.45, 1.3)
    yy, xx = np.mgrid[0:h, 0:w]
    # Brightness rises toward the image border (background), face region (centre) stays dark.
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    glow = np.clip((r - 0.55) / 0.5, 0, 1)[..., None] * 255.0
    return np.clip(dark.astype(np.float32) + glow, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "clean": lambda img, rng: img,
    "dim": lambda img, rng: _noise(_exposure(img, 0.45, 1.4), 6, rng),  # dim room, auto-gain noise
    "very_dark": lambda img, rng: _noise(_exposure(img, 0.22, 1.6), 10, rng),
    "overexposed": lambda img, rng: _exposure(img, 1.9, 0.8),  # strong lamp / washed out
    "backlit": lambda img, rng: _noise(_backlight(img, rng), 5, rng),
    "blur": lambda img, rng: cv2.GaussianBlur(img, (0, 0), 2.5),  # focus / motion blur
    "noise": lambda img, rng: _noise(img, 18, rng),  # cheap sensor
    "jpeg_q10": lambda img, rng: cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])[1], 1),
    "lowres_320": lambda img, rng: cv2.resize(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                                                         interpolation=cv2.INTER_AREA),
                                              (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR),
}


def degraded_frames(session, cfg: Config, name: str, learned=None, gate: bool = False) -> tuple[list[dict], dict, float]:
    """Run landmarks/features (and the learned model) on degraded frames.
    Returns (frames, learned predictions, mean image brightness)."""
    from .quality import QualityMonitor

    fn = DEGRADATIONS[name]
    pipe = Pipeline(cfg)
    qm = QualityMonitor(cfg.zones)
    frames, preds, bright = [], {}, []
    for r in session.rows:
        img = cv2.imread(str(session.frame_path(r["frame"])))
        if img is None:
            continue
        rng = np.random.default_rng(int(r["frame"]))
        img = fn(img, rng)
        bright.append(float(img.mean()))
        lm, feats, _ = pipe.landmarks_and_features(img, float(r["t"]))
        unusable = bool(gate and lm.faces and qm.update(img, lm.faces[0].landmarks_px).unusable)
        frames.append({"row": r, "n_faces": len(lm.faces), "feats": feats, "lat_ms": float("nan"),
                       "unusable": unusable})
        if learned is not None and len(lm.faces) == 1:
            f0 = lm.faces[0]
            n = learned.normalize_frame(img, f0.transform, f0.landmarks_px)
            if n is not None:
                p = learned.predict_patches(n[0][None])[0]
                preds[int(r["frame"])] = (*learned.to_camera_angles(p, n[1]), n[2])
    pipe.close()
    return frames, preds, float(np.mean(bright)) if bright else float("nan")


def run(session_paths: list[str], names: list[str], model: str | None, base: Config, gate: bool = False) -> list[dict]:
    learned = None
    if model:
        from .learn.infer import LearnedGaze

        learned = LearnedGaze(model)
    results = []
    for sp in session_paths:
        s = load_session(sp)
        cfg = _session_cfg(base, s)
        for name in names:
            frames, preds, bright = degraded_frames(s, cfg, name, learned, gate)
            row = {"session": s.dir.name, "glasses": s.meta.get("glasses"), "degradation": name,
                   "brightness": round(bright, 1)}
            try:
                rep = evaluate_protocol(s, base, model=model, fuse=bool(model), frames=frames,
                                        preds=preds if model else None)
                row |= {
                    "face_rate": rep["face_detect_rate_onscreen"],
                    "gaze_deg": rep["gaze"]["angular_deg"].get("mean"),
                    "gaze_p90": rep["gaze"]["angular_deg"].get("p90"),
                    "off_recall": rep["off_screen_binary"]["recall"],
                    "off_precision": rep["off_screen_binary"]["precision"],
                    "zone_f1": rep["zones_frame"]["macro_f1"],
                    "calib_lopo": rep["calibration"].get("lopo_mean_err_deg"),
                    "unknown_frac": rep["unknown_frac"],
                    "confident_wrong": rep["confident_wrong_frac"],
                }
            except SystemExit as e:  # calibration impossible (e.g. no face found)
                face = np.mean([f["n_faces"] == 1 for f in frames if f["row"]["segment"] in ("calib", "screen")])
                row |= {"face_rate": float(face), "error": str(e)}
            results.append(row)
            _print_row(row)
    return results


def _fmt(v, f="{:.2f}"):
    return f.format(v) if isinstance(v, (int, float)) and v == v else "  n/a"


def _print_row(r: dict) -> None:
    if "error" in r:
        print(f"  {r['session'][:22]:22s} {str(r['glasses']):3s} {r['degradation']:12s} bright {r['brightness']:5.1f} "
              f"face {100 * r['face_rate']:5.1f}%  FAILED: {r['error']}", flush=True)
        return
    print(f"  {r['session'][:22]:22s} {str(r['glasses']):3s} {r['degradation']:12s} bright {r['brightness']:5.1f} "
          f"face {100 * r['face_rate']:5.1f}%  gaze {_fmt(r['gaze_deg'])}° (p90 {_fmt(r['gaze_p90'])})  "
          f"off-screen R {_fmt(r['off_recall'])} P {_fmt(r['off_precision'])}  zone F1 {_fmt(r['zone_f1'], '{:.3f}')}  "
          f"UNKNOWN {100 * r['unknown_frac']:4.1f}%  confidently wrong {100 * r['confident_wrong']:4.1f}%",
          flush=True)


def cmd_robustness(args) -> None:
    base = Config.load(args.config)
    names = args.degradations or list(DEGRADATIONS)
    unknown = [n for n in names if n not in DEGRADATIONS]
    if unknown:
        raise SystemExit(f"Unknown degradation(s) {unknown}; choose from {list(DEGRADATIONS)}")
    from .cli import resolve_model

    model = resolve_model(args.model) if args.model else None
    print(f"estimator: {'fused (' + model + ')' if model else 'mediapipe+ridge'}")
    res = run(args.sessions, names, model, base, gate=args.gate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"→ {out}")


def add_parser(sub) -> None:
    c = sub.add_parser("robustness", help="stress-test guided sessions under image degradations")
    c.add_argument("sessions", nargs="+", help="guided (protocol) session folders")
    c.add_argument("--degradations", nargs="*", help=f"subset of {list(DEGRADATIONS)}")
    c.add_argument("--model", default=None, help="learned checkpoint → evaluate the fused estimator")
    c.add_argument("--out", default="runs/robustness.json")
    c.add_argument("--gate", action="store_true", help="apply image-quality gating (UNKNOWN when too dark/blurry)")
    c.set_defaults(fn=cmd_robustness)
