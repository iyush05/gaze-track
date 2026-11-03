"""Command line: calibrate | run | bench | config.

    uv run python -m gaze_track config            # write config.json with defaults to edit
    uv run python -m gaze_track calibrate --profile me
    uv run python -m gaze_track run --profile me --record runs/session.csv
    uv run python -m gaze_track bench --frames 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .calibration import (
    HEAD_PRIOR,
    LAM_EYE,
    LAM_HEAD,
    GazeMapper,
    calibration_targets,
    target_angles,
)
from .camera import PERMISSION_HINT, FrameReader, backend, open_camera, probe
from .config import Config

PROFILE_DIR = Path("profiles")


def cmd_cameras(args) -> None:
    import cv2

    print("Probing cameras (indices 0-4)…")
    found = False
    for i in range(5):
        info = probe(i)
        if not info["opened"]:
            print(f"  [{i}] not available (no more cameras)")
            break
        found = True
        if info["usable"]:
            status = "OK"
        elif info.get("static"):
            status = "STATIC IMAGE (virtual cam?)"
        elif info["frames"]:
            status = "BLANK (virtual cam idle?)"
        else:
            status = "NO FRAMES"
        print(f"  [{i}] {status:28s} frames={info['frames']} size={info['size']} "
              f"brightness={info['mean']} contrast={info['std']} motion={info['motion']}")
        if args.show and info["frames"]:
            cap = cv2.VideoCapture(i, backend())
            end = time.monotonic() + 3
            while time.monotonic() < end:
                ok, f = cap.read()
                if ok:
                    cv2.putText(f, f"camera {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.imshow("cameras", f)
                cv2.waitKey(1)
            cap.release()
    cv2.destroyAllWindows()
    if not found:
        print("\nNo camera could be opened.\n" + PERMISSION_HINT)
    else:
        print("\nPick one with --camera N, or set camera_index in config.json (-1 = auto).")


def _profile_path(name: str) -> Path:
    return PROFILE_DIR / f"{name}.json"


# ---------------------------------------------------------------------------- config
def cmd_config(args) -> None:
    path = Path(args.config)
    if path.exists() and not args.force:
        sys.exit(f"{path} already exists (use --force to overwrite)")
    path.write_text(Config().to_json())
    print(f"Wrote {path}. Edit screen.width_cm / height_cm / camera_to_top_cm for your display.")


# ---------------------------------------------------------------------------- calibrate
def resolve_model(spec: str | None) -> str | None:
    """'latest' → newest runs/train/*/best.pt; otherwise the given path (or None)."""
    if not spec:
        return None
    if spec == "latest":
        # Your own trained models first (newest), then the released model from `download-models`.
        runs = sorted(Path("runs/train").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        runs = runs or sorted(Path("models/gaze").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        if not runs:
            sys.exit("No gaze model found. Run `uv run python -m gaze_track download-models` "
                     "(or train one with `train`), or drop --model.")
        return str(runs[-1])
    if not Path(spec).exists():
        sys.exit(f"Model not found: {spec}")
    return spec


def _learned_raw(learned, frame, lm):
    """Uncalibrated learned (yaw°, pitch°) for a single-face frame, or None."""
    if learned is None or len(lm.faces) != 1:
        return None
    f0 = lm.faces[0]
    n = learned.normalize_frame(frame, f0.transform, f0.landmarks_px)
    if n is None:
        return None
    return learned.to_camera_angles(learned.predict_patches(n[0][None])[0], n[1])


def cmd_calibrate(args) -> None:
    import cv2

    from .pipeline import Pipeline

    model_path = resolve_model(args.model)
    learned = None
    if model_path:
        from .learn.infer import LearnedGaze

        learned = LearnedGaze(model_path)
        print(f"Calibrating the fused estimator with learned model {model_path}")

    cfg = Config.load(args.config)
    pipe = Pipeline(cfg)
    cap = open_camera(cfg, args.camera)
    reader = FrameReader(cap, max_width=cfg.frame_width)
    from .screen_ui import FullscreenCanvas

    canvas = FullscreenCanvas(cfg, "gaze-track calibration")
    targets = calibration_targets(args.points)
    settle_s, collect_s = 1.0, 1.2
    feats_all, tgt_all, groups, ears_all, learned_all = [], [], [], [], []
    t0 = time.monotonic()

    # Intro
    status, frame = "waiting for camera...", None
    while True:
        new = reader.read()
        if new is not None:
            frame = new
            _, f, _ = pipe.landmarks_and_features(frame, time.monotonic() - t0)
            status = "face OK" if f else "NO FACE - adjust lighting/position"
        k = canvas.show(
            f"Sit normally. Follow the dot with your EYES (head natural). [{status}]  SPACE=start  Q=quit",
            preview=frame,
        )
        if k == ord("q"):
            sys.exit("Calibration aborted")
        if k == ord(" "):
            break
    print(f"Calibration window {canvas.ww}x{canvas.wh}; assumed hidden top band {canvas.top_off:.2f} cm")

    for gi, (fx, fy) in enumerate(targets):
        start = time.monotonic()
        while True:
            now = time.monotonic()
            el = now - start
            if el > settle_s + collect_s:
                break
            frame = reader.read()
            if frame is None:
                continue
            lm, f, _ = pipe.landmarks_and_features(frame, now - t0)
            if el > settle_s and f is not None and f.ear >= cfg.zones.eye_closed_ratio:
                u, v = canvas.frac_to_cm(fx, fy)
                feats_all.append(f.vector())
                ears_all.append(f.ear)
                lr = _learned_raw(learned, frame, lm)
                learned_all.append(lr if lr is not None else (np.nan, np.nan))
                tgt_all.append(target_angles(cfg.screen, f.origin_cm, u, v))
                groups.append(gi)
            if canvas.show(f"Point {gi + 1}/{len(targets)}", dot_frac=(fx, fy),
                           progress=min(1.0, el / (settle_s + collect_s))) == ord("q"):
                sys.exit("Calibration aborted")

    cap.release()
    cv2.destroyAllWindows()
    pipe.close()

    if len(set(groups)) < max(4, len(targets) - 2):
        sys.exit(f"Too few valid samples ({len(feats_all)} from {len(set(groups))} points). Check lighting and retry.")
    X, Y, G = np.array(feats_all), np.array(tgt_all), np.array(groups)
    mapper = GazeMapper.fit(X, Y, lam_head=args.lam_head, lam_eye=args.lam_eye, groups=G)
    mapper.stats["ear_baseline"] = float(np.median(ears_all))
    mapper.stats.update({"points": len(targets), "window_px": [canvas.ww, canvas.wh], "top_offset_cm": canvas.top_off})
    out = _profile_path(args.profile)
    mapper.save(out)
    M = np.array(learned_all, float).reshape(-1, 2)
    # Also keep raw samples: useful later as personal fine-tuning data (Phase 4).
    np.savez(out.with_suffix(".samples.npz"), X=X, Y=Y, G=G, M=M)
    print(json.dumps(mapper.stats, indent=2))
    if learned is not None:
        _save_fusion(out, mapper, X, Y, G, M, model_path)
    print(_head_check(mapper))
    print(f"Saved profile → {out}")
    lopo = mapper.stats.get("lopo_mean_err_deg")
    if lopo is not None and lopo > 6:
        print("⚠ Leave-one-point-out error is high; recalibrate with steadier lighting / keep still.")


def _save_fusion(out: Path, mapper: GazeMapper, X, Y, G, M, model_path: str) -> None:
    from .fusion import FusedEstimator

    ok = ~np.isnan(M[:, 0])
    if len(set(G[ok].tolist())) < 6:
        print("⚠ Learned model gave too few usable samples; profile uses the feature mapper only.")
        return
    base = np.array([mapper.predict(x) for x in X[ok]])
    fus = FusedEstimator.fit(base, M[ok], Y[ok], groups=G[ok])
    raw = json.loads(out.read_text())
    raw["fusion"] = fus.to_dict() | {"model": model_path}
    out.write_text(json.dumps(raw, indent=2))
    print(f"fused estimator: LOPO {fus.stats.get('lopo_mean_err_deg', float('nan')):.2f}° "
          f"(feature mapper alone {mapper.stats.get('lopo_mean_err_deg', float('nan')):.2f}°)  weights {fus.stats['weights']}")


def _head_check(m: GazeMapper) -> str:
    dy, dp = m.W[0, 0] * 30, m.W[1, 0] * 30
    return f"30° head turn → gaze yaw {dy:+.1f}°, pitch {dp:+.1f}° (prior: {30 * HEAD_PRIOR:+.0f}, 0)"


def _fusion_kwargs(profile: str | None, disabled: bool = False, threads: int = 2) -> dict:
    """Load the fused estimator + learned model stored in a profile (if any)."""
    if not profile or disabled:
        return {}
    from .fusion import load_profile_fusion

    fus, model = load_profile_fusion(_profile_path(profile))
    if fus is None:
        return {}
    if not model or not Path(model).exists():
        print(f"⚠ Profile references a missing model ({model}); using the feature mapper only.")
        return {}
    from .learn.infer import LearnedGaze

    print(f"Using fused estimator (learned model {model})")
    return {"fusion": fus, "learned": LearnedGaze(model, threads=threads)}


def cmd_refit(args) -> None:
    path = _profile_path(args.profile)
    samples = path.with_suffix(".samples.npz")
    if not samples.exists():
        sys.exit(f"No saved samples at {samples}; run calibrate first.")
    d = np.load(samples)
    old = GazeMapper.load(path)
    m = GazeMapper.fit(d["X"], d["Y"], lam_head=args.lam_head, lam_eye=args.lam_eye, groups=d["G"])
    for k in ("points", "window_px", "top_offset_cm", "ear_baseline"):
        if k in old.stats:
            m.stats[k] = old.stats[k]
    m.save(path)
    print(json.dumps(m.stats, indent=2))
    print("before:", _head_check(old))
    print("after: ", _head_check(m))
    print(f"Saved profile → {path}")


# ---------------------------------------------------------------------------- run
def cmd_run(args) -> None:
    import cv2

    from .pipeline import Pipeline
    from .ui import draw_overlay

    cfg = Config.load(args.config)
    mapper = GazeMapper.load(_profile_path(args.profile)) if args.profile else GazeMapper()
    if args.profile and not mapper.calibrated:
        print(f"Profile '{args.profile}' not found; running uncalibrated.")
    pipe = Pipeline(cfg, mapper, **_fusion_kwargs(args.profile, args.no_fusion, cfg.learned_threads))
    cap = open_camera(cfg, args.camera)
    reader = FrameReader(cap, max_width=cfg.frame_width)

    rec_f = rec_w = ev_f = None
    evidence = None
    if args.evidence is not None:
        from .collect import _save_on_terminate
        from .evidence import EvidenceRecorder

        _save_on_terminate()  # closing the terminal still finalizes the evidence bundle

        ev_dir = Path(args.evidence) if args.evidence else Path("runs/evidence") / (
            f"{args.profile or 'default'}_{time.strftime('%Y%m%d_%H%M%S')}")
        fus = pipe.fusion is not None
        evidence = EvidenceRecorder(ev_dir, {
            "profile": args.profile, "estimator": "fused" if fus else "feature mapper (Phase 0)",
            "calibration_err_deg": ((pipe.fusion.stats if fus else mapper.stats).get("lopo_mean_err_deg")),
            "zones": vars(cfg.zones), "events": vars(cfg.events)})
        print(f"Evidence mode: snapshots around flags only → {ev_dir}")
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        rec_f = open(args.record, "w", newline="")
    if args.events:
        Path(args.events).parent.mkdir(parents=True, exist_ok=True)
        ev_f = open(args.events, "a")

    recent, lat = [], []
    t0 = time.monotonic()
    last, fps = t0, 0.0
    if not args.no_ui:
        wait = np.full((cfg.frame_height, cfg.frame_width, 3), 30, np.uint8)
        cv2.putText(wait, "Starting camera...", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        cv2.imshow("gaze-track", wait)
        cv2.waitKey(1)
    try:
        while True:
            frame = reader.read()
            if frame is None:
                if not args.no_ui and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            now = time.monotonic()
            r = pipe.process(frame, now - t0)
            lat.append(r.timings_ms["total"])
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if evidence is not None:
                evidence.on_frame(frame, r)
            for e in r.events:
                recent.append(e)
                if e.severity != "info":
                    print(f"[{e.t:7.1f}s] {e.severity.upper():7s} {e.kind} {e.detail}")
                if ev_f:
                    ev_f.write(json.dumps(e.to_dict()) + "\n")
                    ev_f.flush()
            if rec_f:
                row = r.row()
                if rec_w is None:
                    rec_w = csv.DictWriter(rec_f, fieldnames=list(row))
                    rec_w.writeheader()
                rec_w.writerow(row)

            if not args.no_ui:
                cv2.imshow("gaze-track", draw_overlay(frame, cfg, r, recent, fps, mapper.calibrated))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif args.duration and now - t0 > args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pipe.close()
        if rec_f:
            rec_f.close()
        if ev_f:
            ev_f.close()
        if evidence is not None:
            evidence.close(duration_s=time.monotonic() - t0)
            print(f"Evidence saved → {evidence.dir}   (review: uv run python -m gaze_track report {evidence.dir})")
        if not args.no_ui:
            cv2.destroyAllWindows()
    if lat:
        a = np.array(lat)
        print(f"Pipeline latency over {len(a)} frames: p50 {np.percentile(a, 50):.2f} ms, "
              f"p95 {np.percentile(a, 95):.2f} ms, p99 {np.percentile(a, 99):.2f} ms")


# ---------------------------------------------------------------------------- bench
def cmd_bench(args) -> None:
    import cv2

    from .pipeline import Pipeline

    cfg = Config.load(args.config)
    mapper = GazeMapper.load(_profile_path(args.profile)) if args.profile else GazeMapper()
    pipe = Pipeline(cfg, mapper, **_fusion_kwargs(args.profile, threads=cfg.learned_threads))
    if args.session:
        from .data.session import load_session

        sess = load_session(args.session)
        paths = [sess.frame_path(r["frame"]) for r in sess.rows]

        class _Replay:
            i = 0

            def read(self):
                if self.i >= len(paths):
                    return False, None
                self.i += 1
                return True, cv2.imread(str(paths[self.i - 1]))

            def release(self):
                pass

        cap = _Replay()
        args.video = args.session  # read frames like a video file below
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = open_camera(cfg, args.camera)
        reader = FrameReader(cap, max_width=cfg.frame_width)
    print("Benchmarking (no video window; use `run` to see the overlay)…")
    stages: dict[str, list[float]] = {}
    n_face = 0
    t = 0.0
    for i in range(args.frames + args.warmup):
        if args.video:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = reader.read()
            if frame is None:
                continue
        t += 1 / 30
        r = pipe.process(frame, t)
        if i < args.warmup:
            continue
        n_face += r.n_faces > 0
        for k, v in r.timings_ms.items():
            stages.setdefault(k, []).append(v)
    cap.release()
    pipe.close()
    n = len(stages.get("total", []))
    if n == 0:
        sys.exit("No frames processed")
    print(f"{n} frames, face found in {n_face} ({100 * n_face / n:.0f}%)")
    print(f"{'stage':20s} {'p50':>8s} {'p95':>8s} {'p99':>8s}  (ms)")
    for k, v in stages.items():
        a = np.array(v)
        print(f"{k:20s} {np.percentile(a, 50):8.2f} {np.percentile(a, 95):8.2f} {np.percentile(a, 99):8.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="gaze_track")
    p.add_argument("--config", default="config.json", help="config file (defaults used if missing)")
    p.add_argument("--camera", type=int, default=None, help="camera index (default: config, -1 = auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cameras", help="list cameras and check which deliver a picture")
    c.add_argument("--show", action="store_true", help="show 3 s of video from each camera")
    c.set_defaults(fn=cmd_cameras)

    c = sub.add_parser("config", help="write a default config.json")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_config)

    c = sub.add_parser("calibrate", help="fullscreen 9-point calibration")
    c.add_argument("--profile", default="default")
    c.add_argument("--points", type=int, default=16, choices=[5, 9, 16],
                   help="16 (default): 4×4 grid close to the edges, ~35 s; 9: 3×3, ~20 s")
    c.add_argument("--lam-head", type=float, default=LAM_HEAD, help="ridge strength on head-pose weights")
    c.add_argument("--lam-eye", type=float, default=LAM_EYE, help="ridge strength on eye weights")
    c.add_argument("--model", default=None,
                   help="learned checkpoint (best.pt) or 'latest' → also fit the fused estimator (recommended)")
    c.set_defaults(fn=cmd_calibrate)

    c = sub.add_parser("refit", help="re-fit a profile from its saved calibration samples")
    c.add_argument("--profile", default="default")
    c.add_argument("--lam-head", type=float, default=LAM_HEAD)
    c.add_argument("--lam-eye", type=float, default=LAM_EYE)
    c.set_defaults(fn=cmd_refit)

    c = sub.add_parser("run", help="live tracking with overlay")
    c.add_argument("--profile", default="default")
    c.add_argument("--record", help="per-frame CSV output path")
    c.add_argument("--events", help="append events as JSONL to this path")
    c.add_argument("--no-ui", action="store_true")
    c.add_argument("--duration", type=float, default=0.0, help="seconds (headless mode only)")
    c.add_argument("--no-fusion", action="store_true", help="ignore the profile's fused estimator")
    c.add_argument("--evidence", nargs="?", const="", default=None,
                   help="save snapshots around flags (no continuous video) to this folder "
                        "(default: runs/evidence/<profile>_<time>); review with `report`")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("bench", help="per-stage latency benchmark")
    c.add_argument("--frames", type=int, default=300)
    c.add_argument("--warmup", type=int, default=30)
    c.add_argument("--video", help="video file instead of the camera")
    c.add_argument("--profile", default=None, help="include this profile's fused estimator in the timing")
    c.add_argument("--session", default=None, help="replay a recorded session's frames instead of the camera")
    c.set_defaults(fn=cmd_bench)

    from . import collect, download, evaluate, report, robustness
    from .data import export, mpiifacegaze

    collect.add_parsers(sub)
    evaluate.add_parser(sub)
    robustness.add_parser(sub)
    download.add_parser(sub)
    report.add_parser(sub)
    export.add_parser(sub)
    mpiifacegaze.add_parser(sub)
    from . import learn

    learn.add_parser(sub)
    learn.add_export_parser(sub)
    learn.add_quantize_parser(sub)
    learn.add_personalize_parser(sub)

    # Accept --camera / --config both before and after the subcommand.
    for sp in sub.choices.values():
        sp.add_argument("--camera", type=int, default=argparse.SUPPRESS, help="camera index (-1 = auto)")
        sp.add_argument("--config", default=argparse.SUPPRESS, help="config file")

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
