"""Guided data collection → labelled session on disk (see data/session.py for the format)."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .camera import FrameReader, open_camera
from .config import Config
from .data.protocol import Trial, build_protocol, gt_screen_zone
from .data.session import DATA_ROOT, SessionWriter, new_session_id
from .features import MP_VERTICAL_FOV_DEG
from .pipeline import Pipeline

CUE_S = 3.0  # read the instruction + countdown before an off-screen trial
RETURN_S = 1.5  # look back at the screen (not recorded)


def _row(trial: Trial, t: float, lm, feats) -> dict:
    row = {
        "t": round(t, 4),
        "trial_id": trial.trial_id,
        "segment": trial.segment,
        "task": trial.task,
        "zone_gt": trial.zone,
        "u_gt": trial.u_cm,
        "v_gt": trial.v_cm,
        "approx": int(trial.approx),
        "n_faces": len(lm.faces),
    }
    if feats is not None:
        row.update(
            head_yaw=round(feats.head_yaw, 3), head_pitch=round(feats.head_pitch, 3),
            head_roll=round(feats.head_roll, 3), eye_h=round(feats.eye_h, 5), eye_v=round(feats.eye_v, 5),
            ear=round(feats.ear, 4), origin_x=round(float(feats.origin_cm[0]), 2),
            origin_y=round(float(feats.origin_cm[1]), 2), origin_z=round(float(feats.origin_cm[2]), 2),
        )
    return row


class _Abort(Exception):
    pass


def _save_on_terminate() -> None:
    """Closing the terminal (SIGHUP) or `kill` (SIGTERM) → behave like Ctrl+C so the session is saved."""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, handler)


def _base_meta(cfg: Config, args, kind: str) -> dict:
    return {
        "kind": kind,
        "subject": args.subject,
        "glasses": args.glasses,
        "lighting": args.lighting,
        "notes": args.notes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "screen": asdict(cfg.screen),
        "zones": asdict(cfg.zones),
        "camera": {"index": args.camera, "frame_width": cfg.frame_width, "frame_height": cfg.frame_height,
                   "assumed_vertical_fov_deg": MP_VERTICAL_FOV_DEG},
    }


def _intro(canvas, reader, pipe, t0, lines: list[str]) -> None:
    status, frame = "waiting for camera...", None
    while True:
        new = reader.read()
        if new is not None:
            frame = new
            _, f, _ = pipe.landmarks_and_features(frame, time.monotonic() - t0)
            status = "face OK" if f else "NO FACE - adjust lighting/position"
        k = canvas.show(f"[{status}]   SPACE = start   Q = quit", preview=frame, big=lines[0],
                        sub=lines[1] if len(lines) > 1 else None)
        if k == ord("q"):
            raise _Abort
        if k == ord(" "):
            return


def _run_dot(trial: Trial, canvas, reader, pipe, writer, t0, label: str) -> None:
    if trial.fx is None:  # screen trial defined in cm → place it in the current window
        trial.fx, trial.fy = canvas.cm_to_frac(trial.u_cm, trial.v_cm)
    else:  # calibration trial defined in window fractions → its physical position
        trial.u_cm, trial.v_cm = canvas.frac_to_cm(trial.fx, trial.fy)
        trial.zone = gt_screen_zone(trial.u_cm, trial.v_cm, canvas.cfg.screen, canvas.cfg.zones)
    start = time.monotonic()
    total = trial.settle_s + trial.record_s
    while (el := time.monotonic() - start) <= total:
        frame = reader.read()
        if frame is None:
            continue
        now = time.monotonic() - t0
        lm, feats, _ = pipe.landmarks_and_features(frame, now)
        if el > trial.settle_s:
            f0 = lm.faces[0] if lm.faces else None
            writer.add(frame, _row(trial, now, lm, feats), f0 and f0.transform, f0 and f0.landmarks_px)
        if canvas.show(label, dot_frac=(trial.fx, trial.fy), progress=min(1.0, el / total)) == ord("q"):
            raise _Abort


def _run_off(trial: Trial, canvas, reader, pipe, writer, t0, label: str) -> None:
    from .screen_ui import beep

    def phase(duration: float, record_after: float | None, countdown: bool = False, **show):
        start = time.monotonic()
        while (el := time.monotonic() - start) <= duration:
            frame = reader.read()
            if frame is None:
                continue
            now = time.monotonic() - t0
            lm, feats, _ = pipe.landmarks_and_features(frame, now)
            if record_after is not None and el > record_after:
                f0 = lm.faces[0] if lm.faces else None
                writer.add(frame, _row(trial, now, lm, feats), f0 and f0.transform, f0 and f0.landmarks_px)
            if countdown:
                show["sub"] = f"look there on the beep in {duration - el:0.0f}...  come back on the second beep"
            if canvas.show(label, **show) == ord("q"):
                raise _Abort

    phase(CUE_S, None, big=trial.instruction, arrow=trial.arrow, countdown=True)
    beep("go")
    phase(trial.settle_s + trial.record_s, trial.settle_s, big=trial.instruction, arrow=trial.arrow,
          sub="hold...")
    beep("back")
    phase(RETURN_S, None, big="Look back at the screen", big_color=(80, 200, 80))


def cmd_collect(args) -> None:
    from .screen_ui import FullscreenCanvas, beep

    _save_on_terminate()
    cfg = Config.load(args.config)
    trials = build_protocol(cfg.screen, cfg.zones, n_screen=args.screen_dots, off_repeats=args.off_repeats,
                            include_no_face=not args.no_no_face, seed=args.seed)
    sid = new_session_id(args.subject)
    out = Path(args.out) / sid
    pipe = Pipeline(cfg)
    cap = open_camera(cfg, args.camera)
    reader = FrameReader(cap, max_width=cfg.frame_width)
    canvas = FullscreenCanvas(cfg, "gaze-track collect")
    meta = _base_meta(cfg, args, "protocol") | {"seed": args.seed}
    writer = SessionWriter(out, meta)
    t0 = time.monotonic()
    aborted = False
    n_off = sum(t.segment == "off" for t in trials)
    try:
        _intro(canvas, reader, pipe, t0, ["Part 1/2: follow the dots with your eyes",
                                          "Sit as in an exam. Look at the centre of each dot until it moves."])
        meta["window_px"], meta["top_offset_cm"] = [canvas.ww, canvas.wh], canvas.top_off
        seg = None
        for i, tr in enumerate(trials):
            if tr.segment == "off" and seg != "off":
                _intro(canvas, reader, pipe, t0, [
                    "Part 2/2: look where you are told",
                    "Read the instruction, look there on the FIRST beep, come back on the SECOND beep.",
                ])
            seg = tr.segment
            label = f"{i + 1}/{len(trials)}  ({tr.segment})"
            if tr.segment in ("calib", "screen"):
                _run_dot(tr, canvas, reader, pipe, writer, t0, label)
            else:
                _run_off(tr, canvas, reader, pipe, writer, t0, f"{label}  off-screen {i - (len(trials) - n_off) + 1}/{n_off}")
        beep("done")
    except (_Abort, KeyboardInterrupt):
        aborted = True
    finally:
        meta["aborted"] = aborted
        meta.setdefault("window_px", [canvas.ww, canvas.wh])
        meta.setdefault("top_offset_cm", canvas.top_off)
        writer.write_json("trials.json", [t.to_dict() for t in trials])
        writer.close()
        canvas.close()
        cap.release()
        pipe.close()
    print(f"{'Aborted; partial' if aborted else 'Saved'} session → {out}  ({writer.n} frames)")
    if not aborted:
        print(f"Evaluate it with:  uv run python -m gaze_track eval {out}")


def cmd_collect_honest(args) -> None:
    """Self-calibration (9 dots), then headless recording while the user works normally.
    Used to measure false alerts per hour."""
    from .screen_ui import FullscreenCanvas, beep

    _save_on_terminate()
    cfg = Config.load(args.config)
    trials = [t for t in build_protocol(cfg.screen, cfg.zones) if t.segment == "calib"]
    sid = new_session_id(args.subject) + "_honest"
    out = Path(args.out) / sid
    pipe = Pipeline(cfg)
    cap = open_camera(cfg, args.camera)
    reader = FrameReader(cap, max_width=cfg.frame_width)
    meta = _base_meta(cfg, args, "honest") | {"minutes": args.minutes, "fps": args.fps, "activity": args.activity}
    writer = SessionWriter(out, meta)
    t0 = time.monotonic()
    canvas = FullscreenCanvas(cfg, "gaze-track collect")
    try:
        _intro(canvas, reader, pipe, t0, ["Quick calibration: follow the dots with your eyes"])
        meta["window_px"], meta["top_offset_cm"] = [canvas.ww, canvas.wh], canvas.top_off
        for i, tr in enumerate(trials):
            _run_dot(tr, canvas, reader, pipe, writer, t0, f"{i + 1}/{len(trials)}")
    except (_Abort, KeyboardInterrupt):
        writer.write_json("trials.json", [t.to_dict() for t in trials])
        writer.close()
        canvas.close()
        cap.release()
        sys.exit("Aborted during calibration")
    canvas.close()
    beep("done")

    honest = Trial(len(trials), "honest", args.activity, "")
    print(f"\nRecording for {args.minutes:.0f} min at {args.fps} fps. Work normally in any app "
          "(read, think, type). Natural glances are fine; don't deliberately look away.\n"
          "Press Ctrl+C to stop early.")
    end = time.monotonic() + args.minutes * 60
    period = 1.0 / args.fps
    next_save = 0.0
    try:
        while time.monotonic() < end:
            frame = reader.read()
            if frame is None:
                continue
            now = time.monotonic() - t0
            if now < next_save:
                continue
            next_save = now + period
            lm, feats, _ = pipe.landmarks_and_features(frame, now)
            f0 = lm.faces[0] if lm.faces else None
            writer.add(frame, _row(honest, now, lm, feats), f0 and f0.transform, f0 and f0.landmarks_px)
            left = end - time.monotonic()
            print(f"\r  {left / 60:4.1f} min left, {writer.n} frames", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    meta["recorded_minutes"] = round((time.monotonic() - t0) / 60, 2)
    writer.write_json("trials.json", [t.to_dict() for t in trials] + [honest.to_dict()])
    writer.close()
    cap.release()
    pipe.close()
    beep("done")
    print(f"Saved honest session → {out}  ({writer.n} frames)")
    print(f"Evaluate it with:  uv run python -m gaze_track eval {out}")


def _typed_update(text: str, key: int) -> str:
    if key in (8, 127):  # backspace / delete
        return text[:-1]
    if key == 13:
        return text + "\n"
    if 32 <= key <= 126:
        return text + chr(key)
    return text


def cmd_collect_behavior(args) -> None:
    """9-dot self-calibration, then scripted honest / cheating segments (see data/behavior.py).
    ESC aborts (q is a normal typing key here)."""
    from .data.behavior import build_behavior_protocol
    from .screen_ui import FullscreenCanvas, beep

    _save_on_terminate()
    cfg = Config.load(args.config)
    segments = build_behavior_protocol(side=args.side, short=args.short)
    calib = [t for t in build_protocol(cfg.screen, cfg.zones) if t.segment == "calib"]
    sid = new_session_id(args.subject) + "_behavior"
    out = Path(args.out) / sid
    pipe = Pipeline(cfg)
    cap = open_camera(cfg, args.camera)
    reader = FrameReader(cap, max_width=cfg.frame_width)
    canvas = FullscreenCanvas(cfg, "gaze-track collect")
    meta = _base_meta(cfg, args, "behavior") | {"side": args.side, "fps": args.fps,
                                                "segments": [vars(sg) | {"label": sg.label} for sg in segments]}
    writer = SessionWriter(out, meta)
    t0 = time.monotonic()
    trials = list(calib)
    aborted = False
    try:
        _intro(canvas, reader, pipe, t0, ["Behaviour session: first a quick calibration",
                                          "Follow the dots with your eyes."])
        meta["window_px"], meta["top_offset_cm"] = [canvas.ww, canvas.wh], canvas.top_off
        for i, tr in enumerate(calib):
            _run_dot(tr, canvas, reader, pipe, writer, t0, f"calibration {i + 1}/{len(calib)}")

        for k, sg in enumerate(segments):
            tr = Trial(len(trials), "behavior", sg.task, sg.label.upper(), instruction=sg.instruction,
                       settle_s=0.0, record_s=sg.seconds)
            trials.append(tr)
            # Setup: user prepares (places notes, etc.) and starts when ready.
            frame = None
            while True:
                new = reader.read()
                frame = new if new is not None else frame
                key = canvas.show_page(f"Part {k + 1}/{len(segments)}: {sg.setup}",
                                       "Press SPACE to start this part.", preview=frame,
                                       footer="ESC = abort session")
                if key == 27:
                    raise _Abort
                if key == ord(" "):
                    break
            beep("go")
            typed, start, period, next_save = "", time.monotonic(), 1.0 / args.fps, 0.0
            while (el := time.monotonic() - start) <= sg.seconds:
                frame = reader.read()
                if frame is None:
                    continue
                now = time.monotonic() - t0
                if now >= next_save:
                    next_save = now + period
                    lm, feats, _ = pipe.landmarks_and_features(frame, now)
                    n = len(lm.faces)
                    if pipe.counter is not None:
                        extra = pipe.counter.count(frame, int(now * 1000))
                        n = extra if extra >= 2 else n
                    f0 = lm.faces[0] if lm.faces else None
                    row = _row(tr, now, lm, feats) | {"n_faces": n}
                    writer.add(frame, row, f0 and f0.transform, f0 and f0.landmarks_px)
                key = canvas.show_page(sg.instruction, sg.content, typed if sg.typing else None,
                                       footer=f"{sg.seconds - el:4.0f} s left   (ESC = abort)",
                                       color=(80, 80, 255) if sg.label == "cheat" else (0, 200, 255))
                if key == 27:
                    raise _Abort
                if sg.typing and key != 255:
                    typed = _typed_update(typed, key)
            beep("back")
        beep("done")
    except (_Abort, KeyboardInterrupt):
        aborted = True
    finally:
        meta["aborted"] = aborted
        writer.write_json("trials.json", [t.to_dict() for t in trials])
        writer.close()
        canvas.close()
        cap.release()
        pipe.close()
    print(f"{'Aborted; partial' if aborted else 'Saved'} behaviour session → {out}  ({writer.n} frames)")
    if not aborted:
        print(f"Evaluate it with:  uv run python -m gaze_track eval {out}")


def add_parsers(sub) -> None:
    def common(c):
        c.add_argument("--subject", required=True, help="short anonymous id, e.g. s01 (used for splits)")
        c.add_argument("--glasses", choices=["yes", "no"], required=True)
        c.add_argument("--lighting", default="normal", help="e.g. normal, dim, backlit, daylight")
        c.add_argument("--notes", default="")
        c.add_argument("--out", default=str(DATA_ROOT))

    c = sub.add_parser("collect", help="guided labelled recording (~4 min)")
    common(c)
    c.add_argument("--screen-dots", type=int, default=24)
    c.add_argument("--off-repeats", type=int, default=2)
    c.add_argument("--no-no-face", action="store_true", help="skip the 'lean out of view' trial")
    c.add_argument("--seed", type=int, default=None, help="protocol randomization (default: random)")
    c.set_defaults(fn=lambda a: cmd_collect(_seed(a)))

    c = sub.add_parser("collect-honest", help="calibrate, then record normal work (false-alert rate)")
    common(c)
    c.add_argument("--minutes", type=float, default=10)
    c.add_argument("--fps", type=float, default=10)
    c.add_argument("--activity", default="reading_typing")
    c.set_defaults(fn=cmd_collect_honest)

    c = sub.add_parser("collect-behavior", help="scripted honest + cheating segments (~5 min) for the event layer")
    common(c)
    c.add_argument("--side", choices=["left", "right"], default="left", help="where you put notes for the side-glance part")
    c.add_argument("--fps", type=float, default=15)
    c.add_argument("--short", action="store_true", help="half-length segments (~2.5 min)")
    c.set_defaults(fn=cmd_collect_behavior)


def _seed(args):
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1_000_000))
    return args
