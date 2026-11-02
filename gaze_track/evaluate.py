"""Offline evaluation of a recorded session.

Protocol sessions
    1. Fit a calibration on the session's own `calib` trials (or use --profile).
    2. Replay every frame of the `screen` and `off` trials through the estimator.
    3. Report gaze error on exact on-screen targets, per-frame zone confusion / macro-F1,
       off-screen detection (binary), and per-trial majority-vote accuracy.
Honest sessions
    Replay the `honest` frames through the full live pipeline (smoothing, hysteresis, events) and
    report warnings/alerts per hour: every one of them is a false alarm by construction.

Frame-level zones are classified *without* temporal smoothing or hysteresis, so the numbers reflect
the estimator itself; the trial-level majority vote approximates what smoothing adds.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .calibration import GazeMapper, target_angles
from .config import Config, ScreenGeometry
from .data.session import Session, load_session
from .metrics import angular_error_deg, binary_report, per_hour, summarize, zone_report
from .features import GazeFeatures
from .pipeline import VETO_ZONES, Pipeline, eye_closed_threshold, gate_zone, gaze_zone
from .zones import Zone

ZONE_LABELS = ["ON_SCREEN", "EDGE", "NEAR_OFF", "FAR_OFF", "DOWN_KEYBOARD", "EYES_CLOSED", "NO_FACE",
               "MULTI_FACE", "UNKNOWN"]
OFF = {"NEAR_OFF", "FAR_OFF"}


def _load_profile(profile: str) -> GazeMapper:
    path = Path(profile) if profile.endswith(".json") else Path("profiles") / f"{profile}.json"
    if not path.exists():
        raise SystemExit(f"Profile not found: {path}")
    return GazeMapper.load(path)


def _session_cfg(base: Config, s: Session) -> Config:
    """Physical screen geometry from the session; decision thresholds from the *current* config,
    so re-evaluating old sessions reflects threshold changes."""
    return Config(**{k: v for k, v in vars(base).items() if k != "screen"}, screen=ScreenGeometry(**s.meta["screen"]))


def _extract(pipe: Pipeline, s: Session) -> list[dict]:
    """Run landmarks + features on every saved frame (in recorded order)."""
    out = []
    for r in s.rows:
        img = cv2.imread(str(s.frame_path(r["frame"])))
        if img is None:
            continue
        lm, feats, tm = pipe.landmarks_and_features(img, float(r["t"]))
        out.append({"row": r, "n_faces": len(lm.faces), "feats": feats, "lat_ms": tm["landmarks"] + tm["features"]})
    return out


def _stored(s: Session) -> list[dict]:
    """Frames from the features saved at capture time (no MediaPipe; seconds instead of minutes).
    Identical features to `_extract` up to MediaPipe's run-to-run nondeterminism."""
    out = []
    for r in s.rows:
        n = int(r["n_faces"] or 0)
        feats = None
        if r.get("head_yaw") is not None:
            feats = GazeFeatures(
                head_yaw=r["head_yaw"], head_pitch=r["head_pitch"], head_roll=r["head_roll"],
                eye_h=r["eye_h"], eye_v=r["eye_v"], ear=r["ear"],
                origin_cm=np.array([r["origin_x"], r["origin_y"], r["origin_z"]], float),
                face_width_px=1e9,  # not stored; face-size gate was passed at capture time
            )
        out.append({"row": r, "n_faces": n, "feats": feats, "lat_ms": float("nan")})
    return out


def _learned_predictions(s: Session, model_path: str, rows: list[dict] | None = None) -> dict:
    """frame → (yaw°, pitch°, face_center_cm) from the learned model, uncalibrated (camera frame)."""
    from .learn.infer import LearnedGaze

    est = LearnedGaze(model_path)
    arr = s.arrays()
    T, L = arr["transform"], arr["landmarks"].astype(np.float32)
    keys, patches, Rs, fcs = [], [], [], []
    for r in rows if rows is not None else s.rows:
        i = int(r["frame"])
        if int(r["n_faces"] or 0) != 1 or np.isnan(T[i]).any() or np.isnan(L[i]).any():
            continue
        img = cv2.imread(str(s.frame_path(i)))
        if img is None:
            continue
        n = est.normalize_frame(img, T[i], L[i])
        if n is None:
            continue
        keys.append(i)
        patches.append(n[0])
        Rs.append(n[1])
        fcs.append(n[2])
    out = {}
    for j in range(0, len(keys), 256):
        pred = est.predict_patches(np.array(patches[j : j + 256]))
        for k, p, R, fc in zip(keys[j : j + 256], pred, Rs[j : j + 256], fcs[j : j + 256]):
            out[k] = (*est.to_camera_angles(p, R), fc)
    return out


def _fit_learned_calibration(cfg: Config, rows: list[dict], preds: dict):
    from .learn.infer import AffineCalibration

    P, Y, G = [], [], []
    for r in rows:
        p = preds.get(int(r["frame"]))
        if r["segment"] != "calib" or p is None:
            continue
        P.append(p[:2])
        Y.append(target_angles(cfg.screen, p[2], r["u_gt"], r["v_gt"]))
        G.append(int(r["trial_id"]))
    if len(set(G)) < 6:
        raise SystemExit(f"Learned calibration has only {len(set(G))} usable points.")
    return AffineCalibration.fit(np.array(P), np.array(Y), groups=np.array(G))


def _fit_fusion(cfg: Config, frames: list[dict], mapper: GazeMapper, preds: dict):
    """Fit the fused estimator on the session's calibration frames (targets from the eye-midpoint
    origin, which is the origin used for screen projection at run time)."""
    from .fusion import FusedEstimator

    B, L, Y, G = [], [], [], []
    thr = eye_closed_threshold(cfg, mapper.stats.get("ear_baseline"))
    for f in frames:
        r, ft = f["row"], f["feats"]
        p = preds.get(int(r["frame"]))
        if r["segment"] != "calib" or ft is None or f["n_faces"] != 1 or ft.ear < thr or p is None:
            continue
        B.append(mapper.predict(ft.vector()))
        L.append(p[:2])
        Y.append(target_angles(cfg.screen, ft.origin_cm, r["u_gt"], r["v_gt"]))
        G.append(int(r["trial_id"]))
    if len(set(G)) < 6:
        raise SystemExit(f"Fusion calibration has only {len(set(G))} usable points.")
    return FusedEstimator.fit(np.array(B), np.array(L), np.array(Y), groups=np.array(G))


def _fit_session_calibration(cfg: Config, frames: list[dict]) -> GazeMapper:
    X, Y, G, E = [], [], [], []
    for f in frames:
        r, ft = f["row"], f["feats"]
        if r["segment"] != "calib" or ft is None or f["n_faces"] != 1 or ft.ear < cfg.zones.eye_closed_ratio:
            continue
        X.append(ft.vector())
        E.append(ft.ear)
        Y.append(target_angles(cfg.screen, ft.origin_cm, r["u_gt"], r["v_gt"]))
        G.append(int(r["trial_id"]))
    if len(set(G)) < 6:
        raise SystemExit(f"Session calibration has only {len(set(G))} usable points; pass --profile instead.")
    m = GazeMapper.fit(np.array(X), np.array(Y), groups=np.array(G))
    m.stats["ear_baseline"] = float(np.median(E))
    return m


def evaluate_protocol(s: Session, base: Config, profile: str | None = None, fast: bool = False,
                      model: str | None = None, fuse: bool = False) -> dict:
    cfg = _session_cfg(base, s)
    t0 = time.perf_counter()
    if fast:
        frames = _stored(s)
    else:
        pipe = Pipeline(cfg)
        frames = _extract(pipe, s)
        pipe.close()
    extract_s = time.perf_counter() - t0

    if profile:
        mapper = _load_profile(profile)
        calib_src = f"profile:{profile}"
    else:
        mapper = _fit_session_calibration(cfg, frames)
        calib_src = "session"
    neutral = tuple(mapper.stats.get("neutral_head", [0.0, 0.0]))
    preds, lcal, fused = None, None, None
    if model:
        preds = _learned_predictions(s, model)
        if fuse:
            fused = _fit_fusion(cfg, frames, mapper, preds)
        else:
            lcal = _fit_learned_calibration(cfg, s.rows, preds)

    gaze_err, cm_err, yaw_err, pitch_err, edge_flag = [], [], [], [], []
    y_true, y_pred, trial_votes = [], [], {}
    for f in frames:
        r, ft = f["row"], f["feats"]
        if r["segment"] not in ("screen", "off"):
            continue
        pre = gate_zone(cfg, ft, f["n_faces"], mapper.stats.get("ear_baseline"))
        uv = None
        origin = ft.origin_cm if ft is not None else None
        if pre is None and fused is not None:
            p = preds.get(int(r["frame"]))
            yaw, pitch = fused.predict(mapper.predict(ft.vector()), p[:2] if p is not None else None)
        elif pre is None and preds is not None:
            p = preds.get(int(r["frame"]))
            if p is None:
                pre = Zone.UNKNOWN
            else:
                yaw, pitch = lcal.apply(p[0], p[1])
                origin = p[2]
        elif pre is None:
            yaw, pitch = mapper.predict(ft.vector())
        if pre is not None:
            zone = pre.value
        else:
            z, uv = gaze_zone(cfg, ft, yaw, pitch, origin, neutral, current=None)
            if fused is not None and z not in VETO_ZONES:
                # Same off-screen veto as the live pipeline (see Pipeline.step).
                zb, _ = gaze_zone(cfg, ft, *mapper.predict(ft.vector()), ft.origin_cm, neutral, current=None)
                if zb in VETO_ZONES:
                    z = zb
            zone = z.value
            if r["segment"] == "screen":
                ty, tp = target_angles(cfg.screen, origin, r["u_gt"], r["v_gt"])
                gaze_err.append(float(angular_error_deg(yaw, pitch, ty, tp)[0]))
                yaw_err.append(yaw - ty)
                pitch_err.append(pitch - tp)
                cm_err.append(float(np.hypot(uv[0] - r["u_gt"], uv[1] - r["v_gt"])) if uv else np.nan)
                edge_flag.append(r["zone_gt"] == "EDGE")
        y_true.append(r["zone_gt"])
        y_pred.append(zone)
        trial_votes.setdefault(int(r["trial_id"]), (r["zone_gt"], r["task"], []))[2].append(zone)

    edge_flag = np.array(edge_flag, bool)
    gaze_err_a, cm_err_a = np.array(gaze_err), np.array(cm_err)
    present = [l for l in ZONE_LABELS if l in set(y_true) | set(y_pred)]

    trial_rows = []
    for tid, (gt, task, votes) in sorted(trial_votes.items()):
        maj = Counter(votes).most_common(1)[0][0]
        trial_rows.append({"trial_id": tid, "task": task, "zone_gt": gt, "majority": maj,
                           "correct": maj == gt, "frac_correct": votes.count(gt) / len(votes)})
    by_task: dict[str, list[bool]] = {}
    for t in trial_rows:
        by_task.setdefault(t["task"] if t["task"] != "dot" else f"dot:{t['zone_gt']}", []).append(t["correct"])

    lat = np.array([f["lat_ms"] for f in frames], float)
    face_rate = float(np.mean([f["n_faces"] == 1 for f in frames if f["row"]["segment"] in ("calib", "screen")]))
    return {
        "session": s.dir.name,
        "subject": s.subject,
        "glasses": s.meta.get("glasses"),
        "lighting": s.meta.get("lighting"),
        "estimator": (f"fused:{Path(model).parent.name}" if fused else f"learned:{Path(model).parent.name}")
        if model else "mediapipe+ridge (phase0)",
        "calibration": {"source": calib_src, **{k: (fused or lcal or mapper).stats.get(k) for k in ("lopo_mean_err_deg", "n_samples")},
                        **({"fusion_weights": fused.stats["weights"]} if fused else {}),
                        **({"uncalibrated_err_deg": lcal.stats["raw_mean_err_deg"]} if lcal else {})},
        "gaze": {
            "angular_deg": summarize(gaze_err_a),
            "angular_deg_interior": summarize(gaze_err_a[~edge_flag]) if len(edge_flag) else {"n": 0},
            "angular_deg_edge_band": summarize(gaze_err_a[edge_flag]) if len(edge_flag) else {"n": 0},
            "screen_cm": summarize(cm_err_a),
            "yaw_bias_deg": float(np.mean(yaw_err)) if yaw_err else None,
            "pitch_bias_deg": float(np.mean(pitch_err)) if pitch_err else None,
            "yaw_mae_deg": float(np.mean(np.abs(yaw_err))) if yaw_err else None,
            "pitch_mae_deg": float(np.mean(np.abs(pitch_err))) if pitch_err else None,
        },
        "zones_frame": zone_report(y_true, y_pred, present),
        "off_screen_binary": binary_report([t in OFF for t in y_true], [p in OFF for p in y_pred]),
        "trials": {"accuracy": float(np.mean([t["correct"] for t in trial_rows])) if trial_rows else None,
                   "by_task": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_task.items())},
                   "rows": trial_rows},
        "face_detect_rate_onscreen": face_rate,
        "latency_ms": {"landmarks+features": summarize(lat)},
        "frames": len(frames),
        "replay_seconds": round(extract_s, 1),
    }


def _replay_setup(s: Session, base: Config, profile, fast, model, fuse):
    """Calibration (session calib dots or a profile) + optional learned/fused estimator."""
    cfg = _session_cfg(base, s)
    calib_rows = Session(s.dir, s.meta, s.trials, [r for r in s.rows if r["segment"] == "calib"])
    if profile:
        mapper = _load_profile(profile)
    elif fast or model:
        mapper = _fit_session_calibration(cfg, _stored(calib_rows))
    else:
        pipe = Pipeline(cfg)
        mapper = _fit_session_calibration(cfg, _extract(pipe, calib_rows))
        pipe.close()
    preds, lcal, fused = None, None, None
    if model:
        fast = True  # learned path replays stored features + saved poses
        preds = _learned_predictions(s, model)
        if fuse:
            fused = _fit_fusion(cfg, _stored(calib_rows), mapper, preds)
        else:
            lcal = _fit_learned_calibration(cfg, s.rows, preds)
    stored = {f["row"]["frame"]: f for f in _stored(s)} if fast else {}
    return cfg, mapper, fast, preds, lcal, fused, stored


def _replay(s: Session, rows: list[dict], cfg, mapper, fast, preds, lcal, fused, stored):
    """Full live pipeline (smoothing, hysteresis, events) over `rows` → (events, zone counter)."""
    pipe = Pipeline(cfg, mapper, landmarker=not fast)
    events, zones = [], Counter()
    for r in rows:
        if fast:
            f = stored[r["frame"]]
            gaze = None
            if fused is not None and f["feats"] is not None and f["n_faces"] == 1:
                p = preds.get(int(r["frame"]))
                gaze = (*fused.predict(mapper.predict(f["feats"].vector()), p[:2] if p is not None else None),
                        f["feats"].origin_cm)
            elif preds is not None and lcal is not None:
                p = preds.get(int(r["frame"]))
                if p is not None:
                    gaze = (*lcal.apply(p[0], p[1]), p[2])
            res = pipe.step(f["feats"], f["n_faces"], float(r["t"]), gaze=gaze)
        else:
            img = cv2.imread(str(s.frame_path(r["frame"])))
            if img is None:
                continue
            res = pipe.process(img, float(r["t"]))
        zones[res.zone.value] += 1
        events += res.events
    pipe.close()
    return events, zones


def evaluate_honest(s: Session, base: Config, profile: str | None = None, fast: bool = False,
                    model: str | None = None, fuse: bool = False) -> dict:
    setup = _replay_setup(s, base, profile, fast, model, fuse)
    rows = [r for r in s.rows if r["segment"] == "honest"]
    events, zones = _replay(s, rows, *setup)
    dur = float(rows[-1]["t"] - rows[0]["t"]) if len(rows) > 1 else 0.0
    kinds = Counter(e.kind for e in events if e.severity != "info")
    n_warn = sum(e.severity == "warning" for e in events)
    n_alert = sum(e.severity == "alert" for e in events)
    return {
        "session": s.dir.name,
        "subject": s.subject,
        "kind": "honest",
        "minutes": round(dur / 60, 2),
        "alerts_per_hour": per_hour(n_alert, dur),
        "warnings_per_hour": per_hour(n_warn, dur),
        "brief_glances_per_hour": per_hour(sum(e.kind == "brief_glance" for e in events), dur),
        "flag_counts": dict(kinds),
        "zone_time_frac": {k: round(v / max(1, sum(zones.values())), 3) for k, v in zones.most_common()},
        "events": [e.to_dict() for e in events if e.severity != "info"],
    }


def evaluate_behavior(s: Session, base: Config, profile: str | None = None, fast: bool = False,
                      model: str | None = None, fuse: bool = False) -> dict:
    """Per scripted segment: was a warning/alert raised (cheat → should, honest → shouldn't), how
    fast, and which kinds. Each segment is replayed with a fresh pipeline (segments are separated
    by setup pauses, so events must not carry over)."""
    setup = _replay_setup(s, base, profile, fast, model, fuse)
    segs = []
    for tid in sorted({int(r["trial_id"]) for r in s.rows if r["segment"] == "behavior"}):
        rows = [r for r in s.rows if r["segment"] == "behavior" and int(r["trial_id"]) == tid]
        events, zones = _replay(s, rows, *setup)
        flags = [e for e in events if e.severity != "info"]
        t0, t1 = float(rows[0]["t"]), float(rows[-1]["t"])
        segs.append({
            "task": rows[0]["task"],
            "label": str(rows[0]["zone_gt"]).lower(),
            "seconds": round(t1 - t0, 1),
            "flagged": bool(flags),
            "n_warnings": sum(e.severity == "warning" for e in flags),
            "n_alerts": sum(e.severity == "alert" for e in flags),
            "first_flag_s": round(flags[0].t - t0, 1) if flags else None,
            "flag_kinds": dict(Counter(e.kind for e in flags)),
            "brief_glances": sum(e.kind == "brief_glance" for e in events),
            "zone_time_frac": {k: round(v / max(1, sum(zones.values())), 3) for k, v in zones.most_common(4)},
        })
    cheat = [g for g in segs if g["label"] == "cheat"]
    honest = [g for g in segs if g["label"] == "honest"]
    honest_s = sum(g["seconds"] for g in honest)
    ttf = [g["first_flag_s"] for g in cheat if g["first_flag_s"] is not None]
    return {
        "session": s.dir.name,
        "subject": s.subject,
        "kind": "behavior",
        "estimator": ("fused" if fuse else "learned") if model else "mediapipe+ridge (phase0)",
        "cheat_detected": f"{sum(g['flagged'] for g in cheat)}/{len(cheat)}",
        "cheat_detection_rate": (sum(g["flagged"] for g in cheat) / len(cheat)) if cheat else None,
        "median_time_to_flag_s": float(np.median(ttf)) if ttf else None,
        "honest_flagged": f"{sum(g['flagged'] for g in honest)}/{len(honest)}",
        "honest_flags_per_hour": per_hour(sum(g["n_warnings"] + g["n_alerts"] for g in honest), honest_s),
        "segments": segs,
    }


def evaluate(path: str | Path, base: Config, profile: str | None = None, fast: bool = False, save: bool = True,
             model: str | None = None, fuse: bool = False) -> dict:
    s = load_session(path)
    fn = {"honest": evaluate_honest, "behavior": evaluate_behavior}.get(s.meta.get("kind"), evaluate_protocol)
    rep = fn(s, base, profile, fast=fast, model=model, fuse=fuse)
    if save:
        kind = "fused" if fuse else "learned"
        name = f"eval_{kind}_{Path(model).parent.name}.json" if model else ("eval_fast.json" if fast else "eval.json")
        (s.dir / name).write_text(json.dumps(rep, indent=2, default=float))
    return rep


# ---------------------------------------------------------------------------- printing
def _fmt(d: dict) -> str:
    return f"{d['mean']:.2f} (median {d['median']:.2f}, p90 {d['p90']:.2f})" if d.get("n") else "n/a"


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['session']}  (subject {rep['subject']}) ===")
    if rep.get("kind") == "behavior":
        print(f"behaviour segments ({rep['estimator']}): cheating detected {rep['cheat_detected']}, "
              f"median time to first flag {rep['median_time_to_flag_s']} s | honest segments flagged "
              f"{rep['honest_flagged']} ({rep['honest_flags_per_hour']:.0f} flags/h)")
        for g in rep["segments"]:
            mark = ("✓" if g["flagged"] else "✗ missed") if g["label"] == "cheat" else ("✗ FALSE FLAG" if g["flagged"] else "✓")
            first = f"first flag {g['first_flag_s']:>5}s" if g["first_flag_s"] is not None else "no flag        "
            print(f"  {g['task']:18s} {g['label']:6s} {g['seconds']:5.1f}s  {mark:13s} {first}  "
                  f"{g['flag_kinds'] or ''}  glances {g['brief_glances']}  zones {g['zone_time_frac']}")
        return
    if rep.get("kind") == "honest":
        print(f"honest work: {rep['minutes']} min")
        print(f"  ALERTS/hour   {rep['alerts_per_hour']:.1f}    warnings/hour {rep['warnings_per_hour']:.1f}"
              f"    brief glances/hour {rep['brief_glances_per_hour']:.0f}")
        print(f"  flags: {rep['flag_counts'] or 'none'}")
        print(f"  time in zones: {rep['zone_time_frac']}")
        return
    g = rep["gaze"]
    cal = rep["calibration"]
    extra = f", uncalibrated {cal['uncalibrated_err_deg']:.2f}°" if "uncalibrated_err_deg" in cal else ""
    print(f"estimator: {rep.get('estimator')}   calibration: {cal['source']}  (LOPO {cal['lopo_mean_err_deg'] or float('nan'):.2f}°{extra})")
    print(f"gaze error  (deg): {_fmt(g['angular_deg'])}")
    print(f"   interior      : {_fmt(g['angular_deg_interior'])}")
    print(f"   edge band     : {_fmt(g['angular_deg_edge_band'])}")
    print(f"   on screen (cm): {_fmt(g['screen_cm'])}")
    if g["yaw_mae_deg"] is not None:
        print(f"   yaw MAE {g['yaw_mae_deg']:.2f} (bias {g['yaw_bias_deg']:+.2f})   "
              f"pitch MAE {g['pitch_mae_deg']:.2f} (bias {g['pitch_bias_deg']:+.2f})")
    z = rep["zones_frame"]
    print(f"zones (per frame): macro-F1 {z['macro_f1']:.3f}   accuracy {z['accuracy']:.3f}")
    labels = z["labels"]
    w = max(len(l) for l in labels) + 1
    print("  " + "true \\ pred".ljust(w) + " ".join(l[:6].rjust(6) for l in labels) + "   recall  F1")
    for i, l in enumerate(labels):
        pc = z["per_class"][l]
        if pc["support"] == 0:
            continue
        print("  " + l.ljust(w) + " ".join(str(v).rjust(6) for v in z["confusion"][i])
              + f"   {pc['recall']:.2f}  {pc['f1']:.2f}")
    b = rep["off_screen_binary"]
    print(f"off-screen detection: precision {b['precision']:.2f}  recall {b['recall']:.2f}  F1 {b['f1']:.2f}")
    print(f"trials (majority vote): accuracy {rep['trials']['accuracy']:.2f}   {rep['trials']['by_task']}")
    lat = rep["latency_ms"]["landmarks+features"]
    lat_s = f"latency p50 {lat['median']:.1f} ms" if lat.get("n") else "latency n/a (--fast replay)"
    print(f"face detected (on-screen trials): {100 * rep['face_detect_rate_onscreen']:.0f}%   {lat_s}")


def print_summary(reps: list[dict]) -> None:
    proto = [r for r in reps if r.get("kind") not in ("honest", "behavior")]
    beh = [r for r in reps if r.get("kind") == "behavior"]
    if beh:
        segs = [g for r in beh for g in r["segments"]]
        ch = [g for g in segs if g["label"] == "cheat"]
        ho = [g for g in segs if g["label"] == "honest"]
        print(f"\n=== behaviour sessions: cheating detected {sum(g['flagged'] for g in ch)}/{len(ch)} segments, "
              f"honest segments falsely flagged {sum(g['flagged'] for g in ho)}/{len(ho)}")
    honest = [r for r in reps if r.get("kind") == "honest"]
    if len(proto) > 1:
        print("\n=== protocol sessions ===")
        print(f"{'session':34s} {'glasses':7s} {'gaze°':>6s} {'cm':>5s} {'zoneF1':>7s} {'offF1':>6s} {'trial':>6s}")
        for r in proto:
            print(f"{r['session'][:34]:34s} {str(r['glasses']):7s} {r['gaze']['angular_deg'].get('mean', float('nan')):6.2f} "
                  f"{r['gaze']['screen_cm'].get('mean', float('nan')):5.1f} {r['zones_frame']['macro_f1']:7.3f} "
                  f"{r['off_screen_binary']['f1']:6.2f} {r['trials']['accuracy']:6.2f}")
    if honest:
        mins = sum(r["minutes"] for r in honest)
        alerts = sum(r["alerts_per_hour"] * r["minutes"] / 60 for r in honest)
        warns = sum(r["warnings_per_hour"] * r["minutes"] / 60 for r in honest)
        print(f"\n=== honest sessions: {mins:.1f} min total → {alerts * 60 / mins:.1f} alerts/hour, "
              f"{warns * 60 / mins:.1f} warnings/hour (target < 1 false flag/hour)")


def cmd_eval(args) -> None:
    base = Config.load(args.config)
    paths = args.sessions
    if not paths:
        from .data.session import list_sessions
        paths = list_sessions()
        if not paths:
            raise SystemExit("No sessions found in data/own. Record one with `collect` first.")
    reps = []
    for p in paths:
        if args.fuse and not args.model:
            raise SystemExit("--fuse needs --model")
        rep = evaluate(p, base, args.profile, fast=args.fast, model=args.model, fuse=args.fuse)
        print_report(rep)
        reps.append(rep)
    print_summary(reps)


def add_parser(sub) -> None:
    c = sub.add_parser("eval", help="evaluate recorded sessions (default: all in data/own)")
    c.add_argument("sessions", nargs="*")
    c.add_argument("--profile", default=None, help="use this calibration profile instead of the session's own")
    c.add_argument("--fast", action="store_true", help="replay features stored at capture time (no MediaPipe)")
    c.add_argument("--model", default=None, help="learned gaze checkpoint (best.pt) instead of the Phase 0 mapper")
    c.add_argument("--fuse", action="store_true", help="with --model: fuse learned + Phase 0 mapper (recommended)")
    c.set_defaults(fn=cmd_eval)
