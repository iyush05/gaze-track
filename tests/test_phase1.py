import math

import numpy as np
import pytest

from gaze_track.calibration import GazeMapper
from gaze_track.config import Config, ScreenGeometry, ZoneThresholds
from gaze_track.data.normalize import (
    from_eth_pitchyaw,
    denormalize_gaze,
    mediapipe_intrinsics,
    normalization_rotation,
    normalize,
    normalize_gaze,
    to_eth_pitchyaw,
)
from gaze_track.data.protocol import build_protocol, gt_screen_zone
from gaze_track.data.session import SessionWriter, load_session, subject_split
from gaze_track.geometry import angles_to_vector, vector_to_angles
from gaze_track.metrics import angular_error_deg, binary_report, per_hour, zone_report

SCREEN, TH = ScreenGeometry(), ZoneThresholds()


# ------------------------------------------------------------------ protocol
def test_protocol_structure_and_determinism():
    a = build_protocol(SCREEN, TH, n_screen=24, off_repeats=2, seed=5)
    b = build_protocol(SCREEN, TH, n_screen=24, off_repeats=2, seed=5)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
    segs = [t.segment for t in a]
    assert segs.count("calib") == 9 and segs.count("screen") == 24 and segs.count("off") == 8 * 2 + 1
    assert [t.trial_id for t in a] == list(range(len(a)))
    screen = [t for t in a if t.segment == "screen"]
    for t in screen:
        assert 0 < t.u_cm < SCREEN.width_cm and 0 < t.v_cm < SCREEN.height_cm
        assert t.zone == gt_screen_zone(t.u_cm, t.v_cm, SCREEN, TH)
    assert {t.zone for t in screen} == {"ON_SCREEN", "EDGE"}
    near = [t for t in a if t.task.startswith("near_")]
    assert near and all(t.approx and t.zone == "NEAR_OFF" for t in near)


# ------------------------------------------------------------------ metrics
def test_angular_error():
    assert angular_error_deg(10, 5, 10, 5)[0] == pytest.approx(0, abs=1e-6)
    assert angular_error_deg(0, 0, 3, 4)[0] == pytest.approx(5.0, abs=0.05)


def test_zone_report_and_binary():
    t = ["ON_SCREEN"] * 4 + ["FAR_OFF"] * 2
    p = ["ON_SCREEN"] * 3 + ["EDGE"] + ["FAR_OFF", "ON_SCREEN"]
    r = zone_report(t, p, ["ON_SCREEN", "EDGE", "FAR_OFF"])
    assert r["confusion"] == [[3, 1, 0], [0, 0, 0], [1, 0, 1]]
    assert r["per_class"]["FAR_OFF"]["recall"] == 0.5
    # EDGE has no ground-truth support → excluded from macro-F1.
    f_on, f_far = r["per_class"]["ON_SCREEN"]["f1"], r["per_class"]["FAR_OFF"]["f1"]
    assert r["macro_f1"] == pytest.approx((f_on + f_far) / 2)
    b = binary_report([False, True, True], [True, True, False])
    assert (b["tp"], b["fp"], b["fn"]) == (1, 1, 1)
    assert per_hour(2, 1800) == 4.0


# ------------------------------------------------------------------ normalization
K = mediapipe_intrinsics(640, 480)


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_normalization_frontal_is_identity_rotation():
    R = normalization_rotation(np.array([0, 0, 55.0]), np.eye(3))
    assert np.allclose(R, np.eye(3))


@pytest.mark.parametrize("fc", [np.array([5, -3, 50.0]), np.array([-8, 6, 70.0])])
@pytest.mark.parametrize("roll", [0, 20, -35])
def test_normalization_properties(fc, roll):
    head = _rot("z", roll) @ _rot("y", 15)
    R = normalization_rotation(fc, head)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    # Face centre lies on the optical axis of the normalized camera.
    v = R @ fc
    assert v[0] == pytest.approx(0, abs=1e-9) and v[1] == pytest.approx(0, abs=1e-9)
    # Roll removed: head x-axis has no vertical component in the normalized camera.
    assert (R @ head[:, 0])[1] == pytest.approx(0, abs=1e-9)
    # Gaze round trip.
    g = angles_to_vector(0.2, -0.1)
    assert np.allclose(denormalize_gaze(normalize_gaze(g, R), R), g)


def test_normalize_warps_face_center_to_patch_center():
    fc = np.array([6.0, -4.0, 55.0])
    px = K @ fc
    px = px[:2] / px[2]
    img = np.zeros((480, 640, 3), np.uint8)
    import cv2

    cv2.circle(img, (int(round(px[0])), int(round(px[1]))), 3, (255, 255, 255), -1)
    n = normalize(img, K, np.eye(3), fc, roi=112)
    ys, xs = np.nonzero(n.patch[..., 0] > 100)
    assert abs(xs.mean() - 56) < 2 and abs(ys.mean() - 56) < 2


def test_eth_convention_roundtrip_and_sign():
    for pitch, yaw in [(0.1, -0.3), (-0.4, 0.5)]:
        g = from_eth_pitchyaw(pitch, yaw)
        assert np.allclose(to_eth_pitchyaw(g), (pitch, yaw))
        ours_yaw, ours_pitch = vector_to_angles(g)
        assert ours_yaw == pytest.approx(-yaw) and ours_pitch == pytest.approx(-pitch)


# ------------------------------------------------------------------ session + eval plumbing
def test_subject_split_is_deterministic():
    assert subject_split("s01") == subject_split("s01")
    splits = {subject_split(f"s{i:03d}") for i in range(200)}
    assert splits == {"train", "val", "test"}


def _fake_session(tmp_path, kind="protocol"):
    cfg = Config()
    trials = build_protocol(cfg.screen, cfg.zones, n_screen=4, off_repeats=1, seed=1)
    from dataclasses import asdict

    meta = {"kind": kind, "subject": "test", "glasses": "no", "lighting": "normal", "screen": asdict(cfg.screen),
            "zones": asdict(cfg.zones), "camera": {"assumed_vertical_fov_deg": 63.0}}
    w = SessionWriter(tmp_path / "sess", meta)
    rng = np.random.default_rng(0)
    t = 0.0
    for tr in trials:
        if tr.segment == "calib":
            tr.u_cm, tr.v_cm = tr.fx * cfg.screen.width_cm, tr.fy * cfg.screen.height_cm
        for _ in range(3):
            t += 1 / 30
            row = {"t": t, "trial_id": tr.trial_id, "segment": tr.segment, "task": tr.task, "zone_gt": tr.zone,
                   "u_gt": tr.u_cm, "v_gt": tr.v_cm, "approx": int(tr.approx), "n_faces": 0}
            w.add(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), row, None, None)
    w.write_json("trials.json", [tr.to_dict() for tr in trials])
    w.close()
    return tmp_path / "sess", trials


def test_session_roundtrip(tmp_path):
    d, trials = _fake_session(tmp_path)
    s = load_session(d)
    assert len(s.rows) == 3 * len(trials) and s.meta["n_frames"] == len(s.rows)
    assert s.rows[0]["segment"] == "calib" and isinstance(s.rows[0]["t"], float)
    a = s.arrays()
    assert a["transform"].shape == (len(s.rows), 4, 4) and a["landmarks"].shape == (len(s.rows), 478, 3)
    assert s.frame_path(0).exists()


def test_eval_and_export_run_end_to_end_on_faceless_session(tmp_path):
    """Plumbing check with real MediaPipe: noise frames → every frame NO_FACE, no crash."""
    from gaze_track.data.export import export_session
    from gaze_track.evaluate import evaluate_protocol

    d, _ = _fake_session(tmp_path)
    prof = tmp_path / "p.json"
    GazeMapper(calibrated=True).save(prof)
    rep = evaluate_protocol(load_session(d), Config(), profile=str(prof))
    assert set(rep["zones_frame"]["labels"]) >= {"NO_FACE"}
    assert rep["off_screen_binary"]["tp"] == 0 and rep["frames"] > 0
    f = export_session(d, tmp_path / "norm")
    with np.load(f) as z:
        assert len(z["patches"]) == 0 and str(z["subject"]) == "test"


def test_behavior_protocol_labels():
    from gaze_track.data.behavior import build_behavior_protocol

    segs = build_behavior_protocol(side="right")
    assert {g.label for g in segs} == {"honest", "cheat"}
    assert any(g.task == "cheat_notes_right" for g in segs)
    assert sum(g.seconds for g in build_behavior_protocol(short=True)) == pytest.approx(sum(g.seconds for g in segs) / 2)


def test_behavior_eval_per_segment(tmp_path):
    """Faceless frames: every segment raises face_absent → cheat segments 'detected', honest ones
    'falsely flagged'. Checks segmentation, fresh pipeline per segment, and summary fields."""
    from dataclasses import asdict

    from gaze_track.calibration import GazeMapper
    from gaze_track.evaluate import evaluate_behavior

    cfg = Config()
    meta = {"kind": "behavior", "subject": "t", "screen": asdict(cfg.screen), "zones": asdict(cfg.zones),
            "camera": {}}
    w = SessionWriter(tmp_path / "b", meta)
    t, img = 0.0, np.zeros((48, 64, 3), np.uint8)
    for tid, (task, label) in enumerate([("honest_reading", "HONEST"), ("cheat_phone_lap", "CHEAT")]):
        for _ in range(int(4 / 0.1)):  # 4 s at 10 fps
            t += 0.1
            w.add(img, {"t": t, "trial_id": tid, "segment": "behavior", "task": task, "zone_gt": label,
                        "n_faces": 0}, None, None)
        t += 5.0  # setup pause between segments
    w.write_json("trials.json", [])
    w.close()
    prof = tmp_path / "p.json"
    GazeMapper(calibrated=True).save(prof)
    rep = evaluate_behavior(load_session(tmp_path / "b"), cfg, profile=str(prof), fast=True)
    assert [g["task"] for g in rep["segments"]] == ["honest_reading", "cheat_phone_lap"]
    assert rep["cheat_detected"] == "1/1" and rep["honest_flagged"] == "1/1"
    # Fresh pipeline per segment: first flag after ~face_absent_s within each segment, not carried over.
    for g in rep["segments"]:
        assert g["flag_kinds"] == {"face_absent": 1}
        assert cfg.events.face_absent_s - 0.2 <= g["first_flag_s"] <= cfg.events.face_absent_s + 0.3


def test_group_incidents_gap_cap_and_breaks():
    from gaze_track.events import Event
    from gaze_track.report import group_incidents

    ev = lambda t, d=3.0: Event(t, "sustained_off_screen", "warning", d)  # noqa: E731
    flags = [ev(10), ev(18), ev(26), ev(60), ev(64)]
    assert [len(g) for g in group_incidents(flags)] == [3, 2]  # 60-3 - 26 > 10 s → new incident
    dense = [ev(t) for t in range(10, 120, 8)]  # continuous flags every 8 s
    groups = group_incidents(dense)
    assert all(max(e.t for e in g) - min(e.t - 3 for e in g) <= 60 for g in groups) and len(groups) >= 2
    assert [len(g) for g in group_incidents([ev(10), ev(16)], breaks=[14.0])] == [1, 1]


def test_describe_direction():
    from gaze_track.report import describe_direction

    assert describe_direction(20, 0) == "left" and describe_direction(-20, 15) == "down-right"
    assert describe_direction(0, -20) == "up" and describe_direction(2, 2) == "straight ahead"


def test_report_end_to_end_faceless_session(tmp_path):
    """Faceless frames → 'face not visible' alerts → incidents with snapshots and decision controls."""
    from gaze_track.calibration import GazeMapper
    from gaze_track.report import build_report

    from dataclasses import asdict

    cfg = Config()
    meta = {"kind": "honest", "subject": "t", "screen": asdict(cfg.screen), "zones": asdict(cfg.zones), "camera": {}}
    w = SessionWriter(tmp_path / "h", meta)
    img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    for i in range(50):  # 5 s at 10 fps without a face → one "face not visible" alert after 2 s
        w.add(img, {"t": i / 10, "trial_id": 0, "segment": "honest", "task": "work", "zone_gt": "", "n_faces": 0},
              None, None)
    w.write_json("trials.json", [])
    w.close()
    prof = tmp_path / "p.json"
    GazeMapper(calibrated=True).save(prof)
    out = build_report(tmp_path / "h", out=tmp_path / "r.html", profile=str(prof))
    doc = out.read_text()
    assert "Face not visible" in doc and "Incident 1" in doc
    assert 'data:image/jpeg;base64,' in doc  # snapshots embedded (self-contained file)
    assert 'id="dl"' in doc and 'type="radio"' in doc  # reviewer decisions + download
    assert "signals for human review, not verdicts" in doc


def test_evidence_recorder_saves_snapshots_only_around_flags(tmp_path):
    from gaze_track.events import Event
    from gaze_track.evidence import EvidenceRecorder, is_bundle, load_bundle
    from gaze_track.pipeline import FrameResult
    from gaze_track.report import build_report_from_bundle
    from gaze_track.zones import Zone

    rec = EvidenceRecorder(tmp_path / "ev", {"profile": "p", "estimator": "x"}, buffer_s=5.0, snap_fps=10)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    for i in range(300):  # 10 s at 30 fps, one warning at t = 8 s lasting 3 s
        t = i / 30
        evs = [Event(t, "sustained_off_screen", "warning", 3.0, -30.0, 5.0)] if i == 240 else []
        if i == 100:
            evs.append(Event(t, "brief_glance", "info", 0.5))  # info → no snapshots
        rec.on_frame(frame, FrameResult(t, Zone.FAR_OFF if i >= 150 else Zone.ON_SCREEN, Zone.ON_SCREEN, 1,
                                        events=evs))
        # In-memory buffer stays bounded (pruned at each 0.1 s snapshot → ≤ buffer_s + one period).
        assert all(t - b[0] <= 5.0 + 0.1 + 1e-9 for b in rec._buf)
    rec.close(duration_s=10.0)
    meta, frames, events = load_bundle(tmp_path / "ev")
    assert is_bundle(tmp_path / "ev") and len(frames) == 300 and meta["flags"] == 1
    assert len(events) == 1 and events[0]["kind"] == "sustained_off_screen"
    shots = events[0]["snapshots"]
    assert len(shots) == 3 and [s["t"] for s in shots] == sorted(s["t"] for s in shots)
    assert abs(shots[0]["t"] - 5.0) < 0.15 and abs(shots[-1]["t"] - 8.0) < 0.15  # start … flag moment
    assert len(list((tmp_path / "ev" / "snapshots").iterdir())) == 3  # nothing else written
    doc = build_report_from_bundle(tmp_path / "ev").read_text()
    assert "Incident 1" in doc and doc.count("data:image/jpeg") == 3 and "Live session" in doc
