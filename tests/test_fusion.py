import numpy as np
import pytest

pytest.importorskip("torch")

from gaze_track.config import Config  # noqa: E402
from gaze_track.features import GazeFeatures  # noqa: E402
from gaze_track.fusion import FusedEstimator  # noqa: E402
from gaze_track.pipeline import Pipeline  # noqa: E402
from gaze_track.zones import Zone  # noqa: E402


def _synthetic(rng, n=270):
    true = np.c_[rng.uniform(-20, 20, n), rng.uniform(-3, 22, n)]
    base = true + np.c_[rng.normal(0, 1.0, n), rng.normal(0, 3.0, n)]  # good yaw, noisy pitch
    learned = 0.35 * (true + np.c_[rng.normal(0, 3.0, n), rng.normal(0, 1.0, n)])  # compressed, good pitch
    return base, learned, true


def test_fusion_beats_each_input_on_held_out_points():
    rng = np.random.default_rng(0)
    b, l, t = _synthetic(rng)
    groups = np.repeat(np.arange(9), 30)
    fus = FusedEstimator.fit(b, l, t, groups=groups)
    b2, l2, t2 = _synthetic(np.random.default_rng(1))
    pred = np.array([fus.predict(x, y) for x, y in zip(b2, l2)])
    e = lambda p: np.hypot(*(p - t2).T).mean()  # noqa: E731
    lcal = np.c_[l2, np.ones(len(l2))] @ fus.learned_cal.A.T
    assert e(pred) < 0.8 * min(e(b2), e(lcal))
    assert "lopo_mean_err_deg" in fus.stats


def test_fusion_falls_back_and_roundtrips():
    fus = FusedEstimator()
    assert fus.predict((3.0, 4.0), None) == (3.0, 4.0)
    assert fus.predict((2.0, 2.0), (2.0, 2.0)) == pytest.approx((2.0, 2.0))  # prior = equal blend
    fus2 = FusedEstimator.from_dict(fus.to_dict())
    assert np.allclose(fus2.W, fus.W) and np.allclose(fus2.learned_cal.A, fus.learned_cal.A)


def test_pipeline_offscreen_veto_trusts_mapper():
    """Fused gaze says 'centre of screen' but the mapper says far right → report off-screen."""
    cfg = Config()
    pipe = Pipeline(cfg, landmarker=False)
    feats = GazeFeatures(head_yaw=0, head_pitch=0, head_roll=0, eye_h=-0.4, eye_v=0.05, ear=0.3,
                         origin_cm=np.array([0.0, 8.0, 50.0]), face_width_px=200)
    by, _ = pipe.mapper.predict(feats.vector())
    assert by < -40  # mapper: far toward the user's right
    for i in range(20):
        r = pipe.step(feats, 1, i / 30, gaze=(0.0, 2.5, feats.origin_cm))  # screen centre from this eye
    assert r.raw_zone in (Zone.NEAR_OFF, Zone.FAR_OFF)
    # Without a fused gaze override there is no veto path; the mapper drives the zone directly.
    pipe2 = Pipeline(cfg, landmarker=False)
    for i in range(20):
        r2 = pipe2.step(feats, 1, i / 30)
    assert r2.raw_zone in (Zone.NEAR_OFF, Zone.FAR_OFF)


def test_pipeline_fused_gaze_used_when_mapper_agrees_on_screen():
    cfg = Config()
    pipe = Pipeline(cfg, landmarker=False)
    feats = GazeFeatures(head_yaw=0, head_pitch=0, head_roll=0, eye_h=0.0, eye_v=0.0, ear=0.3,
                         origin_cm=np.array([0.0, 8.0, 50.0]), face_width_px=200)
    for i in range(20):
        r = pipe.step(feats, 1, i / 30, gaze=(0.0, 2.5, feats.origin_cm))
    assert r.raw_zone == Zone.ON_SCREEN and r.gaze_pitch == pytest.approx(2.5, abs=0.1)


def test_face_counter_runs_periodically_and_reports_latest(monkeypatch):
    from gaze_track.landmarks import FaceCounter

    fc = FaceCounter("models/blaze_face_short_range.tflite", every=3)
    calls = []

    def fake_detect(frame, ts):
        calls.append(ts)
        return 2  # a second person appears

    monkeypatch.setattr(fc, "_detect", fake_detect)
    frame = np.zeros((48, 64, 3), np.uint8)
    seen = []
    for i in range(9):
        seen.append(fc.count(frame, i * 33))
        if fc._pending is not None:  # wait for the worker deterministically (no timing assumptions)
            fc._pending.result(timeout=5)
    fc.close()
    assert len(calls) == 3  # every 3rd frame
    assert seen[0] == 0 and seen[-1] == 2  # result is picked up on a later frame


def test_multi_face_count_gates_to_multi_face():
    cfg = Config()
    pipe = Pipeline(cfg, landmarker=False)
    feats = GazeFeatures(head_yaw=0, head_pitch=0, head_roll=0, eye_h=0.0, eye_v=0.0, ear=0.3,
                         origin_cm=np.array([0.0, 8.0, 50.0]), face_width_px=200)
    for i in range(20):
        r = pipe.step(feats, 2, i / 30)
    assert r.zone == Zone.MULTI_FACE


def test_offscreen_veto_outranks_keyboard():
    """Fused gaze says 'keyboard' but the mapper sees gaze far below the keyboard band (phone in
    lap) → report off-screen (s01 behaviour session 2 was missed before this rule)."""
    cfg = Config()
    pipe = Pipeline(cfg, landmarker=False)
    feats = GazeFeatures(head_yaw=0, head_pitch=0, head_roll=0, eye_h=0.0, eye_v=0.4, ear=0.3,
                         origin_cm=np.array([0.0, 8.0, 50.0]), face_width_px=200)
    assert pipe.mapper.predict(feats.vector())[1] > 45  # mapper: steeply down (lap)
    for i in range(20):  # fused: ~10° below the bottom edge → keyboard band
        r = pipe.step(feats, 1, i / 30, gaze=(0.0, 23.0, feats.origin_cm))
    assert r.raw_zone in (Zone.NEAR_OFF, Zone.FAR_OFF)


def test_keyboard_band_limit():
    from gaze_track.calibration import target_angles
    from gaze_track.zones import classify_point

    cfg = Config()
    eye = np.array([0.0, 8.0, 40.0])
    _, bottom = target_angles(cfg.screen, eye, cfg.screen.width_cm / 2, cfg.screen.height_cm)
    from gaze_track.calibration import project_to_screen

    for below, want in [(15.0, Zone.DOWN_KEYBOARD), (35.0, None)]:
        pitch = bottom + below
        uv = project_to_screen(cfg.screen, eye, 0.0, pitch)
        z = classify_point(uv, pitch, eye, 0, 0, cfg.screen, cfg.zones)
        assert (z == want) if want else (z in (Zone.NEAR_OFF, Zone.FAR_OFF))


def _textured_face(rng, size=200):
    img = np.full((480, 640, 3), 120, np.uint8)
    face = rng.integers(40, 220, (size // 4, size // 4), dtype=np.uint8)
    import cv2

    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_NEAREST)  # edges at 4 px blocks
    img[140:140 + size, 220:220 + size] = face[..., None]
    lm = np.zeros((478, 3), np.float32)
    lm[:, 0] = rng.uniform(222, 218 + size, 478)
    lm[:, 1] = rng.uniform(142, 138 + size, 478)
    lm[0, :2], lm[1, :2] = (220, 140), (220 + size - 1, 140 + size - 1)  # bbox corners
    return img, lm


def test_quality_flags_dark_and_blur_but_not_clean():
    import cv2

    from gaze_track.quality import QualityMonitor, measure

    cfg = Config()
    img, lm = _textured_face(np.random.default_rng(0))
    qm = QualityMonitor(cfg.zones)
    assert qm.classify(*measure(img, lm)).status == "ok"
    dark = (img.astype(np.float32) * 0.1).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 6)
    assert qm.classify(*measure(dark, lm)).status == "too_dark"
    assert qm.classify(*measure(blurred, lm)).status == "blurry"


def test_unusable_frames_report_unknown_and_warn_after_timeout():
    cfg = Config()
    pipe = Pipeline(cfg, landmarker=False)
    feats = GazeFeatures(head_yaw=0, head_pitch=0, head_roll=0, eye_h=0.0, eye_v=0.0, ear=0.3,
                         origin_cm=np.array([0.0, 8.0, 50.0]), face_width_px=200)
    evs = []
    for i in range(int((cfg.events.unreliable_s + 1) * 30)):
        r = pipe.step(feats, 1, i / 30, unusable=True)
        evs += r.events
    assert r.zone == Zone.UNKNOWN
    assert [e.kind for e in evs if e.severity != "info"] == ["tracking_unreliable"]
