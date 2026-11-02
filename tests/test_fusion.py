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
