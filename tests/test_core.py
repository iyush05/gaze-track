import math

import numpy as np
import pytest

from gaze_track.calibration import (
    GazeMapper,
    calibration_targets,
    project_to_screen,
    target_angles,
    target_cm,
    window_top_offset_cm,
)
from gaze_track.config import EventThresholds, ScreenGeometry, ZoneThresholds
from gaze_track.events import EventEngine
from gaze_track.filters import OneEuro
from gaze_track.geometry import GL_TO_CV, angles_to_vector, head_angles_deg, intersect_z0, vector_to_angles
from gaze_track.zones import Zone, ZoneSmoother, classify_point

SCREEN = ScreenGeometry()
TH = ZoneThresholds()
EYE = np.array([0.0, 8.0, 55.0])  # eyes ~8 cm below camera height, 55 cm away


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (0.3, -0.2), (-0.7, 0.4), (1.2, 0.1)])
def test_angle_roundtrip(yaw, pitch):
    y2, p2 = vector_to_angles(angles_to_vector(yaw, pitch))
    assert y2 == pytest.approx(yaw) and p2 == pytest.approx(pitch)


def test_looking_at_camera_hits_origin():
    d = -EYE / np.linalg.norm(EYE)
    hit = intersect_z0(EYE, d)
    assert np.allclose(hit, 0, atol=1e-9)


def test_ray_away_from_screen_is_none():
    assert intersect_z0(EYE, np.array([0, 0, 1.0])) is None


def test_screen_camera_roundtrip():
    for u, v in [(0, 0), (10, 5), (SCREEN.width_cm, SCREEN.height_cm)]:
        x, y, _ = SCREEN.screen_to_camera(u, v)
        assert SCREEN.camera_to_screen(x, y) == pytest.approx((u, v))


def test_screen_mirroring():
    # The user's right (u = width) is camera -x.
    x, _, _ = SCREEN.screen_to_camera(SCREEN.width_cm, 0)
    assert x < 0


def test_target_angles_project_back():
    for u, v in [(5, 5), (14.5, 9.4), (25, 17), (-10, 3), (40, 30)]:
        yaw, pitch = target_angles(SCREEN, EYE, u, v)
        assert project_to_screen(SCREEN, EYE, yaw, pitch) == pytest.approx((u, v), abs=1e-6)


def test_head_angles_frontal_and_turned():
    I = np.eye(3)
    fwd = GL_TO_CV @ (I @ [0, 0, 1.0])
    up = GL_TO_CV @ (I @ [0, 1.0, 0])
    assert head_angles_deg(fwd, up) == pytest.approx((0, 0, 0), abs=1e-9)
    # Rotate the face 30° about the vertical axis (GL frame).
    a = math.radians(30)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    yaw, pitch, roll = head_angles_deg(GL_TO_CV @ (Ry @ [0, 0, 1.0]), GL_TO_CV @ (Ry @ [0, 1.0, 0]))
    assert abs(yaw) == pytest.approx(30, abs=1e-6) and pitch == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------ calibration
def test_window_top_offset():
    assert window_top_offset_cm(SCREEN, 2560, 1664) == pytest.approx(0, abs=0.02)
    off = window_top_offset_cm(SCREEN, 2560, 1600)
    assert 0.5 < off < 0.9  # notch band on a MacBook Air
    u, v = target_cm(SCREEN, 0.0, 0.0, off)
    assert (u, v) == pytest.approx((0, off))


def test_calibration_targets_cover_grid():
    pts = calibration_targets(9)
    assert len(pts) == 9 and len(set(pts)) == 9


def test_mapper_recovers_synthetic_linear_model():
    rng = np.random.default_rng(0)
    # Ground truth that differs from the prior: eye gain 120 + offsets + slight cross-talk.
    from gaze_track.calibration import HEAD_PRIOR

    hp = HEAD_PRIOR
    true_W = np.array([[hp, 0.0, 120.0, 0.0, 0.05, 2.0], [0.0, hp, 5.0, 130.0, 0.0, 14.0]])
    groups = np.repeat(np.arange(9), 20)
    eye_h = np.repeat(np.linspace(-0.12, 0.12, 9), 20) + rng.normal(0, 0.005, 180)
    eye_v = np.repeat(np.tile([-0.05, 0.0, 0.05], 3), 20) + rng.normal(0, 0.005, 180)
    head = rng.normal(0, 3, (180, 2))
    roll = rng.normal(0, 2, 180)
    X = np.c_[head[:, 0], head[:, 1], eye_h, eye_v, roll]
    Y = np.c_[X, np.ones(180)] @ true_W.T + rng.normal(0, 0.3, (180, 2))
    m = GazeMapper.fit(X, Y, groups=groups)
    assert m.stats["train_mean_err_deg"] < 1.0
    assert m.stats["lopo_mean_err_deg"] < 2.0


def test_still_head_calibration_keeps_head_turns_physical():
    """User keeps the head still during calibration (tiny head-pose variance, correlated noise).
    A later head turn must move the predicted gaze by the physical prior (not a wild fitted weight)."""
    rng = np.random.default_rng(3)
    n = 180
    eye_h = np.repeat(np.linspace(-0.08, 0.08, 9), 20)
    eye_v = np.repeat(np.tile([-0.02, 0.0, 0.02], 3), 20)
    head = rng.normal(0, 0.2, (n, 2))
    roll = rng.normal(0, 0.2, n)
    X = np.c_[head[:, 0], head[:, 1], eye_h + rng.normal(0, 0.01, n), eye_v + rng.normal(0, 0.005, n), roll]
    Y = np.c_[eye_h * 230 + rng.normal(0, 2, n), eye_v * 500 + 8 + rng.normal(0, 2, n)]
    m = GazeMapper.fit(X, Y, groups=np.repeat(np.arange(9), 20))
    from gaze_track.calibration import HEAD_PRIOR

    assert m.W[0, 0] * 30 == pytest.approx(30 * HEAD_PRIOR, abs=3)
    assert abs(m.W[1, 0] * 30) < 3
    assert abs(m.W[0, 4] * 20) < 3 and abs(m.W[1, 4] * 20) < 3


def test_mapper_save_load(tmp_path):
    m = GazeMapper(calibrated=True, stats={"x": 1})
    m.save(tmp_path / "p.json")
    m2 = GazeMapper.load(tmp_path / "p.json")
    assert np.allclose(m.W, m2.W) and m2.calibrated


# ------------------------------------------------------------------ zones
def _zone_at(u, v, head_yaw=0.0, head_pitch=0.0):
    yaw, pitch = target_angles(SCREEN, EYE, u, v)
    uv = project_to_screen(SCREEN, EYE, yaw, pitch)
    return classify_point(uv, pitch, EYE, head_yaw, head_pitch, SCREEN, TH)


def test_zones():
    W, H = SCREEN.width_cm, SCREEN.height_cm
    assert _zone_at(W / 2, H / 2) == Zone.ON_SCREEN
    assert _zone_at(1.0, H / 2) == Zone.EDGE
    assert _zone_at(-1.0, H / 2) == Zone.EDGE
    assert _zone_at(-8.0, H / 2) == Zone.NEAR_OFF
    assert _zone_at(W + 10, 2) == Zone.NEAR_OFF
    assert _zone_at(-40.0, H / 2) == Zone.FAR_OFF
    assert _zone_at(W / 2, H + 10) == Zone.DOWN_KEYBOARD
    assert _zone_at(W / 2, H / 2, head_yaw=55) == Zone.FAR_OFF
    assert classify_point(None, 0, EYE, 0, 0, SCREEN, TH) == Zone.FAR_OFF


def test_zone_smoother_hysteresis():
    s = ZoneSmoother(0.15)
    assert s.update(Zone.ON_SCREEN, 0.0) == Zone.UNKNOWN
    assert s.update(Zone.ON_SCREEN, 0.2) == Zone.ON_SCREEN
    assert s.update(Zone.FAR_OFF, 0.25) == Zone.ON_SCREEN  # a single-frame blip is ignored
    assert s.update(Zone.ON_SCREEN, 0.28) == Zone.ON_SCREEN
    s.update(Zone.FAR_OFF, 0.30)
    assert s.update(Zone.FAR_OFF, 0.50) == Zone.FAR_OFF


# ------------------------------------------------------------------ filter
def test_one_euro_reduces_jitter():
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(1)
    raw = 10 + rng.normal(0, 1, 300)
    out = [f(x, i / 30) for i, x in enumerate(raw)]
    assert np.std(out[50:]) < 0.5 * np.std(raw[50:])


# ------------------------------------------------------------------ events
def _feed(engine, seq, dt=1 / 30):
    """seq: list of (zone, seconds, yaw, pitch). Returns all events."""
    t, out = 0.0, []
    for zone, secs, yaw, pitch in seq:
        for _ in range(int(round(secs / dt))):
            out += engine.update(t, zone, yaw, pitch)
            t += dt
    return out


def test_sustained_off_screen():
    ev = _feed(EventEngine(EventThresholds()), [(Zone.ON_SCREEN, 1, 0, 10), (Zone.FAR_OFF, 4, 40, 10), (Zone.ON_SCREEN, 1, 0, 10)])
    kinds = [e.kind for e in ev]
    assert kinds.count("sustained_off_screen") == 1 and "brief_glance" not in kinds


def test_repeated_directional_glance():
    seq = []
    for _ in range(4):
        seq += [(Zone.ON_SCREEN, 3, 0, 10), (Zone.NEAR_OFF, 0.8, -35, 25)]
    seq += [(Zone.ON_SCREEN, 1, 0, 10)]
    ev = _feed(EventEngine(EventThresholds()), seq)
    kinds = [e.kind for e in ev]
    assert kinds.count("brief_glance") == 4
    assert kinds.count("repeated_directional_glance") == 1


def test_scattered_glances_not_repeated():
    seq = []
    for yaw, pitch in [(-35, 25), (35, -10), (0, -30), (40, 30)]:
        seq += [(Zone.ON_SCREEN, 3, 0, 10), (Zone.NEAR_OFF, 0.8, yaw, pitch)]
    seq += [(Zone.ON_SCREEN, 1, 0, 10)]
    assert "repeated_directional_glance" not in [e.kind for e in _feed(EventEngine(EventThresholds()), seq)]


def test_face_absent_and_multi_face():
    ev = _feed(EventEngine(EventThresholds()), [(Zone.NO_FACE, 3, None, None), (Zone.MULTI_FACE, 1, None, None)])
    kinds = [e.kind for e in ev]
    assert kinds == ["face_absent", "multiple_faces"]


def test_spatial_hysteresis_stops_boundary_flicker():
    from dataclasses import replace

    TH = replace(ZoneThresholds(), edge_deg=0.0, top_extra_deg=0.0)  # fixed 2.5 cm band
    H = SCREEN.height_cm
    # Jitter ±1 cm around the EDGE/NEAR_OFF boundary (2.5 cm outside the left edge).
    zones = set()
    cur = Zone.EDGE
    for du in [-0.9, 0.8, -0.7, 0.9, -1.0, 0.6]:
        u = -TH.edge_cm + du
        yaw, pitch = target_angles(SCREEN, EYE, u, H / 2)
        uv = project_to_screen(SCREEN, EYE, yaw, pitch)
        cur = classify_point(uv, pitch, EYE, 0, 0, SCREEN, TH, current=cur)
        zones.add(cur)
    assert zones == {Zone.EDGE}
    # A clear move past the boundary + hysteresis still switches.
    yaw, pitch = target_angles(SCREEN, EYE, -(TH.edge_cm + TH.hysteresis_cm + 1), H / 2)
    uv = project_to_screen(SCREEN, EYE, yaw, pitch)
    assert classify_point(uv, pitch, EYE, 0, 0, SCREEN, TH, current=Zone.EDGE) == Zone.NEAR_OFF


def test_flicker_does_not_split_a_glance():
    # One 2 s look off-screen with two 0.2 s flickers back to EDGE → a single glance, no alert.
    seq = [(Zone.ON_SCREEN, 1, 0, 10)]
    for _ in range(3):
        seq += [(Zone.NEAR_OFF, 0.6, -30, 5), (Zone.EDGE, 0.2, -25, 5)]
    seq += [(Zone.ON_SCREEN, 2, 0, 10)]
    ev = _feed(EventEngine(EventThresholds()), seq)
    kinds = [e.kind for e in ev]
    assert "repeated_directional_glance" not in kinds
    assert kinds.count("brief_glance") + kinds.count("sustained_off_screen") <= 1


def test_edge_band_scales_with_distance_and_top_tolerance():
    from dataclasses import replace

    th = replace(ZoneThresholds(), edge_deg=6.0, top_extra_deg=10.0)
    W, H = SCREEN.width_cm, SCREEN.height_cm
    for dist, u_out in [(35.0, 3.0), (55.0, 5.0)]:  # band = dist·tan(6°) = 3.7 / 5.8 cm
        eye = np.array([0.0, 8.0, dist])
        yaw, pitch = target_angles(SCREEN, eye, -u_out, H / 2)
        uv = project_to_screen(SCREEN, eye, yaw, pitch)
        assert classify_point(uv, pitch, eye, 0, 0, SCREEN, th) == Zone.EDGE
    # 5 cm above the screen at 35 cm is within the top tolerance; 5 cm to the side is not.
    eye = np.array([0.0, 8.0, 35.0])
    for (u, v), want in [((W / 2, -5.0), Zone.EDGE), ((-5.0 - 3.7, H / 2), Zone.NEAR_OFF)]:
        yaw, pitch = target_angles(SCREEN, eye, u, v)
        uv = project_to_screen(SCREEN, eye, yaw, pitch)
        assert classify_point(uv, pitch, eye, 0, 0, SCREEN, th) == want


def test_eye_closed_threshold_relative_to_baseline():
    from gaze_track.config import Config
    from gaze_track.pipeline import eye_closed_threshold

    cfg = Config()
    t = eye_closed_threshold(cfg, 0.285)  # s01's open-eye EAR
    assert 0.02 < t < 0.17  # closed (0.02) below, looking at the keyboard (0.17) above
    assert eye_closed_threshold(cfg, None) == cfg.zones.eye_closed_ratio


def test_16_point_calibration_reaches_close_to_edges():
    pts = calibration_targets(16)
    assert len(pts) == 16 and len(set(pts)) == 16
    xs = sorted({p[0] for p in pts})
    assert xs[0] == pytest.approx(0.04) and xs[-1] == pytest.approx(0.96)
    assert calibration_targets(9)[0][0] == pytest.approx(0.08)  # collect protocol unchanged


def test_head_turn_events():
    from gaze_track.events import EventEngine

    th = EventThresholds()
    eng = EventEngine(th)
    ev, t = [], 0.0

    def run(secs, head_yaw, zone=Zone.ON_SCREEN):
        nonlocal t
        for _ in range(int(secs * 30)):
            ev.extend(eng.update(t, zone, 0.0, 10.0, head_yaw))
            t += 1 / 30

    run(2.0, 2.0)
    run(1.0, 35.0, Zone.FAR_OFF)  # too short → nothing
    run(2.0, 2.0)
    run(2.0, 40.0, Zone.FAR_OFF)  # ≥1.5 s → warning
    run(3.0, 2.0)
    run(2.0, -38.0, Zone.FAR_OFF)  # second turn within 60 s → alert
    run(1.0, 2.0)
    kinds = [e.kind for e in ev if e.severity != "info"]
    assert kinds.count("head_turned_away") == 2 and kinds.count("repeated_head_turns") == 1


def test_upward_look_needs_longer_unless_head_turned():
    from gaze_track.events import EventEngine

    th = EventThresholds()
    up = th.up_pitch_deg - 10
    eng = EventEngine(th)
    ev = [e for i in range(int(5 * 30)) for e in eng.update(i / 30, Zone.FAR_OFF, 0.0, up, 2.0)]
    assert not [e for e in ev if e.kind == "sustained_off_screen"]  # 5 s looking up, head forward
    eng = EventEngine(th)
    ev = [e for i in range(int(5 * 30)) for e in eng.update(i / 30, Zone.FAR_OFF, 0.0, up, 35.0)]
    assert [e for e in ev if e.kind == "sustained_off_screen"]  # same, but head turned → flagged
