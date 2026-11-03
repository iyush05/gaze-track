"""Per-frame pipeline: landmarks → features → gaze angles → screen point → zone → events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .calibration import GazeMapper, project_to_screen
from .config import Config
from .events import Event, EventEngine
from .features import GazeFeatures, extract
from .filters import OneEuro
from .landmarks import FaceLandmarkerWrapper, LandmarkResult
from .zones import OFF_ZONES, Zone, ZoneSmoother, classify_point

MIN_FACE_WIDTH_PX = 60.0
MAX_ROLL_DEG = 35.0


@dataclass
class FrameResult:
    t: float
    zone: Zone  # smoothed (reported)
    raw_zone: Zone
    n_faces: int
    features: GazeFeatures | None = None
    gaze_yaw: float | None = None
    gaze_pitch: float | None = None
    point_uv: tuple[float, float] | None = None
    events: list[Event] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    landmarks: LandmarkResult | None = None
    quality: object | None = None  # quality.Quality

    def row(self) -> dict:
        """Flat per-frame record for CSV logging (seeds Phase 1 datasets)."""
        f = self.features
        return {
            "t": round(self.t, 4),
            "zone": self.zone.value,
            "raw_zone": self.raw_zone.value,
            "n_faces": self.n_faces,
            "head_yaw": f and round(f.head_yaw, 3),
            "head_pitch": f and round(f.head_pitch, 3),
            "head_roll": f and round(f.head_roll, 3),
            "eye_h": f and round(f.eye_h, 5),
            "eye_v": f and round(f.eye_v, 5),
            "ear": f and round(f.ear, 4),
            "origin_x": f and round(float(f.origin_cm[0]), 2),
            "origin_y": f and round(float(f.origin_cm[1]), 2),
            "origin_z": f and round(float(f.origin_cm[2]), 2),
            "gaze_yaw": self.gaze_yaw and round(self.gaze_yaw, 3),
            "gaze_pitch": self.gaze_pitch and round(self.gaze_pitch, 3),
            "u_cm": self.point_uv and round(self.point_uv[0], 2),
            "v_cm": self.point_uv and round(self.point_uv[1], 2),
            "lat_total_ms": round(self.timings_ms.get("total", 0.0), 3),
        }


def eye_closed_threshold(cfg: Config, ear_baseline: float | None) -> float:
    """Relative to the user's open-eye EAR from calibration when known: looking down at the
    keyboard lowers the lids (EAR 0.17 for s01 vs 0.29 open, 0.02 closed), so a fixed threshold
    confuses the two."""
    if ear_baseline:
        return cfg.zones.eye_closed_rel * ear_baseline
    return cfg.zones.eye_closed_ratio


def gate_zone(cfg: Config, feats: GazeFeatures | None, n_faces: int, ear_baseline: float | None = None,
              neutral_pitch: float = 0.0) -> Zone | None:
    """Zones decided before any gaze estimate: face count, unusable face, closed eyes / looking down.
    Returns None when the frame should go on to gaze estimation."""
    if n_faces == 0:
        return Zone.NO_FACE
    if n_faces > 1:
        return Zone.MULTI_FACE
    assert feats is not None
    if feats.face_width_px < MIN_FACE_WIDTH_PX or abs(feats.head_roll) > MAX_ROLL_DEG:
        return Zone.UNKNOWN
    if feats.ear < eye_closed_threshold(cfg, ear_baseline):
        # Lids low with the head tilted down = looking down past the keyboard (lap, desk), where the
        # iris is hidden so gaze can't be estimated; truly closed eyes keep a neutral head pitch.
        if feats.head_pitch - neutral_pitch > cfg.zones.look_down_head_pitch_deg:
            return Zone.FAR_OFF
        return Zone.EYES_CLOSED
    return None


def gaze_zone(
    cfg: Config,
    feats: GazeFeatures,
    yaw: float,
    pitch: float,
    origin: np.ndarray,
    neutral: tuple[float, float],
    current: Zone | None,
) -> tuple[Zone, tuple[float, float] | None]:
    """Gaze angles → screen point → zone (shared by live pipeline and offline evaluation)."""
    uv = project_to_screen(cfg.screen, origin, yaw, pitch)
    zone = classify_point(
        uv, pitch, origin, feats.head_yaw - neutral[0], feats.head_pitch - neutral[1],
        cfg.screen, cfg.zones, current=current,
    )
    return zone, uv


class Pipeline:
    def __init__(self, cfg: Config, mapper: GazeMapper | None = None, landmarker: bool = True,
                 fusion=None, learned=None):
        """fusion: FusedEstimator; learned: LearnedGaze (both from a profile calibrated with --model)."""
        self.cfg = cfg
        self.mapper = mapper or GazeMapper()
        self.fusion, self.learned = fusion, learned
        from .quality import QualityMonitor

        self.quality = QualityMonitor(cfg.zones)
        # Separate smoothing for the feature mapper's estimate used by the off-screen veto.
        self.fb_yaw = OneEuro(min_cutoff=1.5, beta=0.02)
        self.fb_pitch = OneEuro(min_cutoff=1.5, beta=0.02)
        self.counter = None
        if landmarker:
            faces = cfg.max_faces
            if cfg.face_count_every > 0:
                from .landmarks import FaceCounter

                self.counter = FaceCounter(cfg.face_detector_path, every=cfg.face_count_every)
                faces = 1
            self.landmarker = FaceLandmarkerWrapper(cfg.model_path, max_faces=faces, blendshapes=cfg.blendshapes)
        else:
            self.landmarker = None
        self.smoother = ZoneSmoother(cfg.zones.min_dwell_s)
        self.events = EventEngine(cfg.events)
        # Stronger smoothing (min_cutoff 0.5–0.8) gave no fewer zone flips on recorded data and
        # delays fast glances, so keep it light; spatial hysteresis handles boundary jitter.
        self.f_yaw = OneEuro(min_cutoff=1.5, beta=0.02)
        self.f_pitch = OneEuro(min_cutoff=1.5, beta=0.02)
        # Neutral head pose (mean during calibration): head-turn limits are measured from it.
        self.neutral_yaw, self.neutral_pitch = self.mapper.stats.get("neutral_head", [0.0, 0.0])
        self.f_origin = [OneEuro(min_cutoff=0.5, beta=0.0) for _ in range(3)]

    def landmarks_and_features(self, frame: np.ndarray, t: float):
        """Shared by run and calibration: returns (LandmarkResult, GazeFeatures|None, timings)."""
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        lm = self.landmarker.process(frame, int(t * 1000))
        t1 = time.perf_counter()
        feats = extract(lm.faces[0], lm.image_size) if lm.faces else None
        t2 = time.perf_counter()
        tm["landmarks"] = (t1 - t0) * 1e3
        tm["features"] = (t2 - t1) * 1e3
        return lm, feats, tm

    def learned_gaze(self, frame: np.ndarray, lm, feats) -> tuple[float, float, np.ndarray] | None:
        """Fused (yaw°, pitch°, origin) for a single-face frame, or None to use the mapper alone."""
        if self.fusion is None or self.learned is None or feats is None or len(lm.faces) != 1:
            return None
        f0 = lm.faces[0]
        n = self.learned.normalize_frame(frame, f0.transform, f0.landmarks_px)
        learned_raw = None
        if n is not None:
            pred = self.learned.predict_patches(n[0][None])[0]
            learned_raw = self.learned.to_camera_angles(pred, n[1])
        y, p = self.fusion.predict(self.mapper.predict(feats.vector()), learned_raw)
        return y, p, feats.origin_cm

    def process(self, frame: np.ndarray, t: float) -> FrameResult:
        t_start = time.perf_counter()
        lm, feats, tm = self.landmarks_and_features(frame, t)
        n_faces = len(lm.faces)
        if self.counter is not None:
            t_c = time.perf_counter()
            extra = self.counter.count(frame, int(t * 1000))
            tm["face_count"] = (time.perf_counter() - t_c) * 1e3
            if extra >= 2:  # the detector only ever raises the count to flag a second person
                n_faces = extra
        q = self.quality.update(frame, lm.faces[0].landmarks_px) if n_faces == 1 else None
        t_l = time.perf_counter()
        gaze = self.learned_gaze(frame, lm, feats) if (n_faces == 1 and not (q and q.unusable)) else None
        if gaze is not None:
            tm["learned"] = (time.perf_counter() - t_l) * 1e3
        r = self.step(feats, n_faces, t, gaze=gaze, unusable=bool(q and q.unusable))
        r.quality = q
        r.landmarks = lm
        r.timings_ms = tm | {"gaze+zone+events": r.timings_ms["gaze+zone+events"],
                             "total": (time.perf_counter() - t_start) * 1e3}
        return r

    def step(self, feats: GazeFeatures | None, n_faces: int, t: float,
             gaze: tuple[float, float, np.ndarray] | None = None, unusable: bool = False) -> FrameResult:
        """Everything after landmark extraction (also used to replay stored features offline).
        ``gaze`` = (yaw°, pitch°, origin_cm) from another estimator overrides the feature mapper."""
        t2 = time.perf_counter()

        yaw = pitch = None
        uv = None
        pre = gate_zone(self.cfg, feats, n_faces, self.mapper.stats.get("ear_baseline"), self.neutral_pitch)
        if pre is None and unusable:  # image too dark / blurry: don't guess (see quality.py)
            pre = Zone.UNKNOWN
        if pre is not None:
            raw = pre
        else:
            assert feats is not None
            if gaze is not None:
                y_raw, p_raw, origin_raw = gaze
            else:
                y_raw, p_raw = self.mapper.predict(feats.vector())
                origin_raw = feats.origin_cm
            yaw, pitch = self.f_yaw(y_raw, t), self.f_pitch(p_raw, t)
            origin = np.array([f(float(c), t) for f, c in zip(self.f_origin, origin_raw)])
            neutral = (self.neutral_yaw, self.neutral_pitch)
            raw, uv = gaze_zone(self.cfg, feats, yaw, pitch, origin, neutral, self.smoother.current)
            if gaze is not None:
                # Off-screen veto: the fused blend is fitted on on-screen dots and compresses gaze far
                # outside the screen (s01: off-screen recall 0.63 fused vs 0.81 mapper). If the feature
                # mapper alone says off-screen / keyboard, trust it.
                # Priority: off-screen > keyboard > on-screen. The learned model also flattens steep
                # downward gaze, so a phone in the lap read as "keyboard" (s01 behaviour session 2).
                by, bp = self.mapper.predict(feats.vector())
                by, bp = self.fb_yaw(by, t), self.fb_pitch(bp, t)
                if raw not in OFF_ZONES:
                    zb, uvb = gaze_zone(self.cfg, feats, by, bp, origin, neutral, self.smoother.current)
                    if zb in OFF_ZONES or (zb == Zone.DOWN_KEYBOARD and raw != Zone.DOWN_KEYBOARD):
                        raw, uv, yaw, pitch = zb, uvb, by, bp
        if n_faces == 0:
            self.f_yaw.reset()
            self.f_pitch.reset()

        zone = self.smoother.update(raw, t)
        head_yaw = (feats.head_yaw - self.neutral_yaw) if (feats is not None and n_faces == 1) else None
        evs = self.events.update(t, zone, yaw, pitch, head_yaw)
        tm = {"gaze+zone+events": (time.perf_counter() - t2) * 1e3}
        return FrameResult(t, zone, raw, n_faces, feats, yaw, pitch, uv, evs, tm)

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        if self.counter is not None:
            self.counter.close()
