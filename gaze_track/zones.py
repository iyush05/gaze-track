"""Per-frame zone classification and time-based hysteresis."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from .calibration import target_angles
from .config import ScreenGeometry, ZoneThresholds


class Zone(str, Enum):
    ON_SCREEN = "ON_SCREEN"
    EDGE = "EDGE"
    NEAR_OFF = "NEAR_OFF"
    FAR_OFF = "FAR_OFF"
    DOWN_KEYBOARD = "DOWN_KEYBOARD"
    EYES_CLOSED = "EYES_CLOSED"
    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"
    UNKNOWN = "UNKNOWN"


OFF_ZONES = {Zone.NEAR_OFF, Zone.FAR_OFF}

# How far below the bottom edge (in gaze pitch) still counts as "looking at the keyboard".
KEYBOARD_PITCH_DEG = 35.0


def classify_point(
    uv: tuple[float, float] | None,
    gaze_pitch_deg: float,
    origin_cm: np.ndarray,
    head_yaw_deg: float,
    head_pitch_deg: float,
    screen: ScreenGeometry,
    th: ZoneThresholds,
    current: Zone | None = None,
) -> Zone:
    """Classify a gaze point. ``head_*`` angles are relative to the user's neutral head pose.

    ``current`` (the zone currently reported) enables spatial hysteresis: to move to a
    neighbouring zone the point must cross the boundary by an extra ``th.hysteresis_cm``, so gaze
    jitter right on a boundary doesn't make the zone flicker."""
    if abs(head_yaw_deg) > th.far_head_yaw_deg or head_pitch_deg < -th.far_head_pitch_deg:
        return Zone.FAR_OFF
    if uv is None:  # ray never reaches the screen plane (looking up/away)
        return Zone.FAR_OFF
    u, v = uv
    W, H = screen.width_cm, screen.height_cm
    # Gaze error is angular, so tolerances scale with viewing distance: at 35 cm a 2.5 cm band is
    # only ~4°, smaller than typical calibration error.
    dist = max(float(origin_cm[2]), 10.0)
    edge = max(th.edge_cm, dist * math.tan(math.radians(th.edge_deg)))
    top_extra = dist * math.tan(math.radians(th.top_extra_deg))

    dx = -u if u < 0 else (u - W if u > W else 0.0)
    if v < 0:
        # Above the screen: looking up is mostly thinking, and the camera sits right there, so
        # allow extra tolerance before calling it off-screen.
        dy = max(0.0, -v - top_extra)
    else:
        dy = v - H if v > H else 0.0
    outside = math.hypot(dx, dy)
    # Signed distance to the screen border: < 0 inside, > 0 outside.
    s = outside if outside > 0 else -min(u, W - u, max(v, 0.0), H - v)

    # Below the screen, horizontally within its width: keyboard (judged by angle, since the
    # keyboard lies in front of the screen plane, not on it).
    if s > edge and v > H and -th.keyboard_side_slack_cm <= u <= W + th.keyboard_side_slack_cm:
        _, bottom_pitch = target_angles(screen, origin_cm, min(max(u, 0.0), W), H)
        if gaze_pitch_deg - bottom_pitch <= KEYBOARD_PITCH_DEG:
            return Zone.DOWN_KEYBOARD

    # Boundaries between ranked zones ON(0) | EDGE(1) | NEAR_OFF(2) | FAR_OFF(3).
    bounds = [-edge, edge, max(th.near_off_cm, edge + 1.0)]
    cur = _RANK.get(current) if current is not None else None
    if cur is not None:
        h = th.hysteresis_cm
        # Push each boundary away from the current zone.
        bounds = [b + h if cur <= k else b - h for k, b in enumerate(bounds)]
    rank = sum(s > b for b in bounds)
    return _ORDER[rank]


_ORDER = [Zone.ON_SCREEN, Zone.EDGE, Zone.NEAR_OFF, Zone.FAR_OFF]
_RANK = {z: i for i, z in enumerate(_ORDER)}


class ZoneSmoother:
    """Only switch the reported zone after a candidate has persisted for ``min_dwell_s``."""

    def __init__(self, min_dwell_s: float):
        self.min_dwell_s = min_dwell_s
        self.current: Zone = Zone.UNKNOWN
        self._candidate: Zone | None = None
        self._since = 0.0

    def update(self, zone: Zone, t: float) -> Zone:
        if zone == self.current:
            self._candidate = None
            return self.current
        if zone != self._candidate:
            self._candidate, self._since = zone, t
        if t - self._since >= self.min_dwell_s:
            self.current, self._candidate = zone, None
        return self.current
