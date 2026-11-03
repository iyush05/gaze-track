"""Configuration: physical screen geometry and decision thresholds.

Coordinate frames
-----------------
Camera frame (OpenCV convention, units = cm):
    origin at the webcam, +x = image right (the *user's left*), +y = down, +z = toward the user.
    The screen is assumed coplanar with the camera (plane z = 0), hanging below it.

Screen frame (units = cm, from the user's point of view):
    origin at the top-left corner of the visible display area, +u = right (as the user sees it),
    +v = down.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScreenGeometry:
    # Defaults: MacBook Air M2 13.6" (2560x1664, 13.6" diagonal), camera centered above display.
    width_cm: float = 28.96
    height_cm: float = 18.82
    # Vertical distance from the camera centre down to the top edge of the display area.
    camera_to_top_cm: float = 0.6
    # Horizontal offset of the camera from the display's centre (+ = towards user's right).
    camera_x_offset_cm: float = 0.0

    def camera_to_screen(self, x_cam: float, y_cam: float) -> tuple[float, float]:
        """Map a point on the z=0 plane (camera frame) to screen coords (u, v) in cm."""
        # Camera +x is the user's left, so mirror it.
        u = -x_cam + self.width_cm / 2.0 + self.camera_x_offset_cm
        v = y_cam - self.camera_to_top_cm
        return u, v

    def screen_to_camera(self, u: float, v: float) -> tuple[float, float, float]:
        """Map a screen point (u, v) in cm to a 3D point in the camera frame (z = 0)."""
        x = -(u - self.width_cm / 2.0 - self.camera_x_offset_cm)
        y = v + self.camera_to_top_cm
        return x, y, 0.0


@dataclass
class ZoneThresholds:
    edge_cm: float = 2.5  # half-width of the band around the screen border (minimum)
    # ...or this visual angle at the current distance, whichever is larger. ≈ p90 gaze error (s01: 5.2°),
    # i.e. don't call it off-screen unless it's beyond the estimator's own error.
    edge_deg: float = 6.0
    top_extra_deg: float = 10.0  # extra tolerance above the screen (looking up = thinking)
    near_off_cm: float = 20.0  # beyond the edge band up to this distance → NEAR_OFF
    keyboard_band_cm: float = 25.0  # (unused; kept for config compatibility)
    # Gaze up to this angle below the screen's bottom edge = keyboard (allowed). s01: keyboard looks
    # and typing never exceeded 22°, a phone held in the lap was at 34–38° (median) → 25°.
    keyboard_pitch_deg: float = 25.0
    keyboard_side_slack_cm: float = 3.0  # horizontal slack for the keyboard zone
    far_head_yaw_deg: float = 40.0  # |head yaw| beyond this → FAR_OFF regardless of gaze
    far_head_pitch_deg: float = 35.0
    eye_closed_ratio: float = 0.17  # eye aspect ratio below this → closed (no calibration baseline)
    # With a calibration baseline: closed below this × baseline EAR. s01: truly closed ≈ 0.06×, but
    # without glasses looking down at the keyboard lowers the lids to ≈ 0.44× (51 % of keyboard
    # frames were "closed" at 0.45) → 0.25.
    eye_closed_rel: float = 0.25
    # Low EAR *and* head pitched down this far (vs neutral) = looking down (lap/desk), not closed
    # eyes → treated as off-screen. s01: phone in lap +9.1/+9.6°, keyboard/typing −1.4…−3.0°.
    look_down_head_pitch_deg: float = 6.0
    min_dwell_s: float = 0.15  # hysteresis: a zone must persist this long before switching
    hysteresis_cm: float = 1.2  # spatial hysteresis: extra distance needed to leave current zone


@dataclass
class EventThresholds:
    brief_glance_max_s: float = 1.5
    merge_gap_s: float = 0.5  # an off-screen glance only ends after this long back on screen
    sustained_off_s: float = 3.0
    # Looking clearly *up* (pitch below this, i.e. above the screen) is mostly thinking: both s01
    # honest_thinking false flags were upward looks; notes/phones sit beside or below the screen.
    up_pitch_deg: float = -15.0
    sustained_up_s: float = 8.0
    # Head turned to the side (vs neutral): the clearest cheating cue, independent of eye tracking.
    # s01: honest thinking ≤ 4.7°, turning to ask someone 33–50° but only 1.1–3.1 s per turn.
    head_turn_deg: float = 25.0
    head_turn_s: float = 1.5
    head_turn_repeat: int = 2  # this many turns within repeated_window_s → alert
    repeated_window_s: float = 60.0
    repeated_min_count: int = 3
    # Glances within this angular distance count as "the same spot". 30° merged up-left and up-right
    # thinking glances into one pattern (false flag in s01's honest_thinking segment).
    repeated_direction_deg: float = 15.0
    face_absent_s: float = 2.0
    multi_face_s: float = 0.5
    eyes_closed_s: float = 2.0


@dataclass
class Config:
    screen: ScreenGeometry = field(default_factory=ScreenGeometry)
    zones: ZoneThresholds = field(default_factory=ZoneThresholds)
    events: EventThresholds = field(default_factory=EventThresholds)
    model_path: str = "models/face_landmarker.task"
    camera_index: int = -1  # -1 = auto-pick the first camera with a real picture
    max_faces: int = 2  # landmarker faces when face_count_every = 0
    # >0: landmarker tracks 1 face and a BlazeFace detector counts faces every N frames
    # (saves ~1.35 ms/frame vs max_faces=2; second person still noticed within 0.1 s at 30 fps)
    face_count_every: int = 3
    face_detector_path: str = "models/blaze_face_short_range.tflite"
    # ONNX Runtime threads for the learned model. 2 is fastest end-to-end on the M2: 4 makes the model
    # itself faster (1.4 vs 2.0 ms) but slows MediaPipe by ~0.9 ms through core contention.
    learned_threads: int = 2
    blendshapes: bool = False  # MediaPipe expression scores; unused → off for latency
    frame_width: int = 640
    frame_height: int = 480
    # Assumed typical viewing distance, used only before calibration if depth is unreliable.
    default_distance_cm: float = 55.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls(
            screen=ScreenGeometry(**raw.get("screen", {})),
            zones=ZoneThresholds(**raw.get("zones", {})),
            events=EventThresholds(**raw.get("events", {})),
            **{k: v for k, v in raw.items() if k not in ("screen", "zones", "events")},
        )
