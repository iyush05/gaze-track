"""OpenCV overlay: camera view with landmarks + a screen-plane minimap."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .features import L_IRIS, R_IRIS
from .pipeline import FrameResult
from .zones import Zone

ZONE_COLORS = {  # BGR
    Zone.ON_SCREEN: (80, 200, 80),
    Zone.EDGE: (0, 200, 255),
    Zone.NEAR_OFF: (0, 120, 255),
    Zone.FAR_OFF: (40, 40, 230),
    Zone.DOWN_KEYBOARD: (200, 160, 60),
    Zone.EYES_CLOSED: (160, 160, 160),
    Zone.NO_FACE: (40, 40, 230),
    Zone.MULTI_FACE: (200, 0, 200),
    Zone.UNKNOWN: (130, 130, 130),
}
SEVERITY_COLORS = {"info": (200, 200, 200), "warning": (0, 180, 255), "alert": (60, 60, 255)}


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_minimap(cfg: Config, r: FrameResult, size: int = 260) -> np.ndarray:
    """Top-down view of the screen plane, from the user's perspective, in cm."""
    s, th = cfg.screen, cfg.zones
    span_w = s.width_cm + 2 * th.near_off_cm
    span_h = s.height_cm + 2 * th.near_off_cm
    scale = size / max(span_w, span_h)
    w, h = int(span_w * scale), int(span_h * scale)
    img = np.full((h, w, 3), 30, np.uint8)

    def to_px(u, v):
        return int((u + th.near_off_cm) * scale), int((v + th.near_off_cm) * scale)

    # Near-off boundary, edge band, screen, keyboard band.
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 80, 160), 1)
    cv2.rectangle(img, to_px(-th.edge_cm, -th.edge_cm), to_px(s.width_cm + th.edge_cm, s.height_cm + th.edge_cm), (0, 140, 200), 1)
    cv2.rectangle(img, to_px(0, 0), to_px(s.width_cm, s.height_cm), (220, 220, 220), 2)
    cv2.rectangle(img, to_px(th.edge_cm, th.edge_cm), to_px(s.width_cm - th.edge_cm, s.height_cm - th.edge_cm), (0, 140, 200), 1)
    cv2.circle(img, to_px(s.width_cm / 2 + s.camera_x_offset_cm, -s.camera_to_top_cm), 3, (255, 255, 255), -1)

    if r.point_uv is not None:
        u, v = r.point_uv
        u = float(np.clip(u, -th.near_off_cm * 1.5, s.width_cm + th.near_off_cm * 1.5))
        v = float(np.clip(v, -th.near_off_cm * 1.5, s.height_cm + th.near_off_cm * 1.5))
        p = to_px(u, v)
        p = (int(np.clip(p[0], 4, w - 5)), int(np.clip(p[1], 4, h - 5)))
        cv2.circle(img, p, 6, ZONE_COLORS[r.zone], -1)
        cv2.circle(img, p, 6, (255, 255, 255), 1)
    return img


def draw_overlay(frame: np.ndarray, cfg: Config, r: FrameResult, recent_events: list, fps: float, calibrated: bool) -> np.ndarray:
    img = frame.copy()
    if r.landmarks:
        for i, face in enumerate(r.landmarks.faces):
            pts = face.landmarks_px
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            for idx in (R_IRIS, L_IRIS):
                cv2.circle(img, (int(pts[idx, 0]), int(pts[idx, 1])), 2, color, -1)
            x0, y0 = pts[:, :2].min(0).astype(int)
            x1, y1 = pts[:, :2].max(0).astype(int)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    img = cv2.flip(img, 1)  # mirror for a natural "selfie" view

    _text(img, r.zone.value, (12, 34), 0.9, ZONE_COLORS[r.zone], 2)
    lines = [f"fps {fps:5.1f}  latency {r.timings_ms.get('total', 0):5.1f} ms (landmarks {r.timings_ms.get('landmarks', 0):4.1f})"]
    if r.features:
        f = r.features
        lines.append(f"head yaw {f.head_yaw:+5.1f} pitch {f.head_pitch:+5.1f} roll {f.head_roll:+5.1f}")
        lines.append(f"eye h {f.eye_h:+.3f} v {f.eye_v:+.3f}  EAR {f.ear:.2f}  dist {f.origin_cm[2]:4.1f}cm")
    if r.gaze_yaw is not None:
        lines.append(f"gaze yaw {r.gaze_yaw:+5.1f} pitch {r.gaze_pitch:+5.1f}")
    if r.point_uv:
        lines.append(f"screen u {r.point_uv[0]:+6.1f} v {r.point_uv[1]:+6.1f} cm")
    if not calibrated:
        lines.append("UNCALIBRATED - run: python -m gaze_track calibrate")
    for i, s in enumerate(lines):
        _text(img, s, (12, 60 + 20 * i), 0.5, (0, 200, 255) if s.startswith("UNCAL") else (255, 255, 255))

    for i, e in enumerate(recent_events[-5:][::-1]):
        _text(img, f"{e.kind} ({e.duration_s:.1f}s)", (12, img.shape[0] - 14 - 20 * i), 0.5, SEVERITY_COLORS[e.severity])

    mm = draw_minimap(cfg, r, size=min(260, img.shape[1] // 3))
    h, w = mm.shape[:2]
    img[8 : 8 + h, img.shape[1] - w - 8 : img.shape[1] - 8] = mm
    return img
