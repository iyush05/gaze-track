"""Per-frame image quality on the face, used to report UNKNOWN instead of a confident wrong zone.

Chosen from the robustness stress test (s01 glasses session, baseline estimator), where the face was
still detected in 100 % of frames under every degradation, so detection never signals the problem:

| condition  | gaze error | face luminance | face sharpness (p50) |
|------------|-----------:|---------------:|---------------------:|
| clean      | 3.15°      | 90             | 0.027–0.038          |
| dim        | 3.95°      | 27             | 0.040–0.051          |
| very dark  | 8.83°      | 12             | 0.15–0.16            |
| blur       | 4.70°      | 90             | 0.005–0.006          |
| low-res    | 3.25°      | 90             | 0.016–0.020          |

Face luminance is measured on the face itself (backlit frames are *brighter* overall).
Sharpness is measured over the whole face (resized to 128 px), not the eye region: eye-region detail
vanishes whenever the lids drop, so an eye metric called every keyboard look and closed-eye frame
"blurry" (s01 no-glasses: 100 % of keyboard/eyes-closed frames). Face sharpness is the same for
dots, keyboard and closed eyes (0.026–0.038); the lowest 1 % of clean frames is ≥ 0.0135.
Iris jitter was rejected as a signal: it drops under blur, rises under harmless noise, and real eye
movement (reading) raises it too.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Quality:
    face_lum: float
    sharpness: float
    status: str = "ok"  # ok | dim | too_dark | blurry

    @property
    def unusable(self) -> bool:
        return self.status in ("too_dark", "blurry")


def measure(frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> tuple[float, float]:
    """(mean face luminance, normalized face sharpness)."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    p = np.asarray(landmarks_px, dtype=float)
    x0, y0 = np.clip(p[:, :2].min(0).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(p[:, :2].max(0).astype(int), 1, [w, h])
    face = g[y0:y1, x0:x1]
    if face.size < 64:
        return 0.0, 0.0
    lum = float(face.mean())
    f = cv2.resize(face, (128, 128)).astype(np.float32)  # face-size independent
    lap = cv2.Laplacian(cv2.GaussianBlur(f, (3, 3), 0), cv2.CV_32F)
    # Normalized by contrast so dim-but-sharp images aren't called blurry.
    return lum, float(lap.var() / max(float(f.std()) ** 2, 1.0))


class QualityMonitor:
    """Smooths the per-frame measurements (≈1 s) and classifies conditions."""

    def __init__(self, th, alpha: float = 0.1):
        self.th, self.alpha = th, alpha
        self.lum: float | None = None
        self.sharp: float | None = None

    def update(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> Quality:
        lum, sharp = measure(frame_bgr, landmarks_px)
        a = self.alpha
        self.lum = lum if self.lum is None else (1 - a) * self.lum + a * lum
        self.sharp = sharp if self.sharp is None else (1 - a) * self.sharp + a * sharp
        return self.classify(self.lum, self.sharp)

    def classify(self, lum: float, sharp: float) -> Quality:
        if lum < self.th.too_dark_lum:
            status = "too_dark"
        elif sharp < self.th.min_sharpness:
            status = "blurry"
        elif lum < self.th.dim_lum:
            status = "dim"
        else:
            status = "ok"
        return Quality(lum, sharp, status)

    def reset(self) -> None:
        self.lum = self.sharp = None
