"""Hand-crafted gaze features from MediaPipe output (Phase 0 baseline).

Features per frame:
  * head pose (yaw, pitch, roll) in degrees, from the facial transformation matrix
  * eye-in-head offsets: iris position relative to the eye corners / lids, per eye
  * eye aspect ratio (openness), per eye
  * eye-midpoint position in the camera frame (cm), used as the gaze-ray origin
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import GL_TO_CV, head_angles_deg
from .landmarks import FaceObservation

# MediaPipe FaceMesh indices ("right"/"left" = the subject's own right/left).
R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS = 33, 133, 159, 145, 468
L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS = 362, 263, 386, 374, 473

# MediaPipe's geometry pipeline assumes a perspective camera with this vertical FOV.
MP_VERTICAL_FOV_DEG = 63.0


@dataclass
class GazeFeatures:
    head_yaw: float
    head_pitch: float
    head_roll: float
    eye_h: float  # mean horizontal iris offset, ≈ [-0.5, 0.5]; + = toward image right
    eye_v: float  # mean vertical iris offset (normalized by eye width); + = down
    ear: float  # mean eye aspect ratio
    origin_cm: np.ndarray  # (3,) eye midpoint in the camera frame (OpenCV axes)
    face_width_px: float

    def vector(self) -> np.ndarray:
        """Feature vector used by the calibration regressor."""
        return np.array(
            [self.head_yaw, self.head_pitch, self.eye_h, self.eye_v, self.head_roll],
            dtype=np.float64,
        )


def _eye_offsets(p: np.ndarray, corner_a: int, corner_b: int, upper: int, lower: int, iris: int):
    """Iris offset in an eye-aligned frame. corner_a → corner_b runs toward image +x."""
    a, b = p[corner_a, :2], p[corner_b, :2]
    axis = b - a
    width = float(np.linalg.norm(axis)) + 1e-6
    ux = axis / width
    uy = np.array([-ux[1], ux[0]])  # perpendicular, points toward image +y (down) when level
    center = (a + b) / 2.0
    rel = p[iris, :2] - center
    h = float(rel @ ux) / width
    v = float(rel @ uy) / width
    opening = float(np.linalg.norm(p[upper, :2] - p[lower, :2])) / width
    return h, v, opening


def extract(face: FaceObservation, image_size: tuple[int, int]) -> GazeFeatures:
    p = face.landmarks_px
    w, h = image_size

    # Subject's right eye appears on image-left: outer(33) → inner(133) runs toward +x.
    rh, rv, rear = _eye_offsets(p, R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS)
    # Subject's left eye: inner(362) → outer(263) runs toward +x.
    lh, lv, lear = _eye_offsets(p, L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS)

    # Head pose from the transform (OpenGL axes → OpenCV axes).
    M = face.transform
    R_gl, t_gl = M[:3, :3], M[:3, 3]
    fwd_cv = GL_TO_CV @ (R_gl @ np.array([0.0, 0.0, 1.0]))
    up_cv = GL_TO_CV @ (R_gl @ np.array([0.0, 1.0, 0.0]))
    yaw, pitch, roll = head_angles_deg(fwd_cv, up_cv)
    t_cv = GL_TO_CV @ t_gl
    depth = float(t_cv[2]) if t_cv[2] > 5 else 55.0

    # Back-project the eye midpoint at the head's depth using MediaPipe's assumed intrinsics.
    f = (h / 2.0) / math.tan(math.radians(MP_VERTICAL_FOV_DEG) / 2.0)
    mid = (p[R_IRIS, :2] + p[L_IRIS, :2]) / 2.0
    origin = np.array([(mid[0] - w / 2.0) / f * depth, (mid[1] - h / 2.0) / f * depth, depth])

    return GazeFeatures(
        head_yaw=yaw,
        head_pitch=pitch,
        head_roll=roll,
        eye_h=(rh + lh) / 2.0,
        eye_v=(rv + lv) / 2.0,
        ear=(rear + lear) / 2.0,
        origin_cm=origin,
        face_width_px=float(np.ptp(p[:, 0])),
    )
