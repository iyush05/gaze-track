"""Gaze data normalization (Zhang, Sugano & Bulling, "Revisiting Data Normalization for
Appearance-Based Gaze Estimation", ETRA 2018) — the scheme used by ETH-XGaze / MPIIFaceGaze.

A virtual camera is rotated to look straight at the face centre (removing head roll) and scaled to
a fixed distance, so every face patch has the same scale and orientation. Gaze labels are rotated
into that virtual camera; predictions are rotated back with R^T.

All vectors are in the OpenCV camera frame (+x right, +y down, +z out of the camera), units cm.
Sign conventions for (pitch, yaw):
  * this repo  (geometry.py): yaw = atan2(x, -z),  pitch = asin(y)
  * ETH-XGaze / MPIIFaceGaze : yaw = atan2(-x, -z), pitch = asin(-y)   → both are negated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import GL_TO_CV

FOCAL_NORM = 960.0  # ETH-XGaze uses 960 px focal length at 600 mm for 224 px patches
DISTANCE_NORM_CM = 60.0
# Landmarks whose 3D centroid defines the face centre (eye corners + mouth corners).
FACE_CENTER_IDX = [33, 133, 362, 263, 61, 291]


@dataclass
class NormResult:
    patch: np.ndarray  # (roi, roi, 3) BGR
    R: np.ndarray  # (3,3) rotation camera → normalized camera (apply to gaze vectors)
    W: np.ndarray  # (3,3) image warp
    face_center: np.ndarray  # (3,) cm
    head_R_norm: np.ndarray  # (3,3) head rotation in the normalized camera


def mediapipe_intrinsics(width: int, height: int, vertical_fov_deg: float = 63.0) -> np.ndarray:
    f = (height / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def head_rotation_cv(transform_gl: np.ndarray) -> np.ndarray:
    """MediaPipe facial transform (OpenGL axes) → head rotation in the OpenCV frame, with the
    head's local axes also expressed OpenCV-style (x = face's image-right, y = down, z = toward camera
    reversed). For a frontal upright face this is ≈ identity."""
    return GL_TO_CV @ np.asarray(transform_gl)[:3, :3] @ GL_TO_CV


def face_center_cv(landmarks_px: np.ndarray, depth_cm: float, K: np.ndarray) -> np.ndarray:
    """Back-project the 2D centroid of eye/mouth corners at the head's depth."""
    c = np.asarray(landmarks_px, dtype=float)[FACE_CENTER_IDX, :2].mean(0)
    return np.array([(c[0] - K[0, 2]) / K[0, 0] * depth_cm, (c[1] - K[1, 2]) / K[1, 1] * depth_cm, depth_cm])


def normalization_rotation(face_center: np.ndarray, head_R: np.ndarray) -> np.ndarray:
    forward = face_center / np.linalg.norm(face_center)
    hx = head_R[:, 0]
    down = np.cross(forward, hx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.stack([right, down, forward])


def normalize(
    image: np.ndarray,
    K: np.ndarray,
    head_R: np.ndarray,
    face_center: np.ndarray,
    roi: int = 224,
    focal_norm: float = FOCAL_NORM,
    distance_norm: float = DISTANCE_NORM_CM,
) -> NormResult:
    import cv2

    # Keep the face the same relative size for smaller patches.
    f_n = focal_norm * roi / 224.0
    dist = float(np.linalg.norm(face_center))
    cam_n = np.array([[f_n, 0, roi / 2.0], [0, f_n, roi / 2.0], [0, 0, 1.0]])
    S = np.diag([1.0, 1.0, distance_norm / dist])
    R = normalization_rotation(face_center, head_R)
    W = cam_n @ S @ R @ np.linalg.inv(K)
    patch = cv2.warpPerspective(image, W, (roi, roi))
    return NormResult(patch, R, W, np.asarray(face_center, float), R @ head_R)


def normalize_gaze(gaze_vec: np.ndarray, R: np.ndarray) -> np.ndarray:
    g = R @ np.asarray(gaze_vec, float)
    return g / np.linalg.norm(g)


def denormalize_gaze(gaze_vec_norm: np.ndarray, R: np.ndarray) -> np.ndarray:
    g = R.T @ np.asarray(gaze_vec_norm, float)
    return g / np.linalg.norm(g)


def to_eth_pitchyaw(g: np.ndarray) -> tuple[float, float]:
    """Unit gaze vector → (pitch, yaw) radians in the ETH-XGaze convention."""
    g = g / np.linalg.norm(g)
    return math.asin(float(-g[1])), math.atan2(float(-g[0]), float(-g[2]))


def from_eth_pitchyaw(pitch: float, yaw: float) -> np.ndarray:
    return np.array([-math.cos(pitch) * math.sin(yaw), -math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])
