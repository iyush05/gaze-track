"""Gaze-angle conventions and ray–plane intersection (camera frame, cm).

Angle convention (camera frame, +y down, +z toward the user):
    A gaze direction pointing *at the camera / screen* has negative z.
    yaw   > 0  → direction has +x component (toward camera +x = user's left)
    pitch > 0  → direction has +y component (looking down)

    d = ( sin(yaw) * cos(pitch),
          sin(pitch),
         -cos(yaw) * cos(pitch) )
"""

from __future__ import annotations

import math

import numpy as np


def angles_to_vector(yaw: float, pitch: float) -> np.ndarray:
    """Gaze angles (radians) → unit direction in the camera frame."""
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp])


def vector_to_angles(d: np.ndarray) -> tuple[float, float]:
    """Unit (or any non-zero) direction → (yaw, pitch) in radians."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    yaw = math.atan2(float(d[0]), float(-d[2]))
    return yaw, pitch


def intersect_z0(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Intersect a ray with the screen plane z = 0. Returns the 3D point or None if the
    ray points away from (or parallel to) the plane."""
    oz, dz = float(origin[2]), float(direction[2])
    if dz >= -1e-6 or oz <= 0:
        return None
    t = -oz / dz
    return np.asarray(origin, dtype=float) + t * np.asarray(direction, dtype=float)


def angles_toward(origin: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Gaze angles of the ray from ``origin`` to ``target``."""
    return vector_to_angles(np.asarray(target, dtype=float) - np.asarray(origin, dtype=float))


# OpenGL camera frame (MediaPipe: +y up, camera looks down -z) → OpenCV frame (+y down, +z out).
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def head_angles_deg(fwd_cv: np.ndarray, up_cv: np.ndarray) -> tuple[float, float, float]:
    """Head (yaw, pitch, roll) in degrees from the face's forward (nose) and up vectors,
    both in the OpenCV camera frame. Uses the gaze sign convention: a face looking straight
    at the camera has fwd = (0, 0, -1) → yaw = pitch = 0. Roll > 0 = head tilted so that the
    top of the head moves toward camera +x."""
    yaw, pitch = vector_to_angles(fwd_cv)
    roll = math.atan2(float(up_cv[0]), float(-up_cv[1]))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
