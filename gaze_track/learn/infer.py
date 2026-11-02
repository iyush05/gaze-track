"""Learned gaze estimator: MediaPipe pose → normalized face patch → GazeNet → camera-frame gaze.

Per-user calibration for the learned model is an affine correction of its (yaw, pitch) output,
ridge-regularized toward the identity, so a few calibration points can fix a personal bias without
destroying what the network learned about gaze far outside the calibration range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.normalize import (
    face_center_cv,
    from_eth_pitchyaw,
    head_rotation_cv,
    mediapipe_intrinsics,
    normalize,
)
from ..geometry import GL_TO_CV, vector_to_angles
from .model import GazeNet, reparameterize


class LearnedGaze:
    """Runs the ONNX export next to the checkpoint when present (≈1 ms on CPU vs ≈60 ms for PyTorch
    eager on Apple Silicon), else PyTorch."""

    def __init__(self, ckpt: str | Path, device: str = "cpu", threads: int = 4, prefer_int8: bool = False):
        """prefer_int8: use best.int8.onnx when present. Off by default: on MPIIFaceGaze INT8 cost only
        +0.05° (0.71 vs 1.17 ms), but s01's per-user calibration stretches the model's compressed
        output ~3×, which amplifies quantization error too (fused 1.94° → 2.47° with glasses)."""
        torch.set_num_threads(threads)
        blob = torch.load(ckpt, map_location="cpu")
        self.cfg = blob["config"]
        self.roi = int(self.cfg["roi"])
        self.device = device
        self._K: np.ndarray | None = None
        self.model = None
        self.ort = None
        int8_path = Path(ckpt).with_suffix(".int8.onnx")
        onnx_path = int8_path if (prefer_int8 and int8_path.exists()) else Path(ckpt).with_suffix(".onnx")
        self.runtime = onnx_path.name if onnx_path.exists() else "pytorch"
        if onnx_path.exists():
            from .export_onnx import session

            self.ort = session(onnx_path, threads)
        else:
            m = GazeNet(self.cfg["backbone"], pretrained=False)
            m.load_state_dict(blob["state_dict"])
            self.model = reparameterize(m.eval()).to(device)

    def intrinsics(self, w: int, h: int) -> np.ndarray:
        if self._K is None or self._K[0, 2] != w / 2.0:
            self._K = mediapipe_intrinsics(w, h)
        return self._K

    def normalize_frame(self, image: np.ndarray, transform_gl: np.ndarray, landmarks_px: np.ndarray):
        """→ (patch, R, face_center_cm) or None if the pose is unusable."""
        h, w = image.shape[:2]
        K = self.intrinsics(w, h)
        depth = float((GL_TO_CV @ np.asarray(transform_gl)[:3, 3])[2])
        if depth <= 5:
            return None
        fc = face_center_cv(landmarks_px, depth, K)
        n = normalize(image, K, head_rotation_cv(transform_gl), fc, roi=self.roi)
        return n.patch, n.R, fc

    @torch.inference_mode()
    def predict_patches(self, patches: np.ndarray) -> np.ndarray:
        """(N, roi, roi, 3) uint8 BGR → (N, 2) normalized gaze (pitch, yaw) rad, ETH convention."""
        x = np.ascontiguousarray(patches, dtype=np.uint8)
        if self.ort is not None:
            return self.ort.run(None, {"patches": x})[0][:, :2].astype(np.float32)
        out = self.model(torch.from_numpy(x).to(self.device))
        return out[:, :2].float().cpu().numpy()

    @staticmethod
    def to_camera_angles(pred_norm: np.ndarray, R: np.ndarray) -> tuple[float, float]:
        """Normalized ETH (pitch, yaw) → camera-frame gaze as this repo's (yaw, pitch) in degrees."""
        g = R.T @ from_eth_pitchyaw(float(pred_norm[0]), float(pred_norm[1]))
        yaw, pitch = vector_to_angles(g)
        return math.degrees(yaw), math.degrees(pitch)


@dataclass
class AffineCalibration:
    """(yaw, pitch)_true ≈ A @ [yaw, pitch, 1] — ridge toward identity."""

    A: np.ndarray = field(default_factory=lambda: np.c_[np.eye(2), np.zeros(2)])
    stats: dict = field(default_factory=dict)

    def apply(self, yaw: float, pitch: float) -> tuple[float, float]:
        y, p = self.A @ np.array([yaw, pitch, 1.0])
        return float(y), float(p)

    @staticmethod
    def _fit(pred: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
        X = np.c_[pred, np.ones(len(pred))]
        prior = np.c_[np.eye(2), np.zeros(2)]
        reg = lam * len(pred) * np.diag([1.0, 1.0, 0.0])  # bias (personal offset) is free
        B = np.linalg.solve(X.T @ X + reg, X.T @ (target - X @ prior.T))
        return prior + B.T

    LAM_GRID = (30.0, 3.0, 0.3)

    @classmethod
    def fit(cls, pred: np.ndarray, target: np.ndarray, groups: np.ndarray | None = None,
            lam: float | str = "auto") -> "AffineCalibration":
        """pred/target: (N, 2) degrees. lam: deg² per unit slope change, or "auto" = pick from LAM_GRID
        by leave-one-point-out error. Slopes must be free: the MPIIFaceGaze model's yaw on s01 was
        r = 0.98 but compressed ×0.31 (eye-only gaze with a still head), so a strong pull toward
        slope 1 left 6.9° error where a fitted slope gives 3.3°."""
        if lam == "auto":
            if groups is None or len(np.unique(groups)) < 4:
                lam = 3.0
            else:
                fits = [cls.fit(pred, target, groups, l) for l in cls.LAM_GRID]
                best = min(fits, key=lambda c: c.stats["lopo_mean_err_deg"])
                best.stats["lam_auto"] = {str(l): f.stats["lopo_mean_err_deg"] for l, f in zip(cls.LAM_GRID, fits)}
                return best
        A = cls._fit(pred, target, lam)
        c = cls(A)
        e_raw = np.hypot(*(pred - target).T)
        e_fit = np.hypot(*(np.c_[pred, np.ones(len(pred))] @ A.T - target).T)
        c.stats = {"raw_mean_err_deg": float(e_raw.mean()), "train_mean_err_deg": float(e_fit.mean()),
                   "n_samples": int(len(pred)), "lam": lam, "slopes": [float(A[0, 0]), float(A[1, 1])]}
        if groups is not None and len(np.unique(groups)) >= 4:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                Ag = cls._fit(pred[tr], target[tr], lam)
                errs.append(np.hypot(*(np.c_[pred[te], np.ones(te.sum())] @ Ag.T - target[te]).T))
            c.stats["lopo_mean_err_deg"] = float(np.concatenate(errs).mean())
        return c
