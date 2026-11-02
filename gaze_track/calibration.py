"""Feature → gaze-angle mapping, with per-user calibration.

The uncalibrated prior is a simple physical model:
    gaze_yaw   ≈ head_yaw   + EYE_GAIN * eye_h
    gaze_pitch ≈ head_pitch + EYE_GAIN * eye_v
Calibration fits a ridge-regularized *correction* on top of that prior, so coefficients that the
calibration data can't pin down (e.g. head yaw when the user keeps their head still) stay close to
the physically sensible prior instead of collapsing to zero. That matters for extrapolating to
off-screen gaze, which by definition is never seen during on-screen calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ScreenGeometry
from .geometry import angles_toward, angles_to_vector, intersect_z0

EYE_GAIN_DEG = 150.0  # rough: iris offset of 0.2 eye-widths ≈ 30° eye rotation
# The 2D iris offset (eye_h/eye_v) is not pure eye-in-head rotation: turning the head also moves the
# iris between the eye corners in the image (s01: 38° head turn → eye_h 0.24 vs ±0.08 during
# calibration). With head weight 1.0 head turns were counted twice ("10 cm left of the screen",
# true ≈31°, predicted 58°). 0.5 removed all false flags in s01's 10-min honest session at a cost of
# +0.2° on-screen error; the learned model (Phase 2) is the real fix.
HEAD_PRIOR = 0.5
N_FEAT = 5  # head_yaw, head_pitch, eye_h, eye_v, head_roll
# Typical range of each feature. Ridge penalties are applied in these *fixed* units, not per-sample
# std: a feature that barely moves during calibration (head pose, when the user keeps still) must
# not get a huge weight cheaply, or a real head turn later would be wildly mis-predicted.
FEAT_SCALE = np.array([10.0, 10.0, 0.1, 0.05, 10.0])
HEAD_IDX = [0, 1, 4]
EYE_IDX = [2, 3]
LAM_HEAD = 1.0  # strong: keep head-pose weights at the physical prior (gaze moves with the head)
LAM_EYE = 1e-3  # weak: eye gains / offsets are what calibration is for


def _prior_matrix() -> np.ndarray:
    """(2, N_FEAT+1) linear prior: rows = (yaw, pitch), cols = features + bias."""
    W = np.zeros((2, N_FEAT + 1))
    W[0, 0] = HEAD_PRIOR  # yaw ← head_yaw
    W[0, 2] = EYE_GAIN_DEG  # yaw ← eye_h
    W[1, 1] = HEAD_PRIOR  # pitch ← head_pitch
    W[1, 3] = EYE_GAIN_DEG  # pitch ← eye_v
    return W


@dataclass
class GazeMapper:
    W: np.ndarray = field(default_factory=_prior_matrix)
    calibrated: bool = False
    stats: dict = field(default_factory=dict)

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        """Feature vector → (yaw, pitch) in degrees."""
        x = np.append(np.asarray(feat, dtype=float), 1.0)
        yaw, pitch = self.W @ x
        return float(yaw), float(pitch)

    # ---------------------------------------------------------------- fitting
    @staticmethod
    def fit(
        feats: np.ndarray,
        targets_deg: np.ndarray,
        lam_head: float = LAM_HEAD,
        lam_eye: float = LAM_EYE,
        groups: np.ndarray | None = None,
    ) -> "GazeMapper":
        """Fit W = prior + ΔW by ridge regression on the residual.

        feats: (N, N_FEAT); targets_deg: (N, 2) yaw/pitch; groups: calibration-point id per sample
        (for leave-one-point-out error)."""
        W = _fit_W(feats, targets_deg, lam_head, lam_eye)
        mapper = GazeMapper(W=W, calibrated=True)
        mapper.stats = {
            "n_samples": int(len(feats)),
            "lam_head": lam_head,
            "lam_eye": lam_eye,
            "neutral_head": [float(np.median(feats[:, 0])), float(np.median(feats[:, 1]))],
        }
        if groups is not None and len(np.unique(groups)) >= 4:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                Wg = _fit_W(feats[tr], targets_deg[tr], lam_head, lam_eye)
                pred = np.c_[feats[te], np.ones(te.sum())] @ Wg.T
                errs.append(np.hypot(*(pred - targets_deg[te]).T))
            e = np.concatenate(errs)
            mapper.stats["lopo_mean_err_deg"] = float(e.mean())
            mapper.stats["lopo_p90_err_deg"] = float(np.percentile(e, 90))
        pred = np.c_[feats, np.ones(len(feats))] @ W.T
        mapper.stats["train_mean_err_deg"] = float(np.hypot(*(pred - targets_deg).T).mean())
        return mapper

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"W": self.W.tolist(), "calibrated": self.calibrated, "stats": self.stats}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "GazeMapper":
        if path is None or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls(W=np.array(raw["W"]), calibrated=raw.get("calibrated", True), stats=raw.get("stats", {}))


def _fit_W(feats: np.ndarray, targets: np.ndarray, lam_head: float, lam_eye: float) -> np.ndarray:
    prior = _prior_matrix()
    X = np.c_[feats, np.ones(len(feats))]
    resid = targets - X @ prior.T
    # Center, and scale by fixed typical ranges (see FEAT_SCALE).
    mu = feats.mean(0)
    Xs = np.c_[(feats - mu) / FEAT_SCALE, np.ones(len(feats))]
    pen = np.zeros(N_FEAT + 1)
    pen[HEAD_IDX] = lam_head
    pen[EYE_IDX] = lam_eye
    pen[-1] = 1e-12  # don't penalize the bias
    B = np.linalg.solve(Xs.T @ Xs + len(feats) * np.diag(pen), Xs.T @ resid)
    # Convert back: resid ≈ ((x-mu)/scale) @ B[:-1] + B[-1]
    dW = np.zeros_like(prior)
    dW[:, :-1] = (B[:-1] / FEAT_SCALE[:, None]).T
    dW[:, -1] = B[-1] - (mu / FEAT_SCALE) @ B[:-1]
    return prior + dW


# ---------------------------------------------------------------------------- targets
def calibration_targets(n: int = 9, margin: float | None = None) -> list[tuple[float, float]]:
    """Normalized (fx, fy) target positions in [0, 1]² of the visible window.

    16 points use a 4 % margin: users look at the very edge of the screen, and with the 9-point grid
    (8 % ≈ 2.3 cm in) edge gaze was extrapolated: s01 held-out near-edge dots were pushed outward by
    up to 1.6 cm, enough to show NEAR_OFF at the right edge. Adding near-edge dots cut held-out edge
    error 1.85 → 1.52 cm (p90 3.51 → 2.80)."""
    if margin is None:
        margin = 0.04 if n >= 16 else 0.08
    if n == 5:
        return [(0.5, 0.5), (margin, margin), (1 - margin, margin), (margin, 1 - margin), (1 - margin, 1 - margin)]
    k = int(round(math.sqrt(n)))
    vals = np.linspace(margin, 1 - margin, k)
    pts = [(float(x), float(y)) for y in vals for x in vals]
    # Visit in a snake order so the eyes travel short distances.
    rows = [pts[i * k : (i + 1) * k] for i in range(k)]
    return [p for i, r in enumerate(rows) for p in (r if i % 2 == 0 else r[::-1])]


def window_top_offset_cm(screen: ScreenGeometry, win_w: int, win_h: int) -> float:
    """If the fullscreen window is shorter than the physical display (e.g. the MacBook notch band
    is excluded), assume the missing strip is at the top and return its height in cm."""
    if win_w <= 0 or win_h <= 0:
        return 0.0
    visible_h_cm = screen.width_cm * win_h / win_w
    return max(0.0, screen.height_cm - visible_h_cm)


def target_cm(screen: ScreenGeometry, fx: float, fy: float, top_offset_cm: float) -> tuple[float, float]:
    """Normalized window position → screen (u, v) in cm."""
    visible_h = screen.height_cm - top_offset_cm
    return fx * screen.width_cm, top_offset_cm + fy * visible_h


def target_angles(screen: ScreenGeometry, origin_cm: np.ndarray, u: float, v: float) -> tuple[float, float]:
    yaw, pitch = angles_toward(origin_cm, np.array(screen.screen_to_camera(u, v)))
    return math.degrees(yaw), math.degrees(pitch)


def project_to_screen(
    screen: ScreenGeometry, origin_cm: np.ndarray, yaw_deg: float, pitch_deg: float
) -> tuple[float, float] | None:
    hit = intersect_z0(origin_cm, angles_to_vector(math.radians(yaw_deg), math.radians(pitch_deg)))
    if hit is None:
        return None
    return screen.camera_to_screen(float(hit[0]), float(hit[1]))
