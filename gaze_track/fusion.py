"""Fused gaze estimator: per-user ridge regression on the Phase 0 mapper's prediction and the learned
model's (affine-calibrated) prediction.

The two make different mistakes (s01: learned is better in pitch, the feature mapper in yaw), so a
blend fitted on the 9 calibration dots beat both on held-out screen dots: 1.97° vs 3.24 / 3.20
(glasses) and 2.53° vs 2.94 / 3.97 (no glasses).

    final (yaw, pitch) = W @ [base_yaw, base_pitch, learned_yaw, learned_pitch, 1]

W is ridge-regularized toward an equal blend (0.5·base + 0.5·learned per axis), so with little or
ambiguous calibration data it degrades to averaging rather than to an arbitrary combination, and
correlated inputs can't get large opposite-sign weights that would blow up off-screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .learn.infer import AffineCalibration

LAM_FUSION = 1.0


def _prior() -> np.ndarray:
    W = np.zeros((2, 5))
    W[0, 0] = W[0, 2] = 0.5
    W[1, 1] = W[1, 3] = 0.5
    return W


def _fit_W(F: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    prior = _prior()
    R = T - F @ prior.T
    pen = lam * len(F) * np.diag([1.0, 1.0, 1.0, 1.0, 0.0])
    # Penalty scale: inputs are in degrees with ~10° spread, so weight changes cost lam·N·Δw²
    # against a data term of ~N·100·Δw².
    B = np.linalg.solve(F.T @ F + pen, F.T @ R)
    return prior + B.T


@dataclass
class FusedEstimator:
    W: np.ndarray = field(default_factory=_prior)
    learned_cal: AffineCalibration = field(default_factory=AffineCalibration)
    stats: dict = field(default_factory=dict)

    def predict(self, base: tuple[float, float], learned_raw: tuple[float, float] | None) -> tuple[float, float]:
        """Falls back to the feature mapper alone when the learned model has no prediction."""
        if learned_raw is None:
            return float(base[0]), float(base[1])
        ly, lp = self.learned_cal.apply(*learned_raw)
        y, p = self.W @ np.array([base[0], base[1], ly, lp, 1.0])
        return float(y), float(p)

    @classmethod
    def fit(cls, base: np.ndarray, learned_raw: np.ndarray, target: np.ndarray,
            groups: np.ndarray | None = None, lam: float = LAM_FUSION) -> "FusedEstimator":
        """base, learned_raw, target: (N, 2) degrees; groups = calibration point ids."""
        lcal = AffineCalibration.fit(learned_raw, target, groups=groups)
        learned = np.c_[learned_raw, np.ones(len(learned_raw))] @ lcal.A.T

        def design(b, l):
            return np.c_[b, l, np.ones(len(b))]

        F = design(base, learned)
        est = cls(_fit_W(F, target, lam), lcal)
        e_fit = np.hypot(*(F @ est.W.T - target).T)
        est.stats = {"n_samples": int(len(F)), "lam": lam, "train_mean_err_deg": float(e_fit.mean()),
                     "weights": {"yaw": est.W[0, :4].round(3).tolist(), "pitch": est.W[1, :4].round(3).tolist()},
                     "learned_calibration": lcal.stats}
        if groups is not None and len(np.unique(groups)) >= 4:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                lc = AffineCalibration.fit(learned_raw[tr], target[tr], groups=groups[tr])
                lt = np.c_[learned_raw, np.ones(len(learned_raw))] @ lc.A.T
                Fg = design(base, lt)
                Wg = _fit_W(Fg[tr], target[tr], lam)
                errs.append(np.hypot(*(Fg[te] @ Wg.T - target[te]).T))
            est.stats["lopo_mean_err_deg"] = float(np.concatenate(errs).mean())
        return est

    # ------------------------------------------------------------------ io (stored inside a profile)
    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "learned_A": self.learned_cal.A.tolist(), "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "FusedEstimator":
        return cls(np.array(d["W"]), AffineCalibration(np.array(d["learned_A"])), d.get("stats", {}))


def load_profile_fusion(profile_path: str | Path) -> tuple[FusedEstimator | None, str | None]:
    """(fusion, model checkpoint) stored in a calibration profile, if any."""
    p = Path(profile_path)
    if not p.exists():
        return None, None
    raw = json.loads(p.read_text())
    if "fusion" not in raw:
        return None, None
    return FusedEstimator.from_dict(raw["fusion"]), raw["fusion"].get("model")
