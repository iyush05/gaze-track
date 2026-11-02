"""Evaluation metrics: angular gaze error, zone confusion / F1, false alerts per hour."""

from __future__ import annotations

import math

import numpy as np

from .geometry import angles_to_vector


def angular_error_deg(yaw_p, pitch_p, yaw_t, pitch_t) -> np.ndarray:
    """Angle between predicted and true gaze directions (all inputs in degrees)."""
    out = []
    for yp, pp, yt, pt in zip(np.atleast_1d(yaw_p), np.atleast_1d(pitch_p), np.atleast_1d(yaw_t), np.atleast_1d(pitch_t)):
        a = angles_to_vector(math.radians(yp), math.radians(pp))
        b = angles_to_vector(math.radians(yt), math.radians(pt))
        out.append(math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))))
    return np.array(out)


def summarize(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
    }


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def per_class_prf(cm: np.ndarray, labels: list[str]) -> dict:
    out = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    return out


def macro_f1(prf: dict) -> float:
    """Mean F1 over classes that actually occur in the ground truth."""
    f = [v["f1"] for v in prf.values() if v["support"] > 0]
    return float(np.mean(f)) if f else 0.0


def zone_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    cm = confusion(y_true, y_pred, labels)
    prf = per_class_prf(cm, labels)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {"labels": labels, "confusion": cm.tolist(), "per_class": prf, "macro_f1": macro_f1(prf), "accuracy": acc}


def binary_report(y_true: list[bool], y_pred: list[bool]) -> dict:
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "n": int(len(t))}


def per_hour(count: int, seconds: float) -> float:
    return count * 3600.0 / seconds if seconds > 0 else float("nan")
