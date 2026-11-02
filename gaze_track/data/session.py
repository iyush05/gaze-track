"""On-disk format for recorded sessions.

data/own/<session_id>/
    meta.json        subject, conditions (glasses, lighting), screen geometry, window, camera model
    trials.json      the protocol actually run (ground truth per trial)
    frames.csv       one row per saved frame: frame, t, trial_id, segment, task, zone_gt,
                     u_gt, v_gt, approx, + MediaPipe features at capture time (for quick analysis)
    arrays.npz       per-frame float arrays aligned with frames.csv: transform (N,4,4),
                     landmarks (N,478,3) float16 (NaN when no face)
    frames/000000.jpg ...  raw camera frames (for training / re-processing in later phases)
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path("data/own")
N_LANDMARKS = 478

FRAME_FIELDS = [
    "frame", "t", "trial_id", "segment", "task", "zone_gt", "u_gt", "v_gt", "approx",
    "n_faces", "head_yaw", "head_pitch", "head_roll", "eye_h", "eye_v", "ear",
    "origin_x", "origin_y", "origin_z",
]


def new_session_id(subject: str) -> str:
    return f"{subject}_{time.strftime('%Y%m%d_%H%M%S')}"


class SessionWriter:
    def __init__(self, root: Path, meta: dict, jpeg_quality: int = 90):
        self.dir = Path(root)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.q = jpeg_quality
        self._f = open(self.dir / "frames.csv", "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=FRAME_FIELDS)
        self._w.writeheader()
        self._transforms: list[np.ndarray] = []
        self._landmarks: list[np.ndarray] = []
        self.n = 0
        self.write_json("meta.json", meta)

    def write_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2))

    def add(self, frame_bgr: np.ndarray, row: dict, transform: np.ndarray | None, landmarks: np.ndarray | None) -> None:
        import cv2

        cv2.imwrite(str(self.dir / "frames" / f"{self.n:06d}.jpg"), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        self._w.writerow({k: row.get(k) for k in FRAME_FIELDS} | {"frame": self.n})
        if self.n % 30 == 0:
            self._f.flush()  # keep labels on disk even if the process is killed
        self._transforms.append(np.full((4, 4), np.nan) if transform is None else transform)
        self._landmarks.append(
            np.full((N_LANDMARKS, 3), np.nan, np.float16) if landmarks is None else landmarks.astype(np.float16)
        )
        self.n += 1

    def close(self) -> None:
        self._f.close()
        np.savez_compressed(
            self.dir / "arrays.npz",
            transform=np.array(self._transforms, dtype=np.float32).reshape(-1, 4, 4),
            landmarks=np.array(self._landmarks, dtype=np.float16).reshape(-1, N_LANDMARKS, 3),
        )
        self.meta["n_frames"] = self.n
        self.write_json("meta.json", self.meta)


@dataclass
class Session:
    dir: Path
    meta: dict
    trials: list[dict]
    rows: list[dict]

    @property
    def subject(self) -> str:
        return self.meta["subject"]

    def frame_path(self, i: int) -> Path:
        return self.dir / "frames" / f"{int(i):06d}.jpg"

    def arrays(self) -> dict[str, np.ndarray]:
        if not (self.dir / "arrays.npz").exists():
            raise FileNotFoundError(f"{self.dir.name}: arrays.npz missing (recording was interrupted); "
                                    "landmark arrays are unavailable, but frames + labels can still be evaluated")
        with np.load(self.dir / "arrays.npz") as z:
            return {k: z[k] for k in z.files}


def _num(x):
    if x in ("", "None", None):
        return None
    try:
        return float(x)
    except ValueError:
        return x


def load_session(path: str | Path) -> Session:
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    trials = json.loads((d / "trials.json").read_text()) if (d / "trials.json").exists() else []
    with open(d / "frames.csv") as f:
        rows = [{k: (_num(v) if k not in ("segment", "task", "zone_gt") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    return Session(d, meta, trials, rows)


def list_sessions(root: str | Path = DATA_ROOT) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.glob("*") if (p / "meta.json").exists()) if root.exists() else []


def subject_split(subject: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    """Deterministic subject-wise split: all sessions of a subject land in the same split."""
    h = int(hashlib.sha1(subject.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"
