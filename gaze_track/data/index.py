"""Combine normalized .npz files (MPIIFaceGaze persons, own exported sessions) into one dataset
with subject-wise splits. Patches stay memory-mapped per file until requested."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LABEL_KEYS = ("gaze", "head", "R", "face_center", "zone", "segment", "task", "trial_id", "frame")


@dataclass
class NormalizedIndex:
    files: list[Path]
    subject: np.ndarray  # (N,) str, one per sample
    file_idx: np.ndarray  # (N,) which file
    local_idx: np.ndarray  # (N,) index within that file
    labels: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.subject)

    @property
    def subjects(self) -> list[str]:
        return sorted(set(self.subject.tolist()))

    def has_gaze(self) -> np.ndarray:
        return ~np.isnan(self.labels["gaze"][:, 0])

    def patches(self, idx: np.ndarray) -> np.ndarray:
        """Load patches for the given global indices (grouped by file to limit I/O)."""
        idx = np.asarray(idx)
        out = None
        for fi in np.unique(self.file_idx[idx]):
            sel = np.nonzero(self.file_idx[idx] == fi)[0]
            with np.load(self.files[fi]) as z:
                p = z["patches"][self.local_idx[idx[sel]]]
            if out is None:
                out = np.empty((len(idx),) + p.shape[1:], np.uint8)
            out[sel] = p
        return out if out is not None else np.empty((0,), np.uint8)

    def subset(self, mask: np.ndarray) -> "NormalizedIndex":
        return NormalizedIndex(self.files, self.subject[mask], self.file_idx[mask], self.local_idx[mask],
                               {k: v[mask] for k, v in self.labels.items()})


def load_index(paths: list[str | Path]) -> NormalizedIndex:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    files = [f for f in files if not f.name.endswith(".preview.npz")]
    if not files:
        raise FileNotFoundError(f"No .npz files under {paths}")
    subj, fidx, lidx = [], [], []
    labels: dict[str, list[np.ndarray]] = {k: [] for k in LABEL_KEYS}
    for i, f in enumerate(files):
        with np.load(f) as z:
            n = len(z["gaze"])
            subj.append(np.full(n, str(z["subject"]), dtype=object))
            fidx.append(np.full(n, i, np.int32))
            lidx.append(np.arange(n, dtype=np.int64))
            for k in LABEL_KEYS:
                labels[k].append(z[k] if k in z.files else np.full(n, "", dtype=object))
    return NormalizedIndex(
        files,
        np.concatenate(subj).astype(str),
        np.concatenate(fidx),
        np.concatenate(lidx),
        {k: np.concatenate(v) for k, v in labels.items()},
    )


def lopo_folds(index: NormalizedIndex, subjects: list[str] | None = None):
    """Leave-one-person-out folds: yields (held_out_subject, train_mask, test_mask)."""
    for s in subjects or index.subjects:
        test = index.subject == s
        yield s, ~test, test


def summary(index: NormalizedIndex) -> str:
    g = np.degrees(index.labels["gaze"])
    hg = index.has_gaze()
    lines = [f"{len(index)} samples, {len(index.subjects)} subjects, {int(hg.sum())} with gaze labels"]
    for s in index.subjects:
        m = index.subject == s
        gm = g[m & hg]
        rng = (f"pitch {np.percentile(gm[:, 0], 5):+.0f}..{np.percentile(gm[:, 0], 95):+.0f}°, "
               f"yaw {np.percentile(gm[:, 1], 5):+.0f}..{np.percentile(gm[:, 1], 95):+.0f}°") if len(gm) else "no gaze"
        lines.append(f"  {s:16s} {int(m.sum()):6d}  {rng}")
    return "\n".join(lines)
