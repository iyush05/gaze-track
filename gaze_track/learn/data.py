"""In-memory training data + GPU-side augmentation for normalized face patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.index import NormalizedIndex


@dataclass
class ArrayData:
    patches: np.ndarray  # (N, H, W, 3) uint8 BGR
    gaze: np.ndarray  # (N, 2) float32 pitch, yaw (rad, ETH)
    head: np.ndarray  # (N, 2)
    subject: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return len(self.gaze)


def load_arrays(index: NormalizedIndex, mask: np.ndarray | None = None, limit: int | None = None,
                seed: int = 0) -> ArrayData:
    """Gaze-labelled samples only. ~1.4 GB for all of MPIIFaceGaze at 112 px."""
    m = index.has_gaze() if mask is None else (mask & index.has_gaze())
    idx = np.nonzero(m)[0]
    if limit and len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
    return ArrayData(
        patches=index.patches(idx),
        gaze=index.labels["gaze"][idx].astype(np.float32),
        head=index.labels["head"][idx].astype(np.float32),
        subject=index.subject[idx],
    )


def batches(data: ArrayData, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    order = rng.permutation(len(data)) if shuffle else np.arange(len(data))
    for i in range(0, len(order), batch_size):
        b = np.sort(order[i : i + batch_size])  # sorted indices read faster
        yield b, data.patches[b], data.gaze[b], data.head[b]


def augment(x: torch.Tensor, gaze: torch.Tensor, head: torch.Tensor, gen: torch.Generator | None = None):
    """x: (N, H, W, 3) uint8 on device. Returns float (N,3,H,W) in [0,1] RGB and adjusted labels.

    * horizontal flip (p=0.5): a mirrored face looks the mirrored way → negate gaze & head yaw
    * brightness / contrast / gamma jitter, random grayscale, sensor noise (webcam conditions)
    """
    n = x.shape[0]
    dev = x.device
    img = x.flip(-1).permute(0, 3, 1, 2).float().div(255.0)  # BGR→RGB, NCHW
    r = lambda *s: torch.rand(*s, device=dev)  # noqa: E731

    flip = r(n) < 0.5
    img = torch.where(flip.view(-1, 1, 1, 1), img.flip(-1), img)
    gaze = gaze.clone()
    head = head.clone()
    gaze[flip, 1] = -gaze[flip, 1]
    head[flip, 1] = -head[flip, 1]

    bright = 0.6 + 0.8 * r(n, 1, 1, 1)
    contrast = 0.6 + 0.8 * r(n, 1, 1, 1)
    gamma = 0.7 + 0.6 * r(n, 1, 1, 1)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    img = ((img - mean) * contrast + mean) * bright
    img = img.clamp(1e-4, 1).pow(gamma)
    gray = r(n) < 0.15
    g = img.mean(1, keepdim=True).expand_as(img)
    img = torch.where(gray.view(-1, 1, 1, 1), g, img)
    img = (img + 0.02 * r(n, 1, 1, 1) * torch.randn_like(img)).clamp(0, 1)
    return img, gaze, head


def angular_error_deg(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """(N,2) ETH pitch/yaw radians → per-sample angle between gaze vectors, degrees."""
    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return torch.stack([-torch.cos(p) * torch.sin(y), -torch.sin(p), -torch.cos(p) * torch.cos(y)], 1)

    c = (vec(pred) * vec(true)).sum(1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.acos(c))
