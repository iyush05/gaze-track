"""Camera discovery and robust frame reading."""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import Config

PERMISSION_HINT = (
    "On macOS, allow camera access for the app you run this from (Terminal / iTerm / VS Code) in "
    "System Settings → Privacy & Security → Camera, then fully quit and reopen that app.\n"
    "Also close other apps using the camera (Zoom, FaceTime, OBS, browser tabs)."
)


def backend():
    import cv2

    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def probe(index: int, seconds: float = 3.0, width: int = 640, height: int = 480) -> dict:
    """Open a camera index and report whether it delivers real (non-blank) frames."""
    import cv2

    info = {"index": index, "opened": False, "frames": 0, "size": None, "mean": None, "std": None, "motion": None}
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        return info
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    info["opened"] = True
    end = time.monotonic() + seconds
    frame = None
    diffs = []
    while time.monotonic() < end and info["frames"] < 15:
        ok, f = cap.read()
        if ok and f is not None:
            info["frames"] += 1
            if frame is not None and frame.shape == f.shape:
                diffs.append(float(np.abs(f.astype(np.int16) - frame.astype(np.int16)).mean()))
            frame = f
    cap.release()
    if frame is not None:
        info["size"] = f"{frame.shape[1]}x{frame.shape[0]}"
        info["mean"] = round(float(frame.mean()), 1)
        info["std"] = round(float(frame.std()), 1)
    # A real sensor always has noise between frames; a virtual camera's placeholder image is frozen.
    info["motion"] = round(float(np.median(diffs)), 2) if diffs else None
    info["static"] = info["motion"] is not None and info["motion"] < 0.05
    info["usable"] = info["frames"] >= 3 and (info["std"] or 0) > 4.0 and not info["static"]
    return info


def open_camera(cfg: Config, index: int | None = None):
    import cv2

    idx = cfg.camera_index if index is None else index
    if idx < 0:  # auto: first index that delivers a real picture
        for i in range(4):
            info = probe(i, seconds=2.5, width=cfg.frame_width, height=cfg.frame_height)
            if info["usable"]:
                idx = i
                print(f"Using camera {i} ({info['size']})")
                break
        else:
            sys.exit(
                "No camera delivered a usable picture (indices 0-3). Run "
                "`uv run python -m gaze_track cameras` for details.\n" + PERMISSION_HINT
            )
    cap = cv2.VideoCapture(idx, backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {idx}.\n" + PERMISSION_HINT)
    return cap


class FrameReader:
    """cap.read() with a watchdog, so a dead camera produces an error instead of a silent hang."""

    def __init__(self, cap, timeout_s: float = 5.0, max_width: int = 640):
        self.cap, self.timeout_s, self.max_width = cap, timeout_s, max_width
        self._last_ok = time.monotonic()

    def read(self):
        import cv2

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._last_ok = time.monotonic()
            # Some cameras ignore the requested size (e.g. 1080p FaceTime HD); downscale for speed.
            if frame.shape[1] > self.max_width:
                h = int(frame.shape[0] * self.max_width / frame.shape[1])
                frame = cv2.resize(frame, (self.max_width, h), interpolation=cv2.INTER_AREA)
            return frame
        if time.monotonic() - self._last_ok > self.timeout_s:
            sys.exit(
                f"Camera opened but sent no frames for {self.timeout_s:.0f}s. "
                "Try another camera with --camera N (list them with `cameras`).\n" + PERMISSION_HINT
            )
        time.sleep(0.01)
        return None
