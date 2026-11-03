"""Evidence capture for live sessions: keep snapshots only around flags, never continuous video.

Bundle layout (written incrementally, so a crash or closed terminal keeps what was captured):
    meta.json       profile, estimator, thresholds, start/end time
    timeline.csv    t, zone, quality  (one row per processed frame; no images)
    events.jsonl    every warning/alert, with the snapshot files that document it
    snapshots/      320-px JPEGs from a short in-memory buffer (start, middle, flag moment)

`gaze_track report <bundle>` turns a bundle into the same HTML review report as a recorded session.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class EvidenceRecorder:
    def __init__(self, out_dir: str | Path, meta: dict, buffer_s: float = 15.0, snap_fps: float = 10.0,
                 width: int = 320, jpeg_quality: int = 80):
        self.dir = Path(out_dir)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.meta = meta | {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "evidence"}
        self.buffer_s, self.period, self.width, self.q = buffer_s, 1.0 / snap_fps, width, jpeg_quality
        # In-memory only: (t, zone, jpeg bytes) for the last `buffer_s` seconds.
        self._buf: deque[tuple[float, str, bytes]] = deque()
        self._next_snap = 0.0
        self._n_snaps = 0
        self._n_events = 0
        self._next_flush = 0.0
        self._tl = open(self.dir / "timeline.csv", "w", newline="")
        self._tlw = csv.writer(self._tl)
        self._tlw.writerow(["t", "zone", "quality"])
        self._ev = open(self.dir / "events.jsonl", "a")
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def _encode(self, frame_bgr: np.ndarray) -> bytes:
        h = int(frame_bgr.shape[0] * self.width / frame_bgr.shape[1])
        small = cv2.resize(frame_bgr, (self.width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        return buf.tobytes() if ok else b""

    def _nearest(self, t: float) -> tuple[float, str, bytes] | None:
        if not self._buf:
            return None
        return min(self._buf, key=lambda x: abs(x[0] - t))

    def on_frame(self, frame_bgr: np.ndarray, result) -> None:
        t = float(result.t)
        q = getattr(result, "quality", None)
        self._tlw.writerow([f"{t:.3f}", result.zone.value, q.status if q is not None else "ok"])
        if t >= self._next_snap:
            self._next_snap = t + self.period
            self._buf.append((t, result.zone.value, self._encode(frame_bgr)))
            while self._buf and t - self._buf[0][0] > self.buffer_s:
                self._buf.popleft()
        for e in result.events:
            if e.severity != "info":
                self._save_event(e)
        if t >= self._next_flush:  # keep the timeline on disk even if the process is killed
            self._next_flush = t + 5.0
            self._tl.flush()

    def _save_event(self, e) -> None:
        start = e.t - (e.duration_s or 0.0)
        moments = [start, (start + e.t) / 2, e.t] if e.duration_s else [e.t]
        shots, seen = [], set()
        for m in moments:
            snap = self._nearest(m)
            if snap is None or snap[0] in seen:
                continue
            seen.add(snap[0])
            name = f"{self._n_snaps:05d}_{snap[0]:08.2f}.jpg"
            (self.dir / "snapshots" / name).write_bytes(snap[2])
            self._n_snaps += 1
            shots.append({"t": round(snap[0], 3), "zone": snap[1], "file": f"snapshots/{name}"})
        self._ev.write(json.dumps(e.to_dict() | {"snapshots": shots}) + "\n")
        self._ev.flush()
        self._tl.flush()
        self._n_events += 1

    def close(self, duration_s: float | None = None) -> None:
        self._tl.close()
        self._ev.close()
        self._buf.clear()
        self.meta |= {"ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "flags": self._n_events,
                      "snapshots": self._n_snaps}
        if duration_s is not None:
            self.meta["duration_s"] = round(duration_s, 1)
        self._write_meta()


def is_bundle(path: str | Path) -> bool:
    p = Path(path)
    return (p / "timeline.csv").exists() and (p / "events.jsonl").exists()


def load_bundle(path: str | Path) -> tuple[dict, list[dict], list[dict]]:
    """(meta, timeline frames [{t, zone, quality}], event dicts)."""
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
    with open(p / "timeline.csv") as f:
        frames = [{"t": float(r["t"]), "zone": r["zone"], "quality": r["quality"]} for r in csv.DictReader(f)]
    events = [json.loads(line) for line in (p / "events.jsonl").read_text().splitlines() if line.strip()]
    return meta, frames, events
