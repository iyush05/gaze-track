"""Thin wrapper around the MediaPipe Tasks FaceLandmarker (VIDEO mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FaceObservation:
    landmarks_px: np.ndarray  # (478, 3): x, y in pixels; z in MediaPipe's relative units
    transform: np.ndarray  # (4, 4) canonical-face → metric camera frame (OpenGL axes, cm)
    blendshapes: dict[str, float]


@dataclass
class LandmarkResult:
    faces: list[FaceObservation]
    image_size: tuple[int, int]  # (width, height)


class FaceLandmarkerWrapper:
    def __init__(self, model_path: str | Path, max_faces: int = 2, blendshapes: bool = False):
        """blendshapes: 52 expression scores from an extra model per face; unused by the gaze
        pipeline, so off by default (latency)."""
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Download it with:\n  curl -L -o {model_path} "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,  # >1 so we can detect a second person
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> LandmarkResult:
        import cv2

        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(image, ts)

        h, w = frame_bgr.shape[:2]
        faces: list[FaceObservation] = []
        for i, lms in enumerate(res.face_landmarks or []):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            transform = (
                np.asarray(res.facial_transformation_matrixes[i], dtype=np.float64)
                if res.facial_transformation_matrixes
                else np.eye(4)
            )
            bs = (
                {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
                if res.face_blendshapes
                else {}
            )
            faces.append(FaceObservation(pts, transform, bs))
        # Largest face first (the examinee is normally closest to the camera).
        faces.sort(key=lambda f: -float(np.ptp(f.landmarks_px[:, 0])))
        return LandmarkResult(faces, (w, h))

    def close(self) -> None:
        self._landmarker.close()


class FaceCounter:
    """Lightweight multi-person check: MediaPipe BlazeFace (≈1.6 ms) every `every` frames.

    Asking the landmarker for 2 faces makes it re-run its detector on every frame while only one
    face is present (+1.35 ms/frame on the M2); a second person only needs to be noticed within
    ~0.5 s, so a periodic detector is enough."""

    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")

    def __init__(self, model_path: str | Path, every: int = 3, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found. Download it with:\n  curl -L -o {model_path} {self.MODEL_URL}")
        self._mp = mp
        self._det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, min_detection_confidence=min_confidence))
        self.every = max(1, every)
        self._n = 0
        self._last = 0
        self._last_ts = -1
        # Runs on a worker thread in parallel with the landmarker (MediaPipe releases the GIL), so it
        # doesn't add to the per-frame latency; the count lags by ≤1 detection period.
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    def _detect(self, frame_bgr: np.ndarray, ts: int) -> int:
        import cv2

        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return len(self._det.detect_for_video(img, ts).detections)

    def count(self, frame_bgr: np.ndarray, timestamp_ms: int) -> int:
        """Latest known number of faces; a new detection is started every `every` calls."""
        if self._pending is not None and self._pending.done():
            self._last = self._pending.result()
            self._pending = None
        if self._n % self.every == 0 and self._pending is None:
            ts = max(int(timestamp_ms), self._last_ts + 1)
            self._last_ts = ts
            self._pending = self._pool.submit(self._detect, frame_bgr.copy(), ts)
        self._n += 1
        return self._last

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._det.close()
