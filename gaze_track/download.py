"""Download the model files the pipeline needs, verifying each by SHA-256.

    uv run python -m gaze_track download-models            # MediaPipe models + the released gaze model
    uv run python -m gaze_track download-models --no-gaze  # MediaPipe models only (geometric estimator)

Gaze model weights: trained on MPIIFaceGaze, so they are released under CC BY-NC-SA 4.0
(non-commercial, attribution, share-alike) like the dataset — see MODEL_CARD.md.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "iyush05/gaze-track"
RELEASE_TAG = "v0.1.0"
GAZE_MODEL_DIR = Path("models/gaze/mnv4s-mpii-v1")


@dataclass(frozen=True)
class Asset:
    dest: Path
    sha256: str
    url: str | None = None  # direct download
    release_name: str | None = None  # asset name in this repo's GitHub release


MEDIAPIPE = [
    Asset(Path("models/face_landmarker.task"),
          "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
          url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    Asset(Path("models/blaze_face_short_range.tflite"),
          "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
          url="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
]

GAZE = [  # LearnedGaze loads best.pt (config) and uses the ONNX file next to it
    Asset(GAZE_MODEL_DIR / "best.pt", "cfa6cc33e4a08f778188a9c22fb5cdc32989e5ecdf6bb747ec90f0ef336c603d",
          release_name="mnv4s-mpii-v1.pt"),
    Asset(GAZE_MODEL_DIR / "best.onnx", "131a120aed95582f7d33c5388177f7f4d498f09df29c924c92fdbdc75fd2fec2",
          release_name="mnv4s-mpii-v1.onnx"),
    Asset(GAZE_MODEL_DIR / "best.int8.onnx", "b3292306057bb11c493152b2a547aee863ede45b283b1135dcdfc2099dc0f65a",
          release_name="mnv4s-mpii-v1.int8.onnx"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_url(url: str, out: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)


def _fetch_release(name: str, out: Path) -> None:
    """Public repo → plain HTTPS; private repo (or no network auth) → the GitHub CLI's login."""
    url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{name}"
    try:
        _fetch_url(url, out)
        return
    except Exception as e:  # 404 while the repo is private, or offline
        if shutil.which("gh") is None:
            raise RuntimeError(f"could not download {url} ({e}); install the GitHub CLI and `gh auth login`, "
                               f"or download it by hand into {out.parent}") from e
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["gh", "release", "download", RELEASE_TAG, "--repo", REPO, "--pattern", name, "--dir", tmp],
                       check=True, capture_output=True)
        shutil.move(str(Path(tmp) / name), out)


def ensure(asset: Asset, force: bool = False) -> str:
    """Download `asset` if missing or corrupt; returns 'ok' | 'downloaded'."""
    if asset.dest.exists() and not force and sha256(asset.dest) == asset.sha256:
        return "ok"
    asset.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = asset.dest.with_suffix(asset.dest.suffix + ".part")
    try:
        if asset.url:
            _fetch_url(asset.url, tmp)
        else:
            _fetch_release(asset.release_name, tmp)
        got = sha256(tmp)
        if got != asset.sha256:
            raise RuntimeError(f"{asset.dest.name}: checksum mismatch (got {got[:12]}…, expected {asset.sha256[:12]}…)")
        tmp.replace(asset.dest)
    finally:
        tmp.unlink(missing_ok=True)
    return "downloaded"


def cmd_download(args) -> None:
    assets = MEDIAPIPE + ([] if args.no_gaze else GAZE)
    failed = False
    for a in assets:
        try:
            status = ensure(a, force=args.force)
            print(f"  {status:10s} {a.dest}")
        except Exception as e:
            failed = True
            print(f"  FAILED     {a.dest}: {e}", file=sys.stderr)
    if failed:
        sys.exit(1)
    if not args.no_gaze:
        print(f"\nGaze model ready. Calibrate with it:\n  uv run python -m gaze_track calibrate --profile me "
              f"--model {GAZE_MODEL_DIR / 'best.pt'}")


def add_parser(sub) -> None:
    c = sub.add_parser("download-models", help="download + verify MediaPipe models and the released gaze model")
    c.add_argument("--no-gaze", action="store_true", help="MediaPipe models only (geometric estimator)")
    c.add_argument("--force", action="store_true", help="re-download even if present and valid")
    c.set_defaults(fn=cmd_download)
