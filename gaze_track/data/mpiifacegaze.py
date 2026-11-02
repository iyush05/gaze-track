"""MPIIFaceGaze (Zhang et al., CVPR-W 2017) → normalized patches in the same format as `export`.

Dataset: 15 participants (p00–p14) × 3000 laptop-webcam images, gaze targets on the laptop screen.
License: CC BY-NC-SA 4.0 (research use). Download (0.94 GB):
    https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip

Per-person annotation file pXX/pXX.txt, one line per image (space separated):
    1        image path (e.g. day01/0005.jpg)
    2–3      gaze target on screen (px)
    4–15     six 2D facial landmarks (x, y): 4 eye corners, 2 mouth corners
    16–18    head rotation (Rodrigues vector), camera frame
    19–21    head translation (mm), camera frame
    22–24    face centre fc (mm), camera frame
    25–27    3D gaze target gt (mm), camera frame
    28       which eye is used in the evaluation subset ('left' / 'right')
Camera intrinsics: pXX/Calibration/Camera.mat (cameraMatrix, distCoeffs).

Output: data/normalized/mpiifacegaze/pXX.npz (same keys as data/export.py; units cm; ETH pitch/yaw).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np

from .normalize import normalize, normalize_gaze, to_eth_pitchyaw

RAW_ZIP = Path("data/raw/MPIIFaceGaze.zip")
RAW_DIR = Path("data/raw/MPIIFaceGaze")
OUT_DIR = Path("data/normalized/mpiifacegaze")
PERSONS = [f"p{i:02d}" for i in range(15)]


def extract_zip(zip_path: Path = RAW_ZIP, dest: Path = RAW_DIR.parent) -> Path:
    if RAW_DIR.exists() and any(RAW_DIR.glob("p00/*.txt")):
        return RAW_DIR
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if not RAW_DIR.exists():  # tolerate a differently named top-level folder
        cands = [p for p in dest.iterdir() if p.is_dir() and (p / "p00").exists()]
        if cands:
            cands[0].rename(RAW_DIR)
    return RAW_DIR


def load_camera(person_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    m = loadmat(person_dir / "Calibration" / "Camera.mat")
    return np.asarray(m["cameraMatrix"], float), np.asarray(m["distCoeffs"], float).ravel()


def parse_annotations(person_dir: Path) -> list[dict]:
    rows = []
    for line in (person_dir / f"{person_dir.name}.txt").read_text().splitlines():
        p = line.split()
        if len(p) < 27:
            continue
        v = np.array(p[1:27], dtype=float)
        rows.append({
            "image": p[0],
            "screen_px": v[0:2],
            "landmarks": v[2:14].reshape(6, 2),
            "head_rvec": v[14:17],
            "head_t_mm": v[17:20],
            "fc_mm": v[20:23],
            "gt_mm": v[23:26],
            "eval_eye": p[27] if len(p) > 27 else "",
        })
    return rows


def _warp_points(W: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.c_[pts, np.ones(len(pts))] @ W.T
    return h[:, :2] / h[:, 2:3]


def check_conventions(person_dir: Path, n: int = 200) -> dict:
    """Determine the head-frame sign conventions from the data itself.

    x_sign: the normalized face must be upright (eye corners above mouth corners) — with the head
            x-axis flipped, the patch is rotated 180°.
    z_sign: the 'nose' direction (sign · head z-axis) must point toward the camera (negative z)."""
    K, _ = load_camera(person_dir)
    rows = parse_annotations(person_dir)[:n]
    upright = {1: 0, -1: 0}
    nose_z = []
    for r in rows:
        hR = cv2.Rodrigues(r["head_rvec"])[0]
        fc = r["fc_mm"] / 10.0
        for xs in (1, -1):
            hR_s = hR.copy()
            hR_s[:, 0] *= xs
            W = normalize(np.zeros((8, 8, 3), np.uint8), K, hR_s, fc, roi=112).W
            lm = _warp_points(W, r["landmarks"])
            eyes_y, mouth_y = lm[:4, 1].mean(), lm[4:, 1].mean()
            upright[xs] += int(eyes_y < mouth_y)
        nose_z.append(hR[2, 2])  # z-component of the head z-axis in camera coords
    x_sign = 1 if upright[1] >= upright[-1] else -1
    z_sign = -1 if np.median(nose_z) > 0 else 1  # choose the sign that points toward the camera
    return {"x_sign": x_sign, "z_sign": z_sign, "upright_votes": upright, "n": len(rows)}


def process_person(person_dir: Path, conv: dict, roi: int = 112, out_dir: Path = OUT_DIR, limit: int | None = None) -> Path:
    K, _ = load_camera(person_dir)
    rows = parse_annotations(person_dir)
    if limit:
        rows = rows[:limit]
    keys = ("patches", "gaze", "head", "R", "face_center", "frame", "eval_eye")
    out = {k: [] for k in keys}
    upright_ok = 0
    for i, r in enumerate(rows):
        img = cv2.imread(str(person_dir / r["image"]))
        if img is None:
            continue
        hR = cv2.Rodrigues(r["head_rvec"])[0]
        hR[:, 0] *= conv["x_sign"]
        fc = r["fc_mm"] / 10.0
        n = normalize(img, K, hR, fc, roi=roi)
        lm = _warp_points(n.W, r["landmarks"])
        upright_ok += int(lm[:4, 1].mean() < lm[4:, 1].mean())
        g = (r["gt_mm"] - r["fc_mm"]) / 10.0
        nose = n.head_R_norm[:, 2] * conv["z_sign"]
        out["patches"].append(n.patch)
        out["gaze"].append(to_eth_pitchyaw(normalize_gaze(g, n.R)))
        out["head"].append(to_eth_pitchyaw(nose))
        out["R"].append(n.R)
        out["face_center"].append(fc)
        out["frame"].append(i)
        out["eval_eye"].append(r["eval_eye"])
    N = len(out["patches"])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{person_dir.name}.npz"
    np.savez_compressed(
        f,
        patches=np.array(out["patches"], np.uint8).reshape(-1, roi, roi, 3),
        gaze=np.array(out["gaze"], np.float32).reshape(-1, 2),
        head=np.array(out["head"], np.float32).reshape(-1, 2),
        R=np.array(out["R"], np.float32).reshape(-1, 3, 3),
        face_center=np.array(out["face_center"], np.float32).reshape(-1, 3),
        zone=np.array([""] * N),
        segment=np.array(["mpiifacegaze"] * N),
        task=np.array(["screen"] * N),
        trial_id=np.zeros(N, np.int32),
        frame=np.array(out["frame"], np.int32),
        eval_eye=np.array(out["eval_eye"]),
        subject=f"mpii_{person_dir.name}",
        split="loo",  # MPIIFaceGaze is evaluated leave-one-person-out
        upright_frac=upright_ok / max(1, N),
    )
    return f


def stats(f: Path) -> dict:
    with np.load(f) as z:
        g, h = np.degrees(z["gaze"]), np.degrees(z["head"])
        return {
            "n": int(len(g)),
            "gaze_pitch_deg": [float(np.percentile(g[:, 0], 5)), float(np.median(g[:, 0])), float(np.percentile(g[:, 0], 95))],
            "gaze_yaw_deg": [float(np.percentile(g[:, 1], 5)), float(np.median(g[:, 1])), float(np.percentile(g[:, 1], 95))],
            "head_pitch_deg": [float(np.percentile(h[:, 0], 5)), float(np.median(h[:, 0])), float(np.percentile(h[:, 0], 95))],
            "head_yaw_deg": [float(np.percentile(h[:, 1], 5)), float(np.median(h[:, 1])), float(np.percentile(h[:, 1], 95))],
            "upright_frac": float(z["upright_frac"]),
        }


def cmd_prepare(args) -> None:
    import json

    if not RAW_ZIP.exists() and not RAW_DIR.exists():
        raise SystemExit(f"Download first:\n  curl -L -o {RAW_ZIP} https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip")
    root = extract_zip()
    conv = check_conventions(root / "p00")
    print(f"head-frame conventions from data: {conv}")
    persons = args.persons or PERSONS
    for p in persons:
        f = process_person(root / p, conv, roi=args.roi, out_dir=Path(args.out), limit=args.limit)
        print(f"{p}: {json.dumps(stats(f))}")
        if args.preview:
            with np.load(f) as z:
                idx = np.linspace(0, len(z["patches"]) - 1, 24).astype(int)
                tiles = [z["patches"][i] for i in idx]
            grid = np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, 24, 8)])
            cv2.imwrite(str(f.with_suffix(".preview.jpg")), grid)


def add_parser(sub) -> None:
    c = sub.add_parser("prepare-mpii", help="extract + normalize MPIIFaceGaze into data/normalized/mpiifacegaze")
    c.add_argument("--persons", nargs="*", help="subset, e.g. p00 p01 (default: all 15)")
    c.add_argument("--roi", type=int, default=112)
    c.add_argument("--limit", type=int, default=None, help="max images per person (quick test)")
    c.add_argument("--out", default=str(OUT_DIR))
    c.add_argument("--preview", action="store_true")
    c.set_defaults(fn=cmd_prepare)
