"""Export recorded sessions as normalized training data (Phase 3 fine-tuning / Phase 4 personalization).

Output: data/normalized/<session>.npz with
    patches     (N, roi, roi, 3) uint8 BGR normalized face patches
    gaze        (N, 2) float32 normalized gaze (pitch, yaw) rad, ETH convention; NaN where unknown
    head        (N, 2) float32 normalized head pose (pitch, yaw) rad, ETH convention
    R           (N, 3, 3) float32 normalization rotations (to de-normalize predictions)
    face_center (N, 3) float32 cm
    zone        (N,) str  ground-truth zone ('' for calib/honest)
    segment, task (N,) str;  trial_id, frame (N,) int
    subject     str;  split  str (subject-wise)
Gaze labels exist only for exact on-screen targets (calib + screen segments).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..geometry import GL_TO_CV
from .normalize import (
    face_center_cv,
    head_rotation_cv,
    mediapipe_intrinsics,
    normalize,
    normalize_gaze,
    to_eth_pitchyaw,
)
from .session import load_session, subject_split


def _target_cam(meta: dict, u: float, v: float) -> np.ndarray:
    s = meta["screen"]
    x = -(u - s["width_cm"] / 2.0 - s.get("camera_x_offset_cm", 0.0))
    return np.array([x, v + s["camera_to_top_cm"], 0.0])


def export_session(path: str | Path, out_dir: str | Path = "data/normalized", roi: int = 112) -> Path:
    s = load_session(path)
    arr = s.arrays()
    T, L = arr["transform"], arr["landmarks"].astype(np.float32)
    cam = s.meta["camera"]
    K = None
    out = {k: [] for k in ("patches", "gaze", "head", "R", "face_center", "zone", "segment", "task", "trial_id", "frame")}
    for i, r in enumerate(s.rows):
        if np.isnan(T[i]).any() or np.isnan(L[i]).any():
            continue
        img = cv2.imread(str(s.frame_path(r["frame"])))
        if img is None:
            continue
        if K is None:
            K = mediapipe_intrinsics(img.shape[1], img.shape[0], cam.get("assumed_vertical_fov_deg", 63.0))
        depth = float((GL_TO_CV @ T[i][:3, 3])[2])
        if depth <= 5:
            continue
        fc = face_center_cv(L[i], depth, K)
        head_R = head_rotation_cv(T[i])
        n = normalize(img, K, head_R, fc, roi=roi)

        gaze = (math.nan, math.nan)
        if r["segment"] in ("calib", "screen") and r["u_gt"] is not None:
            g = _target_cam(s.meta, r["u_gt"], r["v_gt"]) - fc
            gaze = to_eth_pitchyaw(normalize_gaze(g, n.R))
        # Head pose as the direction the face points (−z of head frame = toward camera).
        hf = n.head_R_norm @ np.array([0.0, 0.0, -1.0])
        out["patches"].append(n.patch)
        out["gaze"].append(gaze)
        out["head"].append(to_eth_pitchyaw(hf))
        out["R"].append(n.R)
        out["face_center"].append(fc)
        out["zone"].append(r["zone_gt"] or "")
        out["segment"].append(r["segment"])
        out["task"].append(r["task"])
        out["trial_id"].append(int(r["trial_id"]))
        out["frame"].append(int(r["frame"]))

    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{s.dir.name}.npz"
    np.savez_compressed(
        f,
        patches=np.array(out["patches"], np.uint8).reshape(-1, roi, roi, 3),
        gaze=np.array(out["gaze"], np.float32).reshape(-1, 2),
        head=np.array(out["head"], np.float32).reshape(-1, 2),
        R=np.array(out["R"], np.float32).reshape(-1, 3, 3),
        face_center=np.array(out["face_center"], np.float32).reshape(-1, 3),
        zone=np.array(out["zone"]),
        segment=np.array(out["segment"]),
        task=np.array(out["task"]),
        trial_id=np.array(out["trial_id"], np.int32),
        frame=np.array(out["frame"], np.int32),
        subject=s.subject,
        split=subject_split(s.subject),
    )
    return f


def cmd_export(args) -> None:
    from .session import list_sessions

    paths = args.sessions or list_sessions()
    if not paths:
        raise SystemExit("No sessions found in data/own.")
    for p in paths:
        f = export_session(p, args.out, args.roi)
        with np.load(f) as z:
            n, ng = len(z["gaze"]), int((~np.isnan(z["gaze"][:, 0])).sum())
            print(f"{Path(p).name}: {n} patches ({ng} with gaze labels), split={z['split']} → {f}")
        if args.preview:
            with np.load(f) as z:
                idx = np.linspace(0, len(z["patches"]) - 1, min(32, len(z["patches"]))).astype(int)
                tiles = [z["patches"][i] for i in idx]
            while len(tiles) % 8:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)])
            pv = f.with_suffix(".preview.jpg")
            cv2.imwrite(str(pv), grid)
            print(f"  preview grid → {pv}")


def add_parser(sub) -> None:
    c = sub.add_parser("export", help="export sessions as normalized face patches + gaze labels")
    c.add_argument("sessions", nargs="*")
    c.add_argument("--out", default="data/normalized")
    c.add_argument("--roi", type=int, default=112, help="patch size in px (ETH-XGaze uses 224)")
    c.add_argument("--preview", action="store_true", help="also write a JPEG grid of sample patches")
    c.set_defaults(fn=cmd_export)
