import cv2
import numpy as np
import pytest
from scipy.io import savemat

from gaze_track.data import mpiifacegaze as M

# 6-point face model (mm), head frame with x = image-right, y = down, z = into the head
# (OpenCV-style): 4 eye corners at y=-35, 2 mouth corners at y=+35.
FACE = np.array([[-45, -35, 0], [-15, -35, 0], [15, -35, 0], [45, -35, 0], [-25, 35, 10], [25, 35, 10]], float)
K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])


def _make_person(tmp_path, flip_x: bool, n=12):
    d = tmp_path / "p00"
    (d / "Calibration").mkdir(parents=True)
    (d / "day01").mkdir()
    savemat(d / "Calibration" / "Camera.mat", {"cameraMatrix": K, "distCoeffs": np.zeros((1, 5))})
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n):
        rvec = np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)])
        # Rotating by ~π about y turns the head's z-axis (into the head) to point away from the camera,
        # so the face looks at the camera; the true x-axis then points to image-left. A dataset whose
        # model used the opposite x-axis is simulated by flip_x.
        R = cv2.Rodrigues(rvec)[0]
        t = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), 600.0])
        pts3 = FACE @ R.T + t
        lm = (pts3 @ K.T)[:, :2] / (pts3 @ K.T)[:, 2:]
        if flip_x:
            R = R @ np.diag([-1.0, 1.0, -1.0])  # same physical pose, x and z axes reversed
        rv = cv2.Rodrigues(R)[0].ravel()
        fc = pts3.mean(0)
        gt = np.array([rng.uniform(-150, 150), rng.uniform(20, 200), 0.0])
        img = np.full((720, 1280, 3), 80, np.uint8)
        cv2.imwrite(str(d / "day01" / f"{i:04d}.jpg"), img)
        vals = [*rng.uniform(0, 1000, 2), *lm.ravel(), *rv, *t, *fc, *gt]
        lines.append(f"day01/{i:04d}.jpg " + " ".join(f"{v:.6f}" for v in vals) + " left")
    (d / "p00.txt").write_text("\n".join(lines) + "\n")
    return d


@pytest.mark.parametrize("flip_x", [False, True])
def test_conventions_detected_and_output_upright(tmp_path, flip_x):
    d = _make_person(tmp_path, flip_x)
    rows = M.parse_annotations(d)
    assert len(rows) == 12 and rows[0]["landmarks"].shape == (6, 2) and rows[0]["eval_eye"] == "left"
    conv = M.check_conventions(d)
    # Whatever the dataset's axis convention, the chosen signs must give upright patches and a
    # nose direction pointing toward the camera.
    f = M.process_person(d, conv, roi=64, out_dir=tmp_path / "out")
    with np.load(f) as z:
        assert float(z["upright_frac"]) == 1.0
        assert len(z["patches"]) == 12 and z["patches"].shape[1:] == (64, 64, 3)
        assert np.all(np.isfinite(z["gaze"]))
        # Faces look roughly at the camera → normalized head pose near frontal.
        assert np.all(np.abs(np.degrees(z["head"])) < 35)


def test_index_combines_files_and_lopo(tmp_path):
    from gaze_track.data.index import load_index, lopo_folds

    for subj, n, with_gaze in [("a", 5, True), ("b", 3, False)]:
        g = np.zeros((n, 2), np.float32) if with_gaze else np.full((n, 2), np.nan, np.float32)
        np.savez(tmp_path / f"{subj}.npz", patches=np.full((n, 8, 8, 3), ord(subj), np.uint8), gaze=g,
                 head=np.zeros((n, 2), np.float32), R=np.zeros((n, 3, 3), np.float32),
                 face_center=np.zeros((n, 3), np.float32), zone=np.array([""] * n), segment=np.array(["x"] * n),
                 task=np.array(["t"] * n), trial_id=np.zeros(n, np.int32), frame=np.arange(n), subject=subj)
    idx = load_index([tmp_path])
    assert len(idx) == 8 and idx.subjects == ["a", "b"] and idx.has_gaze().sum() == 5
    p = idx.patches(np.array([6, 0, 7]))
    assert [int(x[0, 0, 0]) for x in p] == [ord("b"), ord("a"), ord("b")]
    folds = list(lopo_folds(idx))
    assert [f[0] for f in folds] == ["a", "b"] and all((tr ^ te).all() for _, tr, te in folds)
    sub = idx.subset(idx.subject == "b")
    assert len(sub) == 3 and int(sub.patches(np.arange(3))[0, 0, 0, 0]) == ord("b")
