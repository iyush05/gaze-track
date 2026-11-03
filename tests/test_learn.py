import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gaze_track.data.normalize import from_eth_pitchyaw  # noqa: E402
from gaze_track.geometry import vector_to_angles  # noqa: E402
from gaze_track.learn.data import angular_error_deg, augment  # noqa: E402
from gaze_track.learn.infer import AffineCalibration, LearnedGaze  # noqa: E402
from gaze_track.learn.model import GazeNet, reparameterize  # noqa: E402


def test_model_forward_shapes_and_reparam_equivalence():
    m = GazeNet("mobileone_s0", pretrained=False).eval()
    x = torch.randint(0, 255, (2, 112, 112, 3), dtype=torch.uint8)
    with torch.no_grad():
        a = m(x)
        b = reparameterize(m)(x)
    assert a.shape == (2, 4)
    assert torch.allclose(a, b, atol=1e-3)  # reparameterization must not change outputs


def test_flip_augmentation_negates_yaw_only():
    torch.manual_seed(0)
    x = torch.zeros((64, 8, 8, 3), dtype=torch.uint8)
    x[:, :, :4] = 255  # bright left half → after a horizontal flip, bright right half
    g = torch.tensor([[0.1, 0.3]]).repeat(64, 1)
    h = torch.tensor([[-0.2, 0.4]]).repeat(64, 1)
    img, g2, h2 = augment(x, g, h)
    flipped = img[:, :, :, 4:].mean((1, 2, 3)) > img[:, :, :, :4].mean((1, 2, 3))
    assert 10 < flipped.sum() < 54
    assert torch.allclose(g2[flipped, 1], -g[flipped, 1]) and torch.allclose(g2[~flipped, 1], g[~flipped, 1])
    assert torch.allclose(g2[:, 0], g[:, 0]) and torch.allclose(h2[flipped, 1], -h[flipped, 1])


def test_torch_angular_error():
    p = torch.tensor([[0.0, 0.0], [0.0, math.radians(10)]])
    t = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    e = angular_error_deg(p, t)
    assert e[0] == pytest.approx(0, abs=0.05) and e[1] == pytest.approx(10, abs=1e-2)  # acos clamp floor ≈0.03°


def test_to_camera_angles_identity_rotation_matches_conventions():
    yaw, pitch = LearnedGaze.to_camera_angles(np.array([0.1, -0.2]), np.eye(3))
    ry, rp = vector_to_angles(from_eth_pitchyaw(0.1, -0.2))
    assert (yaw, pitch) == pytest.approx((math.degrees(ry), math.degrees(rp)))
    assert yaw == pytest.approx(math.degrees(0.2), abs=1e-6)  # ETH yaw sign is opposite


def test_affine_calibration_removes_personal_offset():
    rng = np.random.default_rng(0)
    true = np.c_[rng.uniform(-20, 20, 200), rng.uniform(-5, 25, 200)]
    pred = true + np.array([3.0, -4.0]) + rng.normal(0, 0.5, (200, 2))  # constant personal bias
    cal = AffineCalibration.fit(pred, true, groups=np.repeat(np.arange(10), 20))
    assert cal.stats["raw_mean_err_deg"] > 4.5
    assert cal.stats["lopo_mean_err_deg"] < 1.2
    y, p = cal.apply(*pred[0])
    assert np.hypot(y - true[0, 0], p - true[0, 1]) < 2.0


def test_train_smoke_on_synthetic_data(tmp_path):
    from gaze_track.learn.train import train

    rng = np.random.default_rng(0)
    d = tmp_path / "norm"
    d.mkdir()
    for subj in ["mpii_p00", "mpii_p03"]:
        n = 64
        np.savez(d / f"{subj}.npz", patches=rng.integers(0, 255, (n, 32, 32, 3), dtype=np.uint8),
                 gaze=rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32),
                 head=rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32), R=np.zeros((n, 3, 3), np.float32),
                 face_center=np.zeros((n, 3), np.float32), zone=np.array([""] * n), segment=np.array(["m"] * n),
                 task=np.array(["s"] * n), trial_id=np.zeros(n, np.int32), frame=np.arange(n), subject=subj)
    args = SimpleNamespace(seed=0, device="cpu", data=[str(d)], val=["p00"], limit=None, backbone="mobilenetv4_conv_small",
                           no_pretrained=True, lr=1e-3, batch=32, epochs=1, out=str(tmp_path / "runs"))
    out = train(args)
    assert (out / "best.pt").exists() and (out / "summary.json").exists()
    est = LearnedGaze(out / "best.pt")
    assert est.predict_patches(rng.integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)).shape == (3, 2)
