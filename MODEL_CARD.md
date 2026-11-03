# Model card: mnv4s-mpii-v1 (gaze estimation)

| | |
|---|---|
| Architecture | MobileNetV4-conv-small (timm, ImageNet-pretrained) + linear head → gaze pitch/yaw + head pitch/yaw (rad, normalized camera) |
| Input | 112×112 BGR uint8 face patch, ETH-XGaze-style normalization (`gaze_track/data/normalize.py`) |
| Training data | MPIIFaceGaze, persons p03–p14 (28,920 samples); held out p00–p02 |
| Accuracy | 5.08° mean angular error on held-out persons (best epoch by validation; last-5-epoch mean 5.57°) |
| Latency | 1.15 ms FP32 / 0.71 ms INT8, batch 1, ONNX Runtime CPU (Apple M2) |
| Files | `best.pt` (PyTorch + config), `best.onnx` (FP32, used at run time), `best.int8.onnx` (optional) |
| Intended use | Research on on-device attention / gaze estimation, always with per-user calibration and human review of any flag |
| Not for | Commercial use (licence); automatic decisions about people |
| Limitations | Under-reacts to eye-only gaze (slope ≈0.3 on one test user before personalization); evaluated on 15 MPII people + 1 own subject; not evaluated across demographics or eye conditions |

## Licence
Trained on MPIIFaceGaze (Zhang et al., CVPR-W 2017), licensed CC BY-NC-SA 4.0. These weights are therefore released under
**CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike): https://creativecommons.org/licenses/by-nc-sa/4.0/.
The source code in this repository is MIT-licensed (see LICENSE).
