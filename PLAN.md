# Real-Time Edge Distraction & Gaze Tracking: Project Plan

Fine-tune a compact vision backbone for **sub-10 ms** head pose, eye-gaze vector, and
**examinee attentiveness** classification on a live webcam stream. The system flags when the
examinee looks at the screen edges, slightly off-screen (notes, phone), or fully away.

## 0. Project decisions (fixed)

| Decision | Choice |
|---|---|
| Deployment target | **Laptop CPU** (dev machine: MacBook Air **M2, 8 GB**) |
| Project type | **Research**, so non-commercial datasets (ETH-XGaze, Gaze360, MPIIFaceGaze…) are OK |
| Camera | **One webcam above the screen** (built-in, top-center) |
| Monitors | **Single monitor**. A second-screen glance is just `NEAR_OFF`/`FAR_OFF` |
| Training compute | M2 (PyTorch MPS) works for compact models (MobileNetV4-small: 1.2 min/epoch on MPIIFaceGaze); **Kaggle/Colab free tier** (T4/P100) for teacher training and big datasets |
| Inference runtime | ONNX Runtime CPU (with XNNPACK / CoreML EP on Mac as an option) |
| Python | 3.13 via `uv` (system Python 3.14 is too new for MediaPipe); mediapipe 1.2.0, opencv-contrib 5.0 |

## 1. Problem framing

The model alone can't tell you "slightly off-screen"; that depends on geometry. Cross-dataset
gaze error is ~4–6°. At 60 cm, 1° ≈ 1 cm on the screen plane, so the raw error is about the
size of the edge band. Therefore:

1. **Per-user calibration** at session start (9 points, ~20 s) gives the largest accuracy gain.
2. **Head pose + eye-in-head gaze** combine into one 3D gaze ray, which is intersected with the
   calibrated screen plane.
3. **Temporal reasoning** (dwell, frequency, direction clustering) decides events, not single frames.

### Per-frame zones
| Zone | Meaning |
|---|---|
| `ON_SCREEN` | Gaze point inside screen (minus edge band) |
| `EDGE` | Within ±`edge_cm` of the screen border |
| `NEAR_OFF` | Outside screen by `edge_cm`..`near_off_cm` |
| `FAR_OFF` | Further out, or head yaw/pitch beyond limits |
| `DOWN_KEYBOARD` | Below screen bottom within keyboard band (allowed) |
| `EYES_CLOSED` / `NO_FACE` / `MULTI_FACE` / `UNKNOWN` | Special states (`UNKNOWN` = low confidence) |

### Temporal events
`brief_glance` (normal), `sustained_off_screen`, `repeated_directional_glance` (suspicious),
`face_absent`, `multiple_faces`, `eyes_closed_long`.

## 2. Architecture

```
Webcam (30 fps) → face detect + track → landmarks / crop normalization (ETH-XGaze style)
  → multi-task backbone [head pose (6D rot) | gaze (binned+reg) | eye state | uncertainty]
  → denormalize → 3D gaze ray → screen-plane intersection (cm) → per-user calibration correction
  → zone classifier (+ One-Euro smoothing, hysteresis) → temporal events → log / UI
```

Latency budget (excl. capture): detect (amortized) ≤1 ms · preprocessing ≤1 ms · backbone 2–5 ms ·
post ≤0.5 ms · ~3 ms headroom → **< 10 ms p95**.

## 3. Models

- **Teacher** (Kaggle/Colab): ResNet-50 or ConvNeXt-T on ETH-XGaze (224 px subset) + Gaze360 + 300W-LP.
- **Student candidates** (benchmark on M2 CPU): **MobileOne-S0/S1**, **RepViT-M0.9**,
  **EfficientViT-M0**, **FastViT-T8**; tiny pure-ViT as ablation. Input 96/112/128 px face + optional eye crops.
- **Losses**: angular + L2CS-style binned CE for gaze; geodesic on 6D rotation for head pose;
  BCE eye state; Gaussian NLL uncertainty; KD (logit + feature) from teacher.
- **Compression**: reparameterize → QAT INT8 → ONNX → ONNX Runtime CPU.

## 4. Data

| Purpose | Datasets |
|---|---|
| Gaze | ETH-XGaze, Gaze360, MPIIFaceGaze, EYEDIAP |
| Head pose | 300W-LP (train), AFLW2000, BIWI (test) |
| Eye state | CEW, MRL Eye |
| **Own data** | Webcam sessions logged by this repo's recorder: dot targets + scripted off-screen targets (sticky notes at measured positions, phone, person beside), varied light, glasses on/off |

Compute notes: on the M2 8 GB, prototype with **MPIIFaceGaze** (small) and our own recordings.
ETH-XGaze and Gaze360 are large, so preprocess and train the teacher on Kaggle (persistent datasets,
~30 GPU-h/week). Precompute normalized crops at low resolution to keep storage and RAM small.
Always split **by subject**.

## 5. Roadmap

### Phase 0: Geometric baseline, no training ✅ done
- MediaPipe Face Landmarker (478 landmarks incl. iris + facial transformation matrix + blendshapes).
- Head pose from transform matrix; eye-in-head from iris position relative to eye corners and lids.
- Screen geometry (camera-centric, cm), ray–plane intersection.
- 9-point calibration UI → ridge regression to gaze angles → saved per-user profile.
- Zone logic with One-Euro smoothing + hysteresis; rules-based temporal events; JSONL event log.
- Live overlay with screen minimap; per-stage latency benchmark; per-frame CSV recorder (seeds Phase 1 data).
- **Exit criteria**: working demo; measured baseline zone accuracy & latency on M2.
- **Result (subject s01, glasses)**: calibration LOPO 4.8°; landmarks+pipeline p50 11.2 ms / p95 13.7 ms.
  Vertical (pitch) error dominates; EDGE↔NEAR_OFF limited by accuracy (spatial hysteresis + glance
  merging added to suppress flicker and false repeated-glance alerts).

### Phase 1: Data & evaluation harness ⏳ *1a built, awaiting recordings*
**1a: own data + harness (built)**
- `collect`: guided protocol: 9-pt self-calibration, 24 on-screen dots (half in the edge band), 17 cued
  off-screen trials (near L/R/top, far L/R/up, keyboard, eyes closed, no face) with audio cues.
- `collect-honest`: calibration, then headless recording of normal work → false alerts/hour.
- Session format `data/own/<id>/` (frames, labels, MediaPipe transforms + landmarks); subject-wise splits.
- `eval`: gaze error (deg/cm, interior vs edge band, yaw/pitch bias), per-frame zone confusion + macro-F1,
  off-screen binary F1, trial majority-vote accuracy, latency; honest sessions → alerts/hour.
- `export`: ETH-XGaze-style normalized face patches + normalized gaze/head labels (`data/normalized/`).
- Target data: ≥ 3 protocol sessions (glasses on/off, 2 lighting conditions) + ≥ 30 min honest work.

**1a results: subject s01 (glasses), first protocol + 10-min honest session**
| | Phase 0 initial | after fixes |
|---|---|---|
| On-screen gaze error | 3.0° (2.0 cm) | 3.2° (2.2 cm) |
| Off-screen precision / recall | 0.95 / 0.83 | 0.97 / 0.82 |
| Keyboard trials correct | 53% (read as eyes closed) | 100% |
| Honest false flags | 48 / h (8 in 10 min) | 0 in 10 min |
| Landmark latency p50 | 8.0 ms | 8.0 ms |
Fixes (all justified by s01 data): edge band as an angle (6°, ≈ p90 error), +10° tolerance above the
screen, eyes-closed threshold relative to calibration EAR, head weight 0.5 (2D iris offset already
contains part of the head rotation → double counting).
Caveats: tuned and evaluated on the same subject; 0 events in 10 min only bounds the rate at
< 18 / h (95%). Needs ≥ 1 h honest data + a held-out second subject / condition.
Known limits for the learned model: eye-only gaze overshoots outside the calibrated range
(near_left true ≈ 31°, predicted ≈ 50°), so NEAR_OFF vs FAR_OFF is unreliable.

**1b: public datasets (MPIIFaceGaze done)**
- `prepare-mpii`: raw MPIIFaceGaze (0.94 GB; the 22 GB pre-normalized release doesn't fit on disk) →
  normalized with our own `normalize.py`, so public and own data share one format
  (`data/normalized/mpiifacegaze/pXX.npz`, 112 px, ETH pitch/yaw, cm).
- Head-frame axis conventions are verified from the data (eye corners must land above mouth
  corners after warping): x_sign = +1 in 200/200 samples.
- 15 subjects, 37,667 samples (the annotation files have 37,667 lines, not the commonly quoted 45k).
- `data/index.py`: combines any normalized files, lazy patch loading, leave-one-person-out folds.
- ETH-XGaze (Phase 2 teacher) → Kaggle; needs a registration form.

**Original scope**
- Dataset loaders, ETH-XGaze normalization, subject-wise splits.
- Recording protocol + labeling for own data (known target positions → ground-truth zones).
- Metrics: gaze angular error, head-pose MAE, zone macro-F1 & confusion, **false flags / exam-hour**,
  latency p50/p95/p99.

### Phase 2a: first learned model on the M2 ✅ (results below)
- `train`: timm backbone → gaze + head (ETH pitch/yaw), L1, flip/photometric augmentation, MPS.
  Held-out MPIIFaceGaze persons p00–p02. `export-onnx` verifies ONNX vs PyTorch and times ONNX Runtime.
- Backbone choice (measured on M2): **MobileNetV4-conv-small**: 1.15 ms ONNX Runtime CPU batch-1,
  1.2 min/epoch on MPS. MobileOne-S0 trains 6.5× slower on MPS (and its ONNX export mismatched);
  PyTorch eager CPU is ~60× slower than ONNX Runtime on Apple Silicon (68 ms) → always deploy via ORT.
- MPIIFaceGaze held-out persons: **5.08°** (15 epochs, best at 9; per-person 3.3–6.9°).
- On s01's sessions the raw model's yaw is r = 0.98 with true gaze but **compressed ×0.31–0.41**
  (pitch ×0.7): likely head-orientation shortcut learned from MPII (people turn their heads; s01 moves
  only the eyes). Not a crop-scale problem (landmark scale ratio own/MPII 1.02–1.13).
- Per-user affine calibration must fit slopes (λ chosen by leave-one-point-out).
- Held-out screen-dot error (calibration fit on the 9 calib dots only):
  | | baseline | learned v1 | **fused (ridge on both)** |
  |---|---|---|---|
  | s01 glasses | 3.24° | 3.20° | **1.97°** |
  | s01 no glasses | 2.94° | 3.97° | **2.53°** |
  Learned alone: worse off-screen recall (0.58/0.70) and 8 false flags in the 10-min honest session
  (amplified noise from the ×3 slope), so it is **not** used alone.
- **Fused estimator (built; `calibrate --model latest`, `eval --model … --fuse`)**: per-user ridge on
  [mapper yaw/pitch, learned yaw/pitch], shrunk toward an equal blend, **plus an off-screen veto**: if
  the mapper alone says NEAR_OFF / FAR_OFF / keyboard, that wins (fusion alone dropped off-screen
  recall to 0.63 / 0.83 because the blend is fitted on on-screen dots only).
  | s01 | baseline | **fused + veto** |
  |---|---|---|
  | gaze error glasses / no glasses | 3.21° / 2.89° | **1.94° / 2.50°** |
  | off-screen F1 glasses / no glasses | 0.88 / 1.00 | **0.88 / 1.00** |
  | zone macro-F1 glasses / no glasses | 0.62 / 0.81 | 0.64 / 0.80 |
  | honest false flags (10 min) | 0 | 0 |
  | end-to-end latency p50 / p95 (M2 CPU) | 8.1 / 8.3 ms | 10.1 / 11.0 ms |
  Latency: the learned path costs 1.4 ms; **MediaPipe Face Landmarker is 8–8.7 ms**, so the < 10 ms
  goal now hinges on replacing/lightening the landmark stage (Phase 3), not on the gaze model.
- Next: (1) reduce the eye-only compression: eye-crop branch / higher-res eyes, train with own
  labelled sessions (leave-one-session-out), ETH-XGaze; (2) lighter face/landmark stage for latency;
  (3) more subjects + ≥ 1 h honest data to validate.

### Phase 2: Teacher (Kaggle/Colab)
- Multi-task teacher; targets ≤4.5° gaze (within-dataset), ≤4° head-pose MAE (AFLW2000).

### Phase 3a: real-time budget ✅ (< 10 ms end-to-end on the M2 CPU)
Re-scoped from "shrink the gaze model": it already ran in 1.15 ms; MediaPipe was 8–8.7 ms.
Measured by replaying 1,800 recorded frames (`bench --session … --profile me`):
| change | effect |
|---|---|
| blendshapes off | ≈0 ms (unused anyway) |
| landmarker 1 face + BlazeFace face counter every 3rd frame on a worker thread | −1.35 ms/frame; second person still checked at 10 Hz |
| ONNX Runtime 2 threads (not 4) | model 2.0 vs 1.4 ms, but MediaPipe −0.9 ms from less core contention |
| INT8 gaze model | 0.71 vs 1.17 ms, +0.05° on MPIIFaceGaze, **but fused s01 1.94° → 2.47°** (per-user slope ×3 amplifies quantization error) → kept FP32, INT8 optional |
Result (fused estimator, FP32): **p50 8.91 ms, p95 9.18 ms, p99 9.47 ms** (was 10.1 / 11.0).
Breakdown p50: landmarks 6.82, learned 2.00, rest < 0.1 ms.
Not yet tested: multi-person detection with a real second person (unit-tested only).

### Phase 3: Compact student + distillation + INT8
- 2–3 backbones × 3 input sizes; KD; QAT; ONNX. Fine-tune on M2 (MPS) where it fits.
- Target: ≤1° degradation vs teacher, ≤5 ms model latency on M2 CPU.

### Phase 4a: personal fine-tuning ✅ (cross-session validated)
Fine-tune the MPIIFaceGaze model on MPIIFaceGaze + one own session (oversampled ×8, lr 3e-4,
4 epochs), test on the *other* session (different glasses condition), with fusion + veto:
| held-out session | raw yaw slope (ideal 1) | raw error (no calib) | fused error base → personal | off-screen F1 | honest flags |
|---|---|---|---|---|---|
| no glasses (fold A) | 0.41 → **0.93** | 8.15° → **3.85°** | 2.50° → 2.58° | 1.00 → 1.00 | 0 → 0 |
| glasses (fold B) | 0.31 → **0.70** | 10.86° → **4.83°** | 1.94° → **1.49°** | 0.88 → 0.88 | 0 → 0 |
- Personal data removes most of the eye-only compression and halves uncalibrated error.
- After calibration: −23% on one fold, no change on the other (that session has a +1.5–1.8° yaw bias
  under every model → recording geometry, not the model). No regressions in detection or false flags.
- `personalize --subject s01`: exports all of a subject's sessions and fine-tunes (3 epochs, keep last).
- Open: fewer calibration points with a personal model (uncalibrated error is already ~4°);
  per-session geometry bias (user position / camera offset estimation).

### Capacity test: does a bigger backbone help? ✅ no (for now)
Question: is a Kaggle-trained large model worth it? Same MPIIFaceGaze split (held-out p00–p02).
| backbone | best epoch (val-selected) | mean of last 5 epochs | ONNX CPU batch-1 |
|---|---|---|---|
| MobileNetV4-small | 5.08° | 5.57° | 1.15 ms |
| ResNet-18 | 4.11° (epoch 3) | 5.47° | 4.4 ms idle |
- The val-selected "best" is optimistic (val swings up to 2° between epochs); averaged, the larger
  model is only ~0.1° better on unseen MPII people.
- **Decision criterion: personalized folds on s01** (same protocol as Phase 4a; fused + veto,
  calibration on the held-out session's own dots):
  | held-out session | raw error MNV4 / R18 | fused error MNV4 / R18 | off-screen F1 | honest flags |
  |---|---|---|---|---|
  | no glasses (fold A) | 3.80° / **2.68°** | 2.58° / **2.07°** | 1.00 / 1.00 | 0 / 0 |
  | glasses (fold B) | **4.78°** / 5.89° | **1.49°** / 1.81° | 0.88 / 0.88 | 0 / 0 |
  | mean | 4.29° / 4.29° | 2.04° / 1.94° | | |
- **Conclusion: model capacity is not the bottleneck.** ResNet-18 wins one fold and loses the other;
  the mean difference (0.1°) is within noise for 2 folds of one person, detection is identical, and
  it costs ~4× CPU latency (can't run live within 10 ms). A capacity-only Kaggle teacher is not
  justified. Still untested: a teacher on *more diverse data* (ETH-XGaze), which targets a different
  weakness (eye-only gaze, head/gaze decorrelation) than capacity does.
- RepViT-M0.9 is unusable in this setup: eval-mode predictions collapse (19.9° vs 8.9° with batch
  statistics on the same weights; val 22° → 42° while train loss fell). BatchNorm recalibration only
  partly helps (12.2°). MobileNetV4 was checked for the same issue and is fine (its eval/train gap
  5.37° vs 4.40° is per-person batch-statistics adaptation; recalibration doesn't improve it).

### Phase 4: Personalization & geometry refinement
- Calibration-fitted adapter (affine / ~1k-param MLP); target ≤2–2.5° post-calibration.

### Phase 5a: behaviour data + event-layer baseline ✅
- `collect-behavior`: 9-dot calibration, then 6 scripted ~30–40 s segments with segment labels:
  honest reading / typing / thinking; cheating: notes beside the screen, phone in lap, asking a
  person beside. `eval` on such a session → per segment: flagged?, time to first flag, flag kinds.
- s01, 2 sessions (glasses, notes on the left), rules engine:
  | | before fixes (baseline / fused) | after fixes (baseline / fused) |
  |---|---|---|
  | cheating segments flagged | 5/6 / 4/6 | **5/6 / 5/6** |
  | honest segments falsely flagged | 1/6 / 1/6 | **0/6 / 0/6** |
  | honest 10-min session | 0 flags | 0 flags |
  | guided sessions zone F1 / off-screen F1 | — | unchanged |
- Fixes (each traced to a failure in the data):
  1. Keyboard band 35° → **25°** below the screen bottom: keyboard looks/typing never exceeded 22°,
     a phone in the lap was at 34–38° (median) and was being read as "keyboard".
  2. Repeated-glance "same spot" 30° → **15°**: up-left and up-right thinking glances were merged
     into one pattern (honest_thinking false flag).
  3. Veto priority off-screen > keyboard > on-screen: the learned model flattens steep downward
     gaze, so the fused estimate called the lap "keyboard".
- Remaining miss: notes placed close beside the screen (session 1): glance yaw ≤ 19.5°, inside the
  range of honest gaze (thinking reached 24°) → not separable by direction at current accuracy;
  this is the case a learned temporal model (frequency/rhythm of glances) would have to address.
- Held-out session 3 (no glasses, notes on the **right**; recorded after the fixes): cheating 3/3,
  but honest_thinking falsely flagged (3.1 s look up-right) and the lap phone caught only as
  "eyes closed" (without glasses, looking down lowers the lids: EAR 0.16× baseline; keyboard 0.44×,
  51 % of keyboard frames read as closed). Second round of fixes (data across all 3 sessions):
  4. Eyes-closed threshold 0.45 → **0.25** × baseline EAR (truly closed ≈ 0.06×).
  5. Low EAR + head pitched down > 6° vs neutral = **looking down → off-screen**, not eyes closed
     (lap +9.1/+9.6°, keyboard/typing −1.4…−3.0°).
  6. **Head-turn events**: head > 25° from neutral for ≥ 1.5 s → warning, 2 within 60 s → alert
     (thinking ≤ 4.7°, asking someone 33–50° but only 1.1–3.1 s per turn).
  7. Looking clearly **up** with the head forward needs 8 s (not 3 s) to count as sustained
     (both thinking false flags were upward looks).
- **Current (all 3 sessions, baseline and fused identical):** cheating 8/9 segments, honest 0/9,
  median time to first flag 5–6.5 s, honest 10-min session 0 flags; no-glasses guided zone F1
  0.806 → 0.875, others unchanged. Remaining miss: notes right beside the screen (session 1).
- Caveat: session 3 was held out for the first fixes but informed the second round; all from 1
  person. Next held-out check: new sessions (other person, other lighting).

### Phase 5: Learned temporal model
- Tiny GRU/TCN over 2–5 s windows on [gaze xy, head pose, uncertainty, eye state, zone];
  off-screen direction clustering for repeated-glance detection.

### Phase 6a: robustness stress test + image-quality gating ✅
- `robustness`: re-runs the full pipeline (MediaPipe, features, learned model) on guided sessions
  with deterministic degradations; nothing from the clean frames is reused.
- Baseline estimator, glasses / no-glasses (face found in 100 % of frames under *every* condition,
  so failures are silent):
  | condition | gaze error | off-screen recall | verdict |
  |---|---|---|---|
  | clean | 3.15° / 3.84° | 0.82 / 0.96 | — |
  | overexposed, noise, 320×240 | ≈ clean | ≈ clean | harmless |
  | dim, backlit | +0.5–0.9° | −0.01–0.09 | mild |
  | **blur** | 4.70° / 4.13° | 0.68 / 0.80 | harmful |
  | **JPEG q10** | 7.34° / 5.22° | 0.73 / 0.85 | harmful (streaming only) |
  | **very dark** | 8.83° / 6.01° | 0.49 / 0.72 | worst |
  The fused estimator degrades the same way (shares MediaPipe landmarks and the off-screen veto).
- Gating (`quality.py`): face luminance < 20 → too dark; whole-face sharpness < 0.009 → blurry;
  both → `UNKNOWN` + on-screen hint; `UNKNOWN` ≥ 10 s → `tracking_unreliable` warning, so dimming
  the light can't silently disable monitoring. Rejected signals: eye-region sharpness (drops with
  lowered lids: flagged 100 % of keyboard/eyes-closed frames), iris jitter (falls with blur, rises
  with harmless noise and with real eye movement).
  | condition | confidently wrong, ungated → gated | UNKNOWN (gated) |
  |---|---|---|
  | very dark | 64 % → **2.2 %** / 35 % → **0.1 %** | 98 % / 96 % |
  | blur | 38 % → **6.9 %** / 31 % → **10.2 %** | 86 % / 79 % |
  | clean / harmless | unchanged | 0–0.7 % (backlit no-glasses 6.7 %) |
  | JPEG q10 | unchanged (not detected) | 0 % |
  Real recordings (6 sessions): gating fires on **0 %** of frames.
- Open: JPEG/compression detection (only if frames are streamed), real dim-room validation,
  review report for human reviewers, an intermittent test failure seen once (not reproduced in 9 runs).

### Phase 6: Hardening
- Robustness: glasses, low light, backlight, skin tones, movement. Uncertainty gating → `UNKNOWN`.
- Fairness & accessibility review; human-in-the-loop for all flags; on-device only.

## 6. Success criteria (v1)
- End-to-end **< 10 ms p95** on M2 CPU (model ≤ 5 ms). ✅ 9.18 ms p95 (Phase 3a)
- **≤ 2.5°** gaze error after calibration on own test set.
- Zone macro-F1 **≥ 0.90** (ON_SCREEN / EDGE / NEAR_OFF / FAR_OFF).
- **< 1 false flag per exam-hour** on honest sessions.

## 7. Risks
| Risk | Mitigation |
|---|---|
| Gaze error > edge band | Calibration, eye crops, uncertainty gating |
| Eye-only glances | Eye branch; ETH-XGaze coverage |
| Glare / low light | Augmentation + own data; fall back to head pose with low confidence |
| False accusations | Optimize false flags/hour; human review; never auto-penalize |
| Strabismus / nystagmus / accessibility | Per-user calibration; accommodation path; subgroup eval |
| Privacy | Fully on-device; store events not video; consent |
| 8 GB RAM limits | Small batches, low-res precomputed crops, offload big training to Kaggle |

## 8. Repo layout
```
gaze-track/
├── PLAN.md
├── pyproject.toml            # uv, Python 3.12
├── models/                   # downloaded / exported model files
├── gaze_track/
│   ├── config.py             # screen geometry, thresholds
│   ├── landmarks.py          # MediaPipe FaceLandmarker wrapper
│   ├── features.py           # head pose, eye-in-head, eye openness
│   ├── geometry.py           # angles ↔ vectors, ray–plane intersection
│   ├── calibration.py        # calibration model + UI
│   ├── filters.py            # One-Euro filter
│   ├── zones.py              # zone classification + hysteresis
│   ├── events.py             # temporal rules engine
│   ├── pipeline.py           # per-frame pipeline + timing
│   ├── ui.py                 # overlay & minimap drawing
│   └── cli.py                # calibrate / run / bench
└── tests/
```
Later phases add `data/`, `train/`, `export/`, `benchmarks/`.
