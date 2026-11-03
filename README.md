# gaze-track

Real-time examinee gaze & distraction tracking on a laptop CPU. See [PLAN.md](PLAN.md) for the full roadmap.

**Current stage: Phase 1** (data + evaluation harness), on top of the Phase 0 geometric baseline (MediaPipe landmarks → head pose + iris features →
calibrated gaze ray → screen-plane zones → temporal events). No training yet.

## Setup
```bash
uv sync
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
On macOS, allow camera access for your terminal (System Settings → Privacy & Security → Camera).

## Use
```bash
uv run python -m gaze_track config                 # optional: writes config.json (screen size in cm, thresholds)
uv run python -m gaze_track calibrate --profile me # fullscreen 9-point calibration (~25 s)
uv run python -m gaze_track run --profile me --record runs/s1.csv --events runs/s1.jsonl
uv run python -m gaze_track bench --frames 300     # per-stage latency
uv run pytest -q
```

## Data collection & evaluation (Phase 1)
```bash
# Guided labelled recording (~4 min, fullscreen, audio cues). One per condition.
uv run python -m gaze_track collect --subject s01 --glasses yes --lighting normal
# Honest work session for the false-alert rate (calibrates first, then records in the background).
uv run python -m gaze_track collect-honest --subject s01 --glasses yes --minutes 10
uv run python -m gaze_track eval                      # all sessions in data/own
uv run python -m gaze_track export --preview          # normalized patches → data/normalized
```

## Live exam with evidence + review (Phase 6)
```bash
uv run python -m gaze_track run --profile me --evidence        # snapshots only around flags
uv run python -m gaze_track report runs/evidence/<bundle>      # HTML review report (offline)
uv run python -m gaze_track report data/own/<session>          # same report for a recorded session
uv run python -m gaze_track robustness data/own/<guided session> --gate   # image-degradation stress test
```

## Learned + fused estimator (Phase 2)
```bash
uv run python -m gaze_track prepare-mpii                                  # MPIIFaceGaze → normalized patches
uv run python -m gaze_track train --backbone mobilenetv4_conv_small --epochs 15   # ~20 min on M2 GPU
uv run python -m gaze_track export-onnx runs/train/<run>/best.pt          # ONNX (1.15 ms CPU)
uv run python -m gaze_track calibrate --profile me --model latest         # fits the fused estimator too
uv run python -m gaze_track run --profile me                              # uses it automatically
uv run python -m gaze_track eval --model runs/train/<run>/best.pt --fuse  # compare against the baseline
```
Defaults assume a MacBook Air M2 13.6" (28.96 × 18.82 cm display, camera 0.6 cm above it).
Edit `config.json` for other screens.

## Zones
`ON_SCREEN`, `EDGE` (±2.5 cm of border), `NEAR_OFF` (≤20 cm outside), `FAR_OFF`, `DOWN_KEYBOARD`,
`EYES_CLOSED`, `NO_FACE`, `MULTI_FACE`, `UNKNOWN`.

## Events
`brief_glance` (info), `sustained_off_screen` (warning, ≥3 s), `repeated_directional_glance`
(alert, ≥3 glances to the same spot within 60 s), `face_absent`, `multiple_faces`, `eyes_closed_long`.
Events are signals for human review, not verdicts.
