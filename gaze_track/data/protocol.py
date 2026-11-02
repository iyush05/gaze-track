"""Data-collection protocol: a sequence of trials with known ground truth.

Segments
--------
calib   9-point grid (same as `calibrate`): used by `eval` to fit a per-session calibration, so
        every session is self-contained and test frames never leak into calibration.
screen  Random on-screen dots, half of them close to the border → exact gaze target (u, v) in cm,
        ground-truth zone ON_SCREEN or EDGE.
off     Cued off-screen looks (audio beeps, user can't read the screen while looking away)
        → ground-truth *zone* only (NEAR_OFF / FAR_OFF / DOWN_KEYBOARD / EYES_CLOSED / NO_FACE).
        NEAR_OFF trials also carry an *approximate* target (≈10 cm beside the screen).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..calibration import calibration_targets
from ..config import ScreenGeometry, ZoneThresholds


@dataclass
class Trial:
    trial_id: int
    segment: str  # calib | screen | off
    task: str  # short machine name, e.g. dot, near_left, far_right, keyboard
    zone: str  # ground-truth zone label
    instruction: str = ""
    arrow: str | None = None
    fx: float | None = None  # dot position, normalized window coords (screen segments)
    fy: float | None = None
    u_cm: float | None = None  # gaze target on the screen plane, cm (approx if `approx`)
    v_cm: float | None = None
    approx: bool = False
    settle_s: float = 1.0  # not recorded (saccade / reaction time)
    record_s: float = 1.2

    def to_dict(self) -> dict:
        return asdict(self)


OFF_TASKS = [
    # task, zone, instruction, arrow, approx (u, v) offset description
    ("near_left", "NEAR_OFF", "Look ~10 cm (a hand's width) LEFT of the screen", "left"),
    ("near_right", "NEAR_OFF", "Look ~10 cm (a hand's width) RIGHT of the screen", "right"),
    ("near_top", "NEAR_OFF", "Look ~10 cm ABOVE the top of the screen", "up"),
    ("far_left", "FAR_OFF", "Turn your head and look at the wall to your LEFT", "left"),
    ("far_right", "FAR_OFF", "Turn your head and look at the wall to your RIGHT", "right"),
    ("far_up", "FAR_OFF", "Look up at the ceiling", "up"),
    ("keyboard", "DOWN_KEYBOARD", "Look down at the keyboard", "down"),
    ("eyes_closed", "EYES_CLOSED", "Close your eyes", None),
]
NO_FACE_TASK = ("no_face", "NO_FACE", "Lean fully out of the camera view", None)


def _near_target(task: str, screen: ScreenGeometry, d: float = 10.0) -> tuple[float, float]:
    W, H = screen.width_cm, screen.height_cm
    return {"near_left": (-d, H / 2), "near_right": (W + d, H / 2), "near_top": (W / 2, -d)}[task]


def screen_dot_positions(n: int, screen: ScreenGeometry, th: ZoneThresholds, rng) -> list[tuple[float, float]]:
    """n on-screen targets in cm: half uniform over the screen, half inside the edge band
    (the hard ON_SCREEN/EDGE boundary cases)."""
    W, H = screen.width_cm, screen.height_cm
    m = 0.8  # keep the dot fully visible
    pts = []
    for i in range(n):
        if i % 2 == 0:
            pts.append((rng.uniform(m, W - m), rng.uniform(m, H - m)))
        else:
            side = rng.integers(4)
            depth = rng.uniform(m, th.edge_cm * 1.6)  # straddles the EDGE/ON boundary
            along_u, along_v = rng.uniform(m, W - m), rng.uniform(m, H - m)
            pts.append([(depth, along_v), (W - depth, along_v), (along_u, depth), (along_u, H - depth)][side])
    return pts


def gt_screen_zone(u: float, v: float, screen: ScreenGeometry, th: ZoneThresholds) -> str:
    inside = min(u, screen.width_cm - u, v, screen.height_cm - v)
    return "ON_SCREEN" if inside >= th.edge_cm else "EDGE"


def build_protocol(
    screen: ScreenGeometry,
    th: ZoneThresholds,
    n_screen: int = 24,
    off_repeats: int = 2,
    include_no_face: bool = True,
    seed: int = 0,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []

    for fx, fy in calibration_targets(9):
        trials.append(Trial(len(trials), "calib", "dot", "", fx=fx, fy=fy, settle_s=1.0, record_s=1.2))

    # Screen dots: positions are stored in cm; fx/fy are filled in by the collector once the
    # real window geometry is known (see `Trial.fx` handling in collect.py).
    for u, v in screen_dot_positions(n_screen, screen, th, rng):
        trials.append(Trial(len(trials), "screen", "dot", gt_screen_zone(u, v, screen, th),
                            u_cm=float(u), v_cm=float(v), settle_s=0.8, record_s=1.0))

    off = [t for t in OFF_TASKS for _ in range(off_repeats)]
    order = rng.permutation(len(off))
    off = [off[i] for i in order]
    if include_no_face:
        off.append(NO_FACE_TASK)
    for task, zone, instr, arrow in off:
        # Reaction time after the beep (read → turn) exceeded 1 s in s01's first session (late head turns
        # inside the recorded window), so wait longer before recording; leaning out takes longest.
        settle = 2.5 if task == "no_face" else 1.5
        tr = Trial(len(trials), "off", task, zone, instruction=instr, arrow=arrow, settle_s=settle, record_s=2.0)
        if task.startswith("near_"):
            tr.u_cm, tr.v_cm = _near_target(task, screen)
            tr.approx = True
        trials.append(tr)
    return trials
