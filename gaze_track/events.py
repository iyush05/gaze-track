"""Rules-based temporal attentiveness events (Phase 0; replaced by a learned model in Phase 5).

Events are *signals for human review*, never automatic verdicts.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

from .config import EventThresholds
from .zones import OFF_ZONES, Zone


@dataclass
class Event:
    t: float
    kind: str
    severity: str  # info | warning | alert
    duration_s: float = 0.0
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class _Episode:
    start: float
    last_off: float = 0.0  # last time the zone was off-screen (for gap merging)
    sum_yaw: float = 0.0
    sum_pitch: float = 0.0
    n: int = 0
    flagged: bool = False

    def add(self, yaw: float | None, pitch: float | None) -> None:
        if yaw is not None and pitch is not None:
            self.sum_yaw += yaw
            self.sum_pitch += pitch
            self.n += 1

    def direction(self) -> tuple[float | None, float | None]:
        if self.n == 0:
            return None, None
        return self.sum_yaw / self.n, self.sum_pitch / self.n


class EventEngine:
    def __init__(self, th: EventThresholds):
        self.th = th
        self._off: _Episode | None = None
        self._absent: _Episode | None = None
        self._multi: _Episode | None = None
        self._closed: _Episode | None = None
        self._unknown: _Episode | None = None
        self._glances: deque[tuple[float, float, float]] = deque()  # (t_end, yaw, pitch)
        self._last_repeat_alert = -1e9
        self._turn: _Episode | None = None
        self._turns: deque[float] = deque()
        self._last_turn_alert = -1e9
        self._head_yaw: float | None = None

    def update(self, t: float, zone: Zone, yaw: float | None, pitch: float | None,
               head_yaw: float | None = None) -> list[Event]:
        """head_yaw: head yaw relative to the user's neutral pose (degrees), if known."""
        out: list[Event] = []
        self._head_yaw = head_yaw
        out += self._track_head_turn(t, head_yaw)
        out += self._track_off(t, zone, yaw, pitch)
        self._absent, ev = self._track_state(
            t, zone == Zone.NO_FACE, self._absent, self.th.face_absent_s, "face_absent", "alert"
        )
        out += ev
        self._multi, ev = self._track_state(
            t, zone == Zone.MULTI_FACE, self._multi, self.th.multi_face_s, "multiple_faces", "alert"
        )
        out += ev
        self._closed, ev = self._track_state(
            t, zone == Zone.EYES_CLOSED, self._closed, self.th.eyes_closed_s, "eyes_closed_long", "warning"
        )
        out += ev
        self._unknown, ev = self._track_state(
            t, zone == Zone.UNKNOWN, self._unknown, self.th.unreliable_s, "tracking_unreliable", "warning"
        )
        out += ev
        return out

    # ------------------------------------------------------------------ off-screen glances
    def _track_off(self, t: float, zone: Zone, yaw: float | None, pitch: float | None) -> list[Event]:
        out: list[Event] = []
        if zone in OFF_ZONES:
            if self._off is None:
                self._off = _Episode(start=t)
            self._off.last_off = t
            self._off.add(yaw, pitch)
            dur = t - self._off.start
            _, p = self._off.direction()
            head_forward = self._head_yaw is None or abs(self._head_yaw) < self.th.head_turn_deg
            looking_up = p is not None and p < self.th.up_pitch_deg and head_forward
            limit = self.th.sustained_up_s if looking_up else self.th.sustained_off_s
            if not self._off.flagged and dur >= limit:
                self._off.flagged = True
                y, p = self._off.direction()
                out.append(Event(t, "sustained_off_screen", "warning", dur, y, p))
            return out

        if self._off is None:
            return out
        # Short returns on-screen (e.g. EDGE/NEAR_OFF flicker) don't end the glance.
        if t - self._off.last_off < self.th.merge_gap_s:
            return out
        # Episode ended: its duration runs to the last off-screen frame.
        ep, self._off = self._off, None
        dur = ep.last_off - ep.start
        y, p = ep.direction()
        if dur <= self.th.brief_glance_max_s:
            out.append(Event(t, "brief_glance", "info", dur, y, p))
        if y is None or p is None:
            return out

        self._glances.append((t, y, p))
        while self._glances and t - self._glances[0][0] > self.th.repeated_window_s:
            self._glances.popleft()
        similar = [g for g in self._glances if math.hypot(g[1] - y, g[2] - p) <= self.th.repeated_direction_deg]
        if len(similar) >= self.th.repeated_min_count and t - self._last_repeat_alert > self.th.repeated_window_s:
            self._last_repeat_alert = t
            out.append(
                Event(
                    t,
                    "repeated_directional_glance",
                    "alert",
                    dur,
                    y,
                    p,
                    detail=f"{len(similar)} glances toward the same spot in {self.th.repeated_window_s:.0f}s",
                )
            )
        return out

    # ------------------------------------------------------------------ head turned to the side
    def _track_head_turn(self, t: float, head_yaw: float | None) -> list[Event]:
        out: list[Event] = []
        if head_yaw is not None and abs(head_yaw) > self.th.head_turn_deg:
            if self._turn is None:
                self._turn = _Episode(start=t)
            self._turn.add(head_yaw, 0.0)
            dur = t - self._turn.start
            if not self._turn.flagged and dur >= self.th.head_turn_s:
                self._turn.flagged = True
                y, _ = self._turn.direction()
                out.append(Event(t, "head_turned_away", "warning", dur, y, None))
                self._turns.append(t)
                while self._turns and t - self._turns[0] > self.th.repeated_window_s:
                    self._turns.popleft()
                if len(self._turns) >= self.th.head_turn_repeat and t - self._last_turn_alert > self.th.repeated_window_s:
                    self._last_turn_alert = t
                    out.append(Event(t, "repeated_head_turns", "alert", dur, y, None,
                                     detail=f"{len(self._turns)} head turns in {self.th.repeated_window_s:.0f}s"))
            return out
        self._turn = None
        return out

    # ------------------------------------------------------------------ generic duration state
    @staticmethod
    def _track_state(t, active, ep, min_s, kind, severity):
        if not active:
            return None, []
        if ep is None:
            ep = _Episode(start=t)
        if not ep.flagged and t - ep.start >= min_s:
            ep.flagged = True
            return ep, [Event(t, kind, severity, t - ep.start)]
        return ep, []
