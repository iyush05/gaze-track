"""Fullscreen stimulus window shared by calibration and data collection.

The canvas always fills the window (FREERATIO), and the window size is re-read every frame, so
normalized dot positions map to the right physical (cm) location even if fullscreen settles late.
"""

from __future__ import annotations

import subprocess
import sys
import time

import cv2
import numpy as np

from .calibration import target_cm, window_top_offset_cm
from .config import Config

ARROWS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


def beep(kind: str = "go") -> None:
    """Non-blocking audio cue (macOS system sounds; silently skipped elsewhere)."""
    if sys.platform != "darwin":
        return
    sound = {"go": "Tink", "back": "Pop", "done": "Glass"}.get(kind, "Tink")
    try:
        subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class FullscreenCanvas:
    def __init__(self, cfg: Config, title: str):
        self.cfg, self.win = cfg, title
        self.ww = self.wh = 0
        self.top_off = 0.0
        self._fallback = (1440, int(1440 * cfg.screen.height_cm / cfg.screen.width_cm))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._wait_stable()

    # ------------------------------------------------------------------ geometry
    def refresh(self) -> bool:
        """Re-read the window size; returns True if it changed."""
        try:
            _, _, w, h = cv2.getWindowImageRect(self.win)
        except cv2.error:
            w, h = 0, 0
        if w < 200 or h < 200:
            w, h = self.ww or self._fallback[0], self.wh or self._fallback[1]
        if (w, h) == (self.ww, self.wh):
            return False
        self.ww, self.wh = w, h
        self.top_off = window_top_offset_cm(self.cfg.screen, w, h)
        return True

    def _wait_stable(self, max_s: float = 4.0, stable_s: float = 0.5) -> None:
        """macOS animates into fullscreen: wait until the size stops changing."""
        stable_since, deadline = time.monotonic(), time.monotonic() + max_s
        while time.monotonic() < deadline:
            cv2.imshow(self.win, np.full((self.wh or 10, self.ww or 10, 3), 25, np.uint8))
            cv2.waitKey(30)
            if self.refresh():
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since > stable_s:
                break

    def frac_to_cm(self, fx: float, fy: float) -> tuple[float, float]:
        return target_cm(self.cfg.screen, fx, fy, self.top_off)

    def cm_to_frac(self, u: float, v: float) -> tuple[float, float]:
        visible_h = self.cfg.screen.height_cm - self.top_off
        return u / self.cfg.screen.width_cm, (v - self.top_off) / visible_h

    # ------------------------------------------------------------------ drawing
    def show(
        self,
        msg: str = "",
        *,
        dot_frac: tuple[float, float] | None = None,
        progress: float = 0.0,
        preview: np.ndarray | None = None,
        big: str | None = None,
        big_color=(0, 200, 255),
        arrow: str | None = None,
        sub: str | None = None,
    ) -> int:
        """Render one frame; returns the pressed key (or 255)."""
        if self.refresh():
            print(f"Window resized → {self.ww}x{self.wh}; positions updated")
        ww, wh = self.ww, self.wh
        img = np.full((wh, ww, 3), 25, np.uint8)
        if preview is not None:
            ph = min(wh // 2, 360)
            pw = int(preview.shape[1] * ph / preview.shape[0])
            small = cv2.resize(cv2.flip(preview, 1), (pw, ph))
            y0, x0 = (wh - ph) // 2, (ww - pw) // 2
            img[y0 : y0 + ph, x0 : x0 + pw] = small
        if msg:
            cv2.putText(img, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        if big:
            scale = 1.6
            (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
            org = ((ww - tw) // 2, wh // 2 - 60)
            cv2.putText(img, big, org, cv2.FONT_HERSHEY_SIMPLEX, scale, big_color, 3, cv2.LINE_AA)
        if sub:
            (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(img, sub, ((ww - tw) // 2, wh // 2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (200, 200, 200), 2, cv2.LINE_AA)
        if arrow in ARROWS:
            dx, dy = ARROWS[arrow]
            c = np.array([ww // 2, wh // 2 + 30])
            L = min(ww, wh) // 5
            cv2.arrowedLine(img, tuple(int(x) for x in c - np.array([dx, dy]) * L // 2),
                            tuple(int(x) for x in c + np.array([dx, dy]) * L // 2),
                            big_color, 8, tipLength=0.35)
        if dot_frac is not None:
            dot = (int(dot_frac[0] * (ww - 1)), int(dot_frac[1] * (wh - 1)))
            r = int(22 - 16 * progress)
            cv2.circle(img, dot, r, (0, 200, 255), -1)
            cv2.circle(img, dot, 3, (0, 0, 0), -1)
        cv2.imshow(self.win, img)
        return cv2.waitKey(1) & 0xFF

    def show_page(self, instruction: str, content: str = "", typed: str | None = None, footer: str = "",
                  preview: np.ndarray | None = None, color=(0, 200, 255)) -> int:
        """Text page for behaviour segments: instruction, wrapped content, optional typing box."""
        if self.refresh():
            print(f"Window resized → {self.ww}x{self.wh}; positions updated")
        ww, wh = self.ww, self.wh
        img = np.full((wh, ww, 3), 25, np.uint8)
        font, x0 = cv2.FONT_HERSHEY_SIMPLEX, int(ww * 0.08)
        max_w = int(ww * 0.84)

        def wrap(text: str, scale: float) -> list[str]:
            lines = []
            for para in text.split("\n"):
                cur = ""
                for word in para.split():
                    cand = f"{cur} {word}".strip()
                    if cv2.getTextSize(cand, font, scale, 1)[0][0] > max_w and cur:
                        lines.append(cur)
                        cur = word
                    else:
                        cur = cand
                lines.append(cur)
            return lines

        y = int(wh * 0.10)
        for line in wrap(instruction, 0.9):
            cv2.putText(img, line, (x0, y), font, 0.9, color, 2, cv2.LINE_AA)
            y += 38
        y += 20
        for line in wrap(content, 0.75):
            cv2.putText(img, line, (x0, y), font, 0.75, (225, 225, 225), 1, cv2.LINE_AA)
            y += 32
        if typed is not None:
            y += 20
            box_h = max(120, wh - y - 90)
            cv2.rectangle(img, (x0 - 10, y - 30), (x0 + max_w + 10, y - 30 + box_h), (90, 90, 90), 1)
            for line in wrap(typed + "_", 0.75)[-(box_h // 32):]:
                cv2.putText(img, line, (x0, y), font, 0.75, (180, 255, 180), 1, cv2.LINE_AA)
                y += 32
        if preview is not None:
            ph = min(wh // 3, 240)
            pw = int(preview.shape[1] * ph / preview.shape[0])
            small = cv2.resize(cv2.flip(preview, 1), (pw, ph))
            img[wh - ph - 60 : wh - 60, ww - pw - 30 : ww - 30] = small
        if footer:
            cv2.putText(img, footer, (x0, wh - 30), font, 0.7, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow(self.win, img)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self.win)
        cv2.waitKey(1)
