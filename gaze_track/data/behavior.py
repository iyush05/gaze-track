"""Scripted behaviour protocol for the temporal (event) layer: realistic honest and cheating segments.

Unlike the gaze protocol (exact targets, cued 2 s looks), each segment here is 30–45 s of natural
behaviour with a *segment-level* label. It measures what the event layer is for:
  * cheating segments → is a warning/alert raised, and how fast?
  * honest segments   → false flags (incl. looking up to think, typing, reading)
and later provides training/validation windows for a learned temporal model.
"""

from __future__ import annotations

from dataclasses import dataclass

READING = (
    "Gaze estimation predicts where a person is looking from an image of their face. Appearance-based "
    "methods learn this mapping directly from pixels with a neural network, while model-based methods "
    "fit a 3D eye model to detected landmarks. In practice, accuracy depends strongly on lighting, head "
    "pose, glasses and on how similar the user is to the people in the training data, which is why most "
    "systems ask for a short per-user calibration before use."
)

QUESTIONS = [
    "Q1. What is 47 x 23?",
    "Q2. Name the capital of Australia.",
    "Q3. What year did the first human land on the Moon?",
    "Q4. What is the chemical symbol for potassium?",
    "Q5. How many bones are in the adult human body?",
]


@dataclass
class BehaviorSegment:
    task: str  # honest_* or cheat_*
    seconds: float
    setup: str  # shown before the segment starts (user prepares, presses SPACE)
    instruction: str  # shown at the top while recording
    content: str = ""  # reading text / questions shown while recording
    typing: bool = False  # show a text box the user types into

    @property
    def label(self) -> str:
        return "cheat" if self.task.startswith("cheat") else "honest"


def build_behavior_protocol(side: str = "left", short: bool = False) -> list[BehaviorSegment]:
    """side: where the user puts their notes/phone for the side-glance cheating segment."""
    k = 0.5 if short else 1.0
    q = "\n".join(QUESTIONS)
    return [
        BehaviorSegment("honest_reading", 40 * k, "Next: read a short text on screen, as in an exam.",
                        "Read the text carefully.", READING),
        BehaviorSegment("honest_typing", 40 * k, "Next: type a summary of the text (keyboard glances are normal).",
                        "Type a 2-3 sentence summary until the time runs out.", READING,
                        typing=True),
        BehaviorSegment("honest_thinking", 35 * k, "Next: answer questions from memory. Think naturally: looking up "
                        "or away briefly while thinking is fine, as in a real exam.",
                        "Answer from memory (type answers). Think the way you normally do.", q, typing=True),
        BehaviorSegment(f"cheat_notes_{side}", 40 * k, f"Next (CHEATING): put a paper/phone with notes on the {side.upper()} "
                        "side of the laptop. Answer by copying from it: glance at it as often as you need.",
                        f"Copy the answers from your notes on the {side}.", q, typing=True),
        BehaviorSegment("cheat_phone_lap", 35 * k, "Next (CHEATING): hold your phone in your lap, below the desk. "
                        "Look up answers on it and type them.", "Check your phone in your lap for answers.", q, typing=True),
        BehaviorSegment("cheat_ask_side", 30 * k, "Next (CHEATING): pretend someone beside you is helping. Turn to "
                        "them a few times as if asking (or ask a real person to sit beside you).",
                        "Ask the person beside you for help a few times.", q, typing=True),
    ]
