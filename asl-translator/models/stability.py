"""
stability.py

Turns a noisy stream of per-moment predictions into a calm stream of
accepted signs.

Why this file exists:
    The app runs the model several times per second on a sliding window of
    recent frames. Raw predictions flicker: while you move between signs,
    or when the lighting changes, the top guess can jump around. If every
    raw prediction were added to the text you would get "hello hello bye
    hello hello hello". This module applies several simple, configurable
    rules before a sign is accepted.

The mechanisms (all settings live in config.yaml -> recognition):
    1. Rolling window + temporal smoothing (smoothing_window):
         keep the last N probability vectors and AVERAGE them, so one odd
         prediction cannot change the result on its own.
    2. Confidence threshold (confidence_threshold):
         if the best averaged probability is below this, report
         "Uncertain sign" instead of guessing.
    3. Stability requirement (stability_required):
         the same sign must be the confident winner for this many
         inference steps IN A ROW before it is accepted.
    4. Repeat prevention / "release" (release_steps):
         after a sign is accepted it is "locked". Holding the sign longer
         will not add it again. The lock is released only after
         release_steps consecutive steps where that sign is NOT the
         confident winner (no hands, uncertain, or a different sign). This
         is how two genuinely consecutive identical signs can be separated:
         a short pause, a hand drop, or a transition releases the lock.
         We do NOT assume a particular "neutral" hand position.
    5. Cooldown (cooldown_seconds):
         a minimum time between ANY two accepted signs.

None of this makes the model more accurate - it only reduces flicker and
duplicates. A wrong sign that is held steadily will still be accepted.

Used by: app.py.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

from translation.vocabulary import Vocabulary

# Status values reported to the user interface.
STATUS_NO_HANDS = "no_hands"
STATUS_UNCERTAIN = "uncertain"
STATUS_STABILIZING = "stabilizing"
STATUS_ACCEPTED = "accepted"
STATUS_HOLDING = "holding"
STATUS_COOLDOWN = "cooldown"


@dataclass
class StabilizerSettings:
    """Configurable rules; see the module docstring for what each one does."""

    confidence_threshold: float = 0.70
    smoothing_window: int = 5
    stability_required: int = 3
    release_steps: int = 4
    cooldown_seconds: float = 1.0

    def validate(self) -> None:
        """Raise ValueError for settings that make no sense."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if self.smoothing_window < 1 or self.stability_required < 1 or self.release_steps < 1:
            raise ValueError("smoothing_window, stability_required and release_steps must be at least 1.")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative.")


@dataclass
class StabilizerOutput:
    """
    What the stabilizer decided for one inference step.

    Attributes:
        status: One of the STATUS_* constants.
        candidate_label: The current smoothed best guess (may be None).
        smoothed_confidence: Averaged probability of candidate_label.
        accepted_label: Set ONLY on the step a sign is accepted - this is the
            signal to add a word to the sentence.
        stable_steps: How many consecutive steps the candidate has led.
    """

    status: str
    candidate_label: Optional[str] = None
    smoothed_confidence: float = 0.0
    accepted_label: Optional[str] = None
    stable_steps: int = 0


class PredictionStabilizer:
    """
    Feed it one probability vector (or "no hands") per inference step via
    update(); it tells you when a sign should be accepted.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        settings: Optional[StabilizerSettings] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            vocabulary: Maps probability positions to labels.
            settings: The rules to apply (defaults if omitted).
            clock: Function returning the current time in seconds. Tests pass
                a fake clock so they do not have to actually wait.
        """
        self.vocabulary = vocabulary
        self.settings = settings or StabilizerSettings()
        self.settings.validate()
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Forget all history (e.g. when the camera is restarted)."""
        self._recent_probabilities: Deque[np.ndarray] = deque(maxlen=self.settings.smoothing_window)
        self._current_candidate: Optional[str] = None
        self._consecutive_candidate_steps = 0
        self._locked_label: Optional[str] = None
        self._steps_since_locked_label_seen = 0
        self._last_accept_time: Optional[float] = None

    def update_settings(self, settings: StabilizerSettings) -> None:
        """Apply new settings (from the app's sliders) without losing the lock state."""
        settings.validate()
        if settings.smoothing_window != self.settings.smoothing_window:
            self._recent_probabilities = deque(self._recent_probabilities, maxlen=settings.smoothing_window)
        self.settings = settings

    def _register_step_without_locked_label(self) -> None:
        """Count a step where the locked sign is absent; release the lock once enough pass."""
        if self._locked_label is None:
            return
        self._steps_since_locked_label_seen += 1
        if self._steps_since_locked_label_seen >= self.settings.release_steps:
            self._locked_label = None

    def update_no_hands(self) -> StabilizerOutput:
        """
        Record an inference step where no usable hand data was available.

        Clearing the smoothing history means a new sign starts fresh after
        the hands come back, instead of being averaged with an old one.
        """
        self._recent_probabilities.clear()
        self._current_candidate = None
        self._consecutive_candidate_steps = 0
        self._register_step_without_locked_label()
        return StabilizerOutput(status=STATUS_NO_HANDS)

    def update(self, probabilities: np.ndarray) -> StabilizerOutput:
        """
        Process one model output.

        Args:
            probabilities: Array of shape (num_classes,) from the predictor.

        Returns:
            A StabilizerOutput. Check output.accepted_label: when it is not
            None, a new sign has been accepted.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (self.vocabulary.num_classes,):
            raise ValueError(
                f"Expected {self.vocabulary.num_classes} probabilities, got shape {probabilities.shape}."
            )

        # 1. Temporal smoothing: average the recent probability vectors.
        self._recent_probabilities.append(probabilities)
        smoothed_probabilities = np.mean(self._recent_probabilities, axis=0)
        best_index = int(np.argmax(smoothed_probabilities))
        best_label = self.vocabulary.index_to_label(best_index)
        smoothed_confidence = float(smoothed_probabilities[best_index])

        # 2. Confidence threshold.
        if smoothed_confidence < self.settings.confidence_threshold:
            self._current_candidate = None
            self._consecutive_candidate_steps = 0
            self._register_step_without_locked_label()
            return StabilizerOutput(
                status=STATUS_UNCERTAIN, candidate_label=best_label, smoothed_confidence=smoothed_confidence
            )

        # 3. Stability: count how many steps in a row this label has led.
        if best_label == self._current_candidate:
            self._consecutive_candidate_steps += 1
        else:
            self._current_candidate = best_label
            self._consecutive_candidate_steps = 1

        # 4. Repeat prevention: track whether the locked sign is still shown.
        if best_label == self._locked_label:
            self._steps_since_locked_label_seen = 0
            return StabilizerOutput(
                status=STATUS_HOLDING,
                candidate_label=best_label,
                smoothed_confidence=smoothed_confidence,
                stable_steps=self._consecutive_candidate_steps,
            )
        self._register_step_without_locked_label()

        if self._consecutive_candidate_steps < self.settings.stability_required:
            return StabilizerOutput(
                status=STATUS_STABILIZING,
                candidate_label=best_label,
                smoothed_confidence=smoothed_confidence,
                stable_steps=self._consecutive_candidate_steps,
            )

        # 5. Cooldown between accepted signs.
        now = self._clock()
        if self._last_accept_time is not None and now - self._last_accept_time < self.settings.cooldown_seconds:
            return StabilizerOutput(
                status=STATUS_COOLDOWN,
                candidate_label=best_label,
                smoothed_confidence=smoothed_confidence,
                stable_steps=self._consecutive_candidate_steps,
            )

        # All rules passed: accept the sign and lock it.
        self._locked_label = best_label
        self._steps_since_locked_label_seen = 0
        self._last_accept_time = now
        return StabilizerOutput(
            status=STATUS_ACCEPTED,
            candidate_label=best_label,
            smoothed_confidence=smoothed_confidence,
            accepted_label=best_label,
            stable_steps=self._consecutive_candidate_steps,
        )
