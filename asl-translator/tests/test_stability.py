"""Tests for models/stability.py - turning flickering predictions into accepted signs."""

import numpy as np
import pytest

from models.stability import (
    STATUS_ACCEPTED,
    STATUS_COOLDOWN,
    STATUS_HOLDING,
    STATUS_NO_HANDS,
    STATUS_STABILIZING,
    STATUS_UNCERTAIN,
    PredictionStabilizer,
    StabilizerSettings,
)
from translation.vocabulary import Vocabulary

VOCABULARY = Vocabulary(["hello", "yes", "no"])


class FakeClock:
    """A controllable clock so tests do not need to sleep."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def confident(label, confidence=0.9):
    probabilities = np.full(3, (1 - confidence) / 2)
    probabilities[VOCABULARY.label_to_index(label)] = confidence
    return probabilities


def make_stabilizer(clock=None, **overrides):
    settings = dict(confidence_threshold=0.7, smoothing_window=1, stability_required=3, release_steps=2, cooldown_seconds=0.0)
    settings.update(overrides)
    return PredictionStabilizer(VOCABULARY, StabilizerSettings(**settings), clock=clock or FakeClock())


def test_sign_is_accepted_only_after_enough_stable_steps():
    stabilizer = make_stabilizer()
    statuses = [stabilizer.update(confident("hello")).status for _ in range(3)]
    assert statuses == [STATUS_STABILIZING, STATUS_STABILIZING, STATUS_ACCEPTED]


def test_held_sign_is_not_added_repeatedly():
    stabilizer = make_stabilizer()
    accepted = [stabilizer.update(confident("hello")).accepted_label for _ in range(20)]
    assert accepted.count("hello") == 1
    assert stabilizer.update(confident("hello")).status == STATUS_HOLDING


def test_same_sign_can_repeat_after_a_release():
    stabilizer = make_stabilizer()
    for _ in range(3):
        stabilizer.update(confident("hello"))
    stabilizer.update_no_hands()
    stabilizer.update_no_hands()  # release_steps = 2 -> lock released
    outputs = [stabilizer.update(confident("hello")) for _ in range(3)]
    assert outputs[-1].accepted_label == "hello"


def test_brief_dropout_does_not_release_the_lock():
    stabilizer = make_stabilizer(release_steps=3)
    for _ in range(3):
        stabilizer.update(confident("hello"))
    stabilizer.update_no_hands()  # only one step - not enough to release
    accepted = [stabilizer.update(confident("hello")).accepted_label for _ in range(5)]
    assert "hello" not in accepted


def test_low_confidence_is_reported_as_uncertain():
    stabilizer = make_stabilizer()
    output = stabilizer.update(confident("yes", confidence=0.5))
    assert output.status == STATUS_UNCERTAIN
    assert output.accepted_label is None


def test_smoothing_stops_a_single_odd_prediction_from_winning():
    stabilizer = make_stabilizer(smoothing_window=5, stability_required=1)
    for _ in range(4):
        stabilizer.update(confident("hello", 0.6))  # below threshold, builds history
    output = stabilizer.update(confident("no", 0.95))
    # Averaged over 5 steps, "hello" still leads and nothing is accepted.
    assert output.candidate_label == "hello"
    assert output.accepted_label is None


def test_cooldown_delays_the_next_sign():
    clock = FakeClock()
    stabilizer = make_stabilizer(clock=clock, cooldown_seconds=1.0, stability_required=1)
    assert stabilizer.update(confident("hello")).status == STATUS_ACCEPTED
    clock.now = 0.5
    assert stabilizer.update(confident("yes")).status == STATUS_COOLDOWN
    clock.now = 1.5
    assert stabilizer.update(confident("yes")).accepted_label == "yes"


def test_no_hands_status_and_bad_input():
    stabilizer = make_stabilizer()
    assert stabilizer.update_no_hands().status == STATUS_NO_HANDS
    with pytest.raises(ValueError):
        stabilizer.update(np.ones(5))


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        StabilizerSettings(confidence_threshold=1.5).validate()
    with pytest.raises(ValueError):
        StabilizerSettings(stability_required=0).validate()
