"""Tests for vision/preprocessing.py - turning landmarks into model features."""

import numpy as np
import pytest

from tests.synthetic import make_hand, make_sequence
from vision.preprocessing import (
    FEATURES_PER_FRAME,
    FEATURES_PER_HAND,
    InsufficientHandDataError,
    LEFT_HAND_SLOT,
    RIGHT_HAND_SLOT,
    canonicalize_hands,
    choose_dominant_slot,
    count_frames_with_hands,
    empty_frame_landmarks,
    hand_presence_mask,
    landmarks_to_model_input,
    normalize_hand_shape,
    resample_sequence,
    validate_landmark_sequence,
)


def test_output_shape_is_fixed_regardless_of_clip_length():
    for num_frames in (5, 30, 90):
        features = landmarks_to_model_input(make_sequence("horizontal", num_frames), sequence_length=32)
        assert features.shape == (32, FEATURES_PER_FRAME)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()


def test_hand_shape_ignores_position_and_size():
    small_hand_left = make_hand(0.2, 0.3, hand_size=0.05, curl=0.4)
    big_hand_right = make_hand(0.7, 0.6, hand_size=0.2, curl=0.4)
    np.testing.assert_allclose(
        normalize_hand_shape(small_hand_left), normalize_hand_shape(big_hand_right), atol=1e-5
    )


def test_whole_sequence_features_ignore_where_the_signer_stands():
    near_left = make_sequence("vertical", start=(0.3, 0.4), hand_size=0.15)
    far_right = make_sequence("vertical", start=(0.7, 0.6), hand_size=0.15)
    np.testing.assert_allclose(
        landmarks_to_model_input(near_left, 32), landmarks_to_model_input(far_right, 32), atol=1e-4
    )


def test_degenerate_hand_gives_zeros_instead_of_infinity():
    collapsed_hand = np.full((21, 3), 0.5, dtype=np.float32)
    assert np.all(normalize_hand_shape(collapsed_hand) == 0)


def test_missing_hands_are_detected_and_frames_dropped():
    sequence = make_sequence("horizontal", 20)
    sequence[5:10] = np.nan  # hands out of view for 5 frames
    assert count_frames_with_hands(sequence) == (15, 20)
    # Still produces valid features from the remaining frames.
    assert np.isfinite(landmarks_to_model_input(sequence, 16)).all()


def test_partially_detected_hand_counts_as_missing():
    sequence = make_sequence("horizontal", 4)
    sequence[0, RIGHT_HAND_SLOT, 3, 0] = np.nan
    presence = hand_presence_mask(sequence)
    assert not presence[0, RIGHT_HAND_SLOT]
    assert presence[1, RIGHT_HAND_SLOT]


def test_no_hands_raises_clear_error():
    empty = np.repeat(empty_frame_landmarks()[None], 10, axis=0)
    with pytest.raises(InsufficientHandDataError):
        landmarks_to_model_input(empty, 32)


def test_too_few_frames_with_hands_raises():
    sequence = make_sequence("horizontal", 10)
    sequence[3:] = np.nan
    with pytest.raises(InsufficientHandDataError):
        landmarks_to_model_input(sequence, 32, min_frames_with_hands=5)


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        validate_landmark_sequence(np.zeros((10, 21, 3)))
    with pytest.raises(ValueError):
        validate_landmark_sequence(np.full((3, 2, 21, 3), np.inf))


def test_dominant_hand_is_the_most_visible_one():
    sequence = make_sequence("horizontal", 10, slot=LEFT_HAND_SLOT)
    sequence[:3, RIGHT_HAND_SLOT] = make_hand(0.8, 0.5)
    assert choose_dominant_slot(sequence) == LEFT_HAND_SLOT


def test_left_handed_signing_is_mirrored_to_match_right_handed():
    right_handed = make_sequence("horizontal", 20, slot=RIGHT_HAND_SLOT, start=(0.6, 0.5))
    # Build the mirror image: same motion with the left hand, x flipped.
    left_handed = np.repeat(empty_frame_landmarks()[None], 20, axis=0)
    left_handed[:, LEFT_HAND_SLOT] = right_handed[:, RIGHT_HAND_SLOT]
    left_handed[..., 0] = 1.0 - left_handed[..., 0]
    np.testing.assert_allclose(
        landmarks_to_model_input(right_handed, 32), landmarks_to_model_input(left_handed, 32), atol=1e-4
    )
    # The dominant hand always ends up in slot 0 after canonicalization.
    canonical = canonicalize_hands(left_handed)
    assert np.isfinite(canonical[:, 0]).all()
    assert np.isnan(canonical[:, 1]).all()


def test_presence_flags_are_set_correctly():
    features = landmarks_to_model_input(make_sequence("curl", 10), 10)
    dominant_presence = features[:, FEATURES_PER_HAND - 1]
    other_presence = features[:, 2 * FEATURES_PER_HAND - 1]
    assert np.all(dominant_presence == 1.0)
    assert np.all(other_presence == 0.0)


def test_resample_interpolates_linearly():
    values = np.array([[0.0], [10.0]], dtype=np.float32)
    np.testing.assert_allclose(resample_sequence(values, 5)[:, 0], [0, 2.5, 5, 7.5, 10])
    np.testing.assert_allclose(resample_sequence(np.array([[3.0]]), 4)[:, 0], [3, 3, 3, 3])
