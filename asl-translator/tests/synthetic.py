"""
synthetic.py

Helpers that build FAKE hand landmark data for automated tests only.

These synthetic hands are simple geometric shapes, not real ASL. They let
the tests check that the code runs and behaves correctly (shapes, edge
cases, saving/loading) without needing a webcam or a downloaded dataset.
They are never used to train the real model.
"""

import numpy as np

from vision.preprocessing import NUM_HAND_LANDMARKS, empty_frame_landmarks

# A rough open-hand layout: offsets of the 21 landmarks from the wrist,
# in units of "hand size" (wrist -> middle knuckle distance = 1).
_BASE_HAND_OFFSETS = np.array(
    [[0.0, 0.0, 0.0]]
    + [[-0.3 - 0.15 * i, -0.3 - 0.25 * i, 0.0] for i in range(4)]  # thumb
    + [[-0.25 + 0.17 * finger, -0.9 - 0.3 * joint, 0.0] for finger in range(4) for joint in range(4)],
    dtype=np.float32,
)
assert _BASE_HAND_OFFSETS.shape == (NUM_HAND_LANDMARKS, 3)
# Make landmark 9 (middle knuckle) exactly 1 hand-size above the wrist.
_BASE_HAND_OFFSETS[9] = [0.0, -1.0, 0.0]


def make_hand(wrist_x: float, wrist_y: float, hand_size: float = 0.1, curl: float = 0.0) -> np.ndarray:
    """
    Build one synthetic hand of shape (21, 3).

    Args:
        wrist_x, wrist_y: Wrist position in image coordinates (0-1).
        hand_size: Distance from wrist to middle knuckle (0-1 image units).
        curl: 0 = open hand, 1 = fingers pulled toward the palm.
    """
    offsets = _BASE_HAND_OFFSETS.copy()
    offsets[5:, 1] *= 1.0 - 0.6 * curl
    offsets[9] = [0.0, -1.0, 0.0]
    hand = offsets * hand_size
    hand[:, 0] += wrist_x
    hand[:, 1] += wrist_y
    return hand.astype(np.float32)


def make_sequence(
    pattern: str,
    num_frames: int = 30,
    slot: int = 1,
    start=(0.5, 0.5),
    hand_size: float = 0.1,
    noise: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Build a synthetic landmark sequence of shape (num_frames, 2, 21, 3).

    Args:
        pattern: "horizontal" (hand moves sideways), "vertical" (moves up and
            down) or "curl" (hand stays still and closes).
        slot: Which hand slot holds the hand (0 = left, 1 = right).
        start: Starting wrist position.
        hand_size: Apparent hand size.
        noise: Standard deviation of random jitter added to all points.
    """
    rng = rng or np.random.default_rng(0)
    sequence = np.repeat(empty_frame_landmarks()[np.newaxis], num_frames, axis=0)
    for frame in range(num_frames):
        progress = frame / max(1, num_frames - 1)
        x, y, curl = start[0], start[1], 0.0
        if pattern == "horizontal":
            x += 0.2 * np.sin(2 * np.pi * progress)
        elif pattern == "vertical":
            y += 0.2 * np.sin(2 * np.pi * progress)
        elif pattern == "curl":
            curl = progress
        else:
            raise ValueError(f"unknown pattern {pattern}")
        sequence[frame, slot] = make_hand(x, y, hand_size, curl)
    if noise:
        sequence = sequence + rng.normal(0, noise, size=sequence.shape).astype(np.float32)
    return sequence.astype(np.float32)
