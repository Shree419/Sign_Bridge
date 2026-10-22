"""
preprocessing.py

Converts raw hand landmarks into the fixed-size numerical features that the
neural network reads.

Why this file exists:
    MediaPipe gives us x, y, z coordinates for 21 points on each hand, for
    every video frame. Those raw numbers depend heavily on things that have
    nothing to do with the sign: where you stand in the picture, how close
    you are to the camera, and how long you take to sign. This file removes
    as much of that "nuisance" variation as it can so the model can focus
    on hand SHAPE and MOVEMENT.

    The exact same functions are used for training data AND for the live
    webcam, which is essential: a model only works well on inputs that are
    processed the same way as the data it learned from.

The data format used everywhere in this project ("landmark sequence"):
    A NumPy array of shape (num_frames, 2, 21, 3)
        num_frames : how many video frames the sample contains
        2          : two hand "slots" - slot 0 holds the hand the detector
                     labelled LEFT, slot 1 the hand it labelled RIGHT
        21         : MediaPipe's 21 hand landmarks (wrist, knuckles, tips...)
        3          : x, y, z.  x and y are fractions of the image width and
                     height (0 to 1). z is depth relative to the wrist
                     (smaller = closer to the camera), on roughly the same
                     scale as x.
    A hand that was not detected in a frame is stored as NaN ("not a number").

Pipeline (see landmarks_to_model_input):
    1. validate      - check the array has the right shape
    2. drop frames   - remove frames where no hand was seen at all
    3. canonicalize  - put the dominant (most visible) hand in slot 0 and
                       mirror left-handed signing so it looks right-handed
    4. features      - per frame: normalized hand shape, hand position
                       relative to the sign's average position, and a
                       "hand present" flag
    5. resample      - stretch/squeeze to a fixed number of time steps

Used by: vision/landmarks.py, models/predictor.py, training/dataset.py,
data_collection/collect_samples.py and app.py.
"""

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------
# Constants describing the landmark layout.
# ---------------------------------------------------------------------
NUM_HAND_SLOTS = 2
LEFT_HAND_SLOT = 0
RIGHT_HAND_SLOT = 1
NUM_HAND_LANDMARKS = 21
NUM_COORDINATES = 3

# MediaPipe landmark numbers we use as reference points.
# (0 = wrist, 9 = the knuckle at the base of the middle finger.)
WRIST_INDEX = 0
MIDDLE_FINGER_KNUCKLE_INDEX = 9

# Per hand, per frame we produce:
#   21 * 3 = 63 numbers describing the hand's shape,
#   2 numbers describing where the wrist is (x, y) relative to the sign's
#     average position (this captures movement through space),
#   1 number that is 1.0 if the hand is visible, 0.0 if not.
HAND_SHAPE_FEATURES = NUM_HAND_LANDMARKS * NUM_COORDINATES
HAND_TRAJECTORY_FEATURES = 2
FEATURES_PER_HAND = HAND_SHAPE_FEATURES + HAND_TRAJECTORY_FEATURES + 1
FEATURES_PER_FRAME = FEATURES_PER_HAND * NUM_HAND_SLOTS  # = 132

# Distances smaller than this are treated as "zero" to avoid dividing by
# (almost) zero, which would create huge or infinite numbers.
TINY_DISTANCE = 1e-6


class InsufficientHandDataError(ValueError):
    """Raised when a sequence does not contain enough frames with a visible hand."""


def empty_frame_landmarks() -> np.ndarray:
    """
    Create the landmark array for one frame in which no hands were seen.

    Returns:
        An array of shape (2, 21, 3) filled with NaN.
    """
    return np.full((NUM_HAND_SLOTS, NUM_HAND_LANDMARKS, NUM_COORDINATES), np.nan, dtype=np.float32)


def validate_landmark_sequence(landmark_sequence) -> np.ndarray:
    """
    Check that a landmark sequence has the expected shape and convert it to float32.

    Args:
        landmark_sequence: Anything NumPy can turn into an array of shape
            (num_frames, 2, 21, 3).

    Returns:
        The same data as a float32 NumPy array.

    Raises:
        ValueError: if the shape is wrong or the array contains infinite values.

    Why this function exists:
        Catching malformed data early gives a clear error message instead of
        a confusing crash deep inside the neural network.
    """
    array = np.asarray(landmark_sequence, dtype=np.float32)
    expected_shape = (NUM_HAND_SLOTS, NUM_HAND_LANDMARKS, NUM_COORDINATES)
    if array.ndim != 4 or array.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected a landmark sequence of shape (num_frames, {NUM_HAND_SLOTS}, "
            f"{NUM_HAND_LANDMARKS}, {NUM_COORDINATES}) but got {array.shape}."
        )
    if np.isinf(array).any():
        raise ValueError("Landmark sequence contains infinite values.")
    return array


def hand_presence_mask(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Work out which hands are visible in which frames.

    Args:
        landmark_sequence: Array of shape (num_frames, 2, 21, 3).

    Returns:
        Boolean array of shape (num_frames, 2). True means every landmark of
        that hand has real (non-NaN) coordinates in that frame. A hand with
        only some landmarks is treated as missing, so later steps never have
        to deal with half-detected hands.
    """
    return np.isfinite(landmark_sequence).all(axis=(2, 3))


def remove_frames_without_hands(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Drop frames where neither hand was detected.

    Why: frames with no hands carry no information about hand shape, and the
    number of empty frames depends on the camera and on when recording
    started, not on the sign. Removing them makes training and live data
    more alike.
    """
    any_hand_visible = hand_presence_mask(landmark_sequence).any(axis=1)
    return landmark_sequence[any_hand_visible]


def choose_dominant_slot(landmark_sequence: np.ndarray) -> int:
    """
    Decide which hand slot holds the "dominant" (main signing) hand.

    Rule: the hand that is visible in more frames wins. If both are visible
    equally often, the hand whose wrist moves more wins (the dominant hand
    usually does most of the moving). If that also ties, choose the right slot.

    Returns:
        LEFT_HAND_SLOT (0) or RIGHT_HAND_SLOT (1).
    """
    presence = hand_presence_mask(landmark_sequence)
    frames_visible = presence.sum(axis=0)
    if frames_visible[LEFT_HAND_SLOT] != frames_visible[RIGHT_HAND_SLOT]:
        return int(np.argmax(frames_visible))

    wrist_movement = []
    for slot in (LEFT_HAND_SLOT, RIGHT_HAND_SLOT):
        visible_wrists = landmark_sequence[presence[:, slot], slot, WRIST_INDEX, :2]
        # Standard deviation = how spread out the wrist positions are.
        wrist_movement.append(float(visible_wrists.std(axis=0).sum()) if len(visible_wrists) > 1 else 0.0)
    if wrist_movement[LEFT_HAND_SLOT] > wrist_movement[RIGHT_HAND_SLOT]:
        return LEFT_HAND_SLOT
    return RIGHT_HAND_SLOT


def canonicalize_hands(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Reorder and mirror the hands into a standard ("canonical") arrangement.

    After this step:
        slot 0 = the dominant hand, slot 1 = the other hand, and if the
        dominant hand was the LEFT one, the x coordinates are mirrored
        (x becomes 1 - x) so the sign looks as if it were made right-handed.

    Why this matters:
        Left-handed signers mirror their signs. Without this step the model
        would have to learn every sign twice. It also means it does not
        matter whether the dataset and the webcam label "left" and "right"
        the same way: each sample is arranged by its own dominant hand.

    Returns:
        A new array of shape (num_frames, 2, 21, 3).
    """
    dominant_slot = choose_dominant_slot(landmark_sequence)
    other_slot = 1 - dominant_slot
    canonical = landmark_sequence[:, [dominant_slot, other_slot]].copy()
    if dominant_slot == LEFT_HAND_SLOT:
        canonical[..., 0] = 1.0 - canonical[..., 0]
    return canonical


def normalize_hand_shape(hand_landmarks: np.ndarray) -> np.ndarray:
    """
    Describe a single hand's shape independently of its position and size.

    Args:
        hand_landmarks: Array of shape (21, 3) for one hand in one frame.

    Returns:
        Array of shape (21, 3) where the wrist is at (0, 0, 0) and distances
        are measured in "hand sizes". If the hand size is (almost) zero, the
        detection is unusable and all zeros are returned.

    The formula:
        normalized_point = (point - wrist) / hand_size
        - point     : the x, y, z coordinates of one landmark
        - wrist     : the coordinates of landmark 0
        - hand_size : the 2D distance from the wrist to the middle-finger
                      knuckle (landmark 9). Its x/y are measured in the
                      image, so it shrinks when you move away from the camera.
        Subtracting the wrist removes WHERE the hand is in the picture;
        dividing by hand_size removes HOW BIG it looks. A hand making the same
        shape anywhere in the picture now gives (nearly) the same numbers.
    """
    wrist = hand_landmarks[WRIST_INDEX]
    hand_size = np.linalg.norm(hand_landmarks[MIDDLE_FINGER_KNUCKLE_INDEX, :2] - wrist[:2])
    if not np.isfinite(hand_size) or hand_size < TINY_DISTANCE:
        return np.zeros_like(hand_landmarks)
    return (hand_landmarks - wrist) / hand_size


def compute_frame_features(canonical_sequence: np.ndarray) -> np.ndarray:
    """
    Turn a canonical landmark sequence into one feature vector per frame.

    Args:
        canonical_sequence: Output of canonicalize_hands(), shape (num_frames, 2, 21, 3),
            with at least one hand visible in every frame.

    Returns:
        Array of shape (num_frames, FEATURES_PER_FRAME) = (num_frames, 132).
        For each hand slot the layout is:
            [63 hand-shape numbers, 2 trajectory numbers, 1 presence flag]
        Missing hands are filled with zeros and presence flag 0.

    Trajectory formula:
        trajectory = (wrist_xy - reference_center) / reference_scale
        - wrist_xy         : the wrist's x, y in this frame
        - reference_center : the average wrist x, y of the DOMINANT hand
                             over the whole sequence
        - reference_scale  : the average hand size of the dominant hand
        This keeps information about how the hands MOVE (and where the
        second hand is relative to the first) while ignoring where in the
        picture the person is standing.

    Note: hand landmarks alone carry no information about where the hands
    are relative to the face or body, so signs that differ only in body
    location (for example "mom" at the chin vs "dad" at the forehead) are
    hard for this model to tell apart.
    """
    num_frames = canonical_sequence.shape[0]
    presence = hand_presence_mask(canonical_sequence)
    features = np.zeros((num_frames, FEATURES_PER_FRAME), dtype=np.float32)

    # Choose a reference point and scale from the dominant hand (slot 0).
    # If (rarely) the dominant hand never appears, fall back to slot 1.
    reference_slot = 0 if presence[:, 0].any() else 1
    reference_frames = canonical_sequence[presence[:, reference_slot], reference_slot]
    reference_center = reference_frames[:, WRIST_INDEX, :2].mean(axis=0)
    hand_sizes = np.linalg.norm(
        reference_frames[:, MIDDLE_FINGER_KNUCKLE_INDEX, :2] - reference_frames[:, WRIST_INDEX, :2], axis=1
    )
    reference_scale = float(hand_sizes.mean())
    if not np.isfinite(reference_scale) or reference_scale < TINY_DISTANCE:
        reference_scale = 1.0

    for frame_index in range(num_frames):
        for slot in range(NUM_HAND_SLOTS):
            if not presence[frame_index, slot]:
                continue  # leave zeros, presence flag stays 0
            hand = canonical_sequence[frame_index, slot]
            start = slot * FEATURES_PER_HAND
            shape_end = start + HAND_SHAPE_FEATURES
            trajectory_end = shape_end + HAND_TRAJECTORY_FEATURES
            features[frame_index, start:shape_end] = normalize_hand_shape(hand).reshape(-1)
            features[frame_index, shape_end:trajectory_end] = (hand[WRIST_INDEX, :2] - reference_center) / reference_scale
            features[frame_index, trajectory_end] = 1.0
    return features


def resample_sequence(frame_features: np.ndarray, target_length: int) -> np.ndarray:
    """
    Stretch or squeeze a sequence in time so it has exactly target_length steps.

    Args:
        frame_features: Array of shape (num_frames, num_features).
        target_length: The number of time steps the model expects.

    Returns:
        Array of shape (target_length, num_features).

    How it works (linear interpolation):
        We place target_length evenly spaced sample points along the original
        sequence. A point that falls between frame 3 and frame 4, 25% of the
        way along, gets 0.75 * frame3 + 0.25 * frame4. This is like playing
        the sign in slow motion or fast-forward until it has the right length.

    Why: the GRU model can technically read any length, but batching many
    samples together for training requires equal lengths, and making every
    sign the same length also removes differences in signing speed.
    """
    num_frames = frame_features.shape[0]
    if num_frames == 0:
        raise InsufficientHandDataError("Cannot resample an empty sequence.")
    if num_frames == 1:
        return np.repeat(frame_features, target_length, axis=0)

    sample_positions = np.linspace(0, num_frames - 1, target_length)
    lower_indices = np.floor(sample_positions).astype(int)
    upper_indices = np.minimum(lower_indices + 1, num_frames - 1)
    # How far each sample point is between its lower and upper frame (0 to 1).
    upper_weights = (sample_positions - lower_indices)[:, None]
    resampled = (1.0 - upper_weights) * frame_features[lower_indices] + upper_weights * frame_features[upper_indices]
    return resampled.astype(np.float32)


def prepare_canonical_landmarks(landmark_sequence, min_frames_with_hands: int = 1) -> np.ndarray:
    """
    Steps 1-3 of the pipeline: validate, drop empty frames, canonicalize.

    Args:
        landmark_sequence: Raw sequence of shape (num_frames, 2, 21, 3).
        min_frames_with_hands: Minimum number of frames that must contain a hand.

    Returns:
        Canonical landmark sequence (only frames with at least one hand).

    Raises:
        InsufficientHandDataError: if too few frames contain a hand.

    This is separated from the feature step so the training code can apply
    data augmentation (random rotations etc.) to canonical landmarks before
    features are computed.
    """
    sequence = validate_landmark_sequence(landmark_sequence)
    sequence = remove_frames_without_hands(sequence)
    if sequence.shape[0] < max(1, min_frames_with_hands):
        raise InsufficientHandDataError(
            f"Only {sequence.shape[0]} frame(s) contain a visible hand; "
            f"at least {max(1, min_frames_with_hands)} are needed."
        )
    return canonicalize_hands(sequence)


def canonical_landmarks_to_features(canonical_sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Steps 4-5 of the pipeline: compute per-frame features, then resample in time.

    Returns:
        Array of shape (sequence_length, FEATURES_PER_FRAME).
    """
    return resample_sequence(compute_frame_features(canonical_sequence), sequence_length)


def landmarks_to_model_input(
    landmark_sequence, sequence_length: int, min_frames_with_hands: int = 1
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on one raw landmark sequence.

    Args:
        landmark_sequence: Raw sequence of shape (num_frames, 2, 21, 3).
        sequence_length: Number of time steps the model expects (from config).
        min_frames_with_hands: Reject sequences with fewer frames containing a hand.

    Returns:
        Model-ready features of shape (sequence_length, 132).

    Raises:
        ValueError / InsufficientHandDataError for unusable input.
    """
    canonical = prepare_canonical_landmarks(landmark_sequence, min_frames_with_hands)
    return canonical_landmarks_to_features(canonical, sequence_length)


def count_frames_with_hands(landmark_sequence: np.ndarray) -> Tuple[int, int]:
    """
    Report how many frames contain at least one hand.

    Returns:
        (frames_with_a_hand, total_frames). Used by the data collection tool
        to reject recordings where the hands were mostly out of view.
    """
    sequence = validate_landmark_sequence(landmark_sequence)
    return int(hand_presence_mask(sequence).any(axis=1).sum()), int(sequence.shape[0])
