"""
landmarks.py

Detects hands in webcam images and extracts their landmarks with MediaPipe.

A landmark is a point marking a location on the hand, such as a fingertip
or a knuckle. MediaPipe's hand model finds 21 of them per hand:
    0 = wrist, 1-4 = thumb, 5-8 = index finger, 9-12 = middle finger,
    13-16 = ring finger, 17-20 = little finger (the last number of each
    finger is its tip).

IMPORTANT: MediaPipe only tells us WHERE the hand points are. It does not
know anything about sign language. Recognizing a sign is the job of our
own neural network (models/model.py), which learns from labelled examples.

Main pieces:
    HandDetectionResult  - the landmarks found in one frame.
    HandLandmarkDetector - wraps MediaPipe Hands; call detect(frame).

Used by: app.py and data_collection/collect_samples.py. The landmarks it
returns are turned into model features by vision/preprocessing.py.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from vision.preprocessing import LEFT_HAND_SLOT, RIGHT_HAND_SLOT, empty_frame_landmarks

# Short names for the MediaPipe modules we use.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class HandDetectionResult:
    """
    The hands found in a single video frame.

    Attributes:
        frame_landmarks: Array of shape (2, 21, 3). Slot 0 = hand labelled
            "Left", slot 1 = hand labelled "Right". Missing hands are NaN.
            This is exactly one frame of the "landmark sequence" format
            described in vision/preprocessing.py.
        handedness_labels: e.g. ["Right"] or ["Left", "Right"].
        handedness_scores: MediaPipe's confidence (0-1) in each label.
        raw_results: MediaPipe's original output, kept only so we can draw
            the hand skeleton on the image.
    """

    frame_landmarks: np.ndarray
    handedness_labels: List[str] = field(default_factory=list)
    handedness_scores: List[float] = field(default_factory=list)
    raw_results: Optional[Any] = None

    @property
    def num_hands(self) -> int:
        """How many hands were found (0, 1 or 2)."""
        return len(self.handedness_labels)


class HandLandmarkDetector:
    """
    Finds hands in BGR video frames and returns their 21 landmarks each.

    Create it ONCE and reuse it for every frame: MediaPipe loads its model
    when the detector is created, and in video mode it also tracks hands
    from one frame to the next, which is faster and steadier than detecting
    them from scratch each time.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            max_num_hands: Maximum number of hands to look for.
            model_complexity: 0 (faster) or 1 (more accurate).
            min_detection_confidence: How sure MediaPipe must be (0-1) to
                report a newly found hand.
            min_tracking_confidence: How sure it must be to keep tracking a
                hand between frames; below this it re-runs full detection.
        """
        # static_image_mode=False means "this is a video": MediaPipe tracks
        # hands across frames instead of treating each frame independently.
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_frame: np.ndarray) -> HandDetectionResult:
        """
        Find hands in one frame.

        Args:
            bgr_frame: Image from OpenCV, shape (height, width, 3), BGR colours.

        Returns:
            A HandDetectionResult. If no hands are visible, frame_landmarks is
            all NaN and handedness_labels is empty - this is a normal
            situation, not an error.
        """
        # OpenCV uses BGR colour order but MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Marking the image read-only lets MediaPipe avoid copying it.
        rgb_frame.flags.writeable = False
        # MediaPipe estimates the locations of the 21 points on each hand.
        results = self._hands.process(rgb_frame)

        frame_landmarks = empty_frame_landmarks()
        labels: List[str] = []
        scores: List[float] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                classification = handedness.classification[0]
                label = classification.label  # "Left" or "Right"
                slot = LEFT_HAND_SLOT if label == "Left" else RIGHT_HAND_SLOT
                # If MediaPipe gives both hands the same label (it can happen
                # when hands cross), keep the first one and skip the duplicate
                # rather than silently overwriting it.
                if np.isfinite(frame_landmarks[slot]).all():
                    continue
                # Each landmark has x, y in 0-1 (fractions of the image size)
                # and z = depth relative to the wrist.
                frame_landmarks[slot] = np.array(
                    [[point.x, point.y, point.z] for point in hand_landmarks.landmark], dtype=np.float32
                )
                labels.append(label)
                scores.append(float(classification.score))

        return HandDetectionResult(
            frame_landmarks=frame_landmarks,
            handedness_labels=labels,
            handedness_scores=scores,
            raw_results=results,
        )

    def close(self) -> None:
        """Release MediaPipe's resources."""
        self._hands.close()

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def draw_hand_landmarks(bgr_frame: np.ndarray, detection: HandDetectionResult) -> np.ndarray:
    """
    Draw the detected hand skeleton (dots and connecting lines) on a copy of the frame.

    Args:
        bgr_frame: The original image.
        detection: The result returned by HandLandmarkDetector.detect().

    Returns:
        A new image with the landmarks drawn on it (the original is unchanged).
    """
    annotated_frame = bgr_frame.copy()
    results = detection.raw_results
    if results is None or not results.multi_hand_landmarks:
        return annotated_frame
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return annotated_frame
