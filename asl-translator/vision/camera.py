"""
camera.py

Opens a webcam and reads video frames from it using OpenCV.

Why this file exists:
    Every part of the project that needs live video (the Streamlit app and
    the data collection tool) uses this one small class, so camera handling
    (opening, mirroring, error messages, closing) is written only once.

Main pieces:
    Camera                    - opens a webcam, returns frames, releases it.
    list_available_cameras()  - finds which camera numbers actually work.

A "frame" is a single picture from the video stream. OpenCV stores it as a
NumPy array of shape (height, width, 3). The 3 colour channels are in
BGR order (blue, green, red) - not the more common RGB - which is why other
modules convert the colours before handing frames to MediaPipe or Streamlit.

Used by: app.py and data_collection/collect_samples.py.
"""

import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


def _open_video_capture(camera_index: int) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture, preferring DirectShow on Windows.

    Why: on Windows the default backend can take several seconds to open a
    webcam; DirectShow (CAP_DSHOW) is usually much faster.
    """
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(camera_index)


class Camera:
    """
    A small wrapper around OpenCV's VideoCapture.

    Example:
        with Camera(camera_index=0) as camera:
            frame = camera.read_frame()
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, mirror: bool = True):
        """
        Args:
            camera_index: Which camera to open (0 = default webcam).
            width, height: Requested resolution. The camera may pick the
                closest size it supports.
            mirror: Flip frames left-to-right so the preview acts like a mirror.
                MediaPipe's hand labels assume this mirrored "selfie" view.
        """
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.mirror = mirror
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the webcam. Raises CameraError with a helpful message if it fails."""
        if self.is_open:
            return
        capture = _open_video_capture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera {self.camera_index}. Check that a webcam is "
                "connected, that no other program (Zoom, Teams, another browser tab) "
                "is using it, and that your operating system allows camera access."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self._capture = capture

    @property
    def is_open(self) -> bool:
        """True while the webcam is open and usable."""
        return self._capture is not None and self._capture.isOpened()

    def read_frame(self) -> np.ndarray:
        """
        Grab the next frame from the webcam.

        Returns:
            A BGR image as a NumPy array of shape (height, width, 3).

        Raises:
            CameraError: if the camera is not open or returns no image.
        """
        if not self.is_open:
            raise CameraError("The camera is not open. Call open() first.")
        frame_was_read, frame = self._capture.read()
        if not frame_was_read or frame is None:
            raise CameraError(f"Camera {self.camera_index} did not return a frame.")
        if self.mirror:
            # flipCode=1 flips around the vertical axis (left <-> right).
            frame = cv2.flip(frame, 1)
        return frame

    def actual_resolution(self) -> Tuple[int, int]:
        """Return the (width, height) the camera is really using, or (0, 0) if closed."""
        if not self.is_open:
            return 0, 0
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        """Close the webcam so other programs can use it."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def list_available_cameras(max_index_to_check: int = 4) -> List[int]:
    """
    Try camera numbers 0, 1, 2, ... and report which ones open successfully.

    Args:
        max_index_to_check: Highest camera number to try.

    Returns:
        A list of working camera indices (may be empty).

    Note: opening a camera takes a moment, so the app calls this once and
    caches the result rather than calling it on every refresh.
    """
    working_indices = []
    for camera_index in range(max_index_to_check + 1):
        capture = _open_video_capture(camera_index)
        if capture.isOpened():
            frame_was_read, _ = capture.read()
            if frame_was_read:
                working_indices.append(camera_index)
        capture.release()
    return working_indices
