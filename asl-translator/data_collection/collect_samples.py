"""
collect_samples.py

Records your own labelled training examples through the webcam.

Why this file exists:
    A model can only recognize signs it has been trained on, and it works
    best on people and conditions similar to its training data. Recording
    your own examples lets you (a) add new signs, and (b) teach the model
    what YOUR signing, camera and lighting look like.

What is saved:
    ONLY hand landmark coordinates (numbers), in the same format as the
    imported dataset (see data/README.md). Webcam video is NOT saved unless
    you explicitly pass --save-video.

How many samples are enough?
    There is no magic number, and a handful is NOT enough for reliable
    recognition. As a rough starting point, aim for at least 30-50 samples
    per sign, recorded across several sessions with small variations
    (different distance, lighting, position, speed). Examples from several
    different people help the model work for people it has not seen.

Using new samples:
    Samples are added to data/samples/index.csv. Add any new sign names to
    vocabulary.signs in config.yaml, then retrain with
        python -m training.train

Controls (in the video window):
    SPACE  record one sample (after a short countdown)
    L      change the sign label (type it in the terminal)
    Q/ESC  quit

Run:
    python -m data_collection.collect_samples --sign hello --signer-id alex
    python -m data_collection.collect_samples --list        (show sample counts)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from training.dataset import SampleStore
from utils.config import load_config, resolve_path
from vision.camera import Camera, CameraError
from vision.landmarks import HandLandmarkDetector, draw_hand_landmarks
from vision.preprocessing import count_frames_with_hands

WINDOW_NAME = "ASL sample recorder (SPACE = record, L = change label, Q = quit)"
SOURCE_NAME = "custom"
# A recording is rejected if hands were visible in less than this fraction
# of its frames, because it would mostly teach the model about empty frames.
MIN_FRACTION_FRAMES_WITH_HANDS = 0.6


def draw_status(frame: np.ndarray, lines: List[str], color=(255, 255, 255)) -> None:
    """Write a few lines of text in the top-left corner of the frame (in place)."""
    for line_number, text in enumerate(lines):
        position = (10, 30 + 28 * line_number)
        # Draw a thick black outline first so the text is readable on any background.
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def print_sample_counts(store: SampleStore) -> None:
    """Print how many samples exist for each sign."""
    counts = store.count_by_label()
    if counts.empty:
        print("No samples recorded yet.")
    else:
        print("Samples per sign:")
        print(counts.to_string())


def record_one_sample(
    camera: Camera,
    detector: HandLandmarkDetector,
    frames_per_sample: int,
    countdown_seconds: float,
    label: str,
    video_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Show a countdown, then capture `frames_per_sample` frames of landmarks.

    Args:
        camera, detector: Already-open camera and hand detector.
        frames_per_sample: How many frames to record.
        countdown_seconds: Time to get ready before recording starts.
        label: Shown on screen while recording.
        video_path: If given, the raw video is ALSO saved here.

    Returns:
        The landmark sequence (frames_per_sample, 2, 21, 3), or None if the
        user cancelled with Q during recording.
    """
    countdown_end = time.time() + countdown_seconds
    while time.time() < countdown_end:
        frame = camera.read_frame()
        draw_status(frame, [f"Get ready to sign '{label}'", f"{countdown_end - time.time():.1f} s"], (0, 255, 255))
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return None

    video_writer = None
    recorded_frames = []
    try:
        for frame_number in range(frames_per_sample):
            frame = camera.read_frame()
            if video_path is not None and video_writer is None:
                height, width = frame.shape[:2]
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            if video_writer is not None:
                video_writer.write(frame)

            detection = detector.detect(frame)
            recorded_frames.append(detection.frame_landmarks)
            preview = draw_hand_landmarks(frame, detection)
            draw_status(preview, [f"RECORDING '{label}'", f"frame {frame_number + 1}/{frames_per_sample}"], (0, 0, 255))
            cv2.imshow(WINDOW_NAME, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return None
    finally:
        if video_writer is not None:
            video_writer.release()
    return np.stack(recorded_frames)


def run_collector(args: argparse.Namespace, config: dict) -> int:
    """Open the camera and run the interactive recording loop."""
    store = SampleStore(resolve_path(config["paths"]["samples_dir"]))
    label = args.sign or input("Name of the sign to record (e.g. hello): ").strip()
    if not label:
        print("A sign name is required.", file=sys.stderr)
        return 1
    known_signs = config["vocabulary"]["signs"]
    if label not in known_signs:
        print(f"Note: '{label}' is not yet in vocabulary.signs in config.yaml. Add it before retraining.")

    camera_config = config["camera"]
    tracking = config["hand_tracking"]
    samples_this_session = 0

    try:
        with Camera(args.camera, camera_config["width"], camera_config["height"], camera_config["mirror"]) as camera, \
                HandLandmarkDetector(
                    max_num_hands=tracking["max_num_hands"],
                    model_complexity=tracking["model_complexity"],
                    min_detection_confidence=tracking["min_detection_confidence"],
                    min_tracking_confidence=tracking["min_tracking_confidence"],
                ) as detector:
            last_message = "Press SPACE to record a sample."
            # Read the counts once (not every frame) and update them as we save.
            saved_counts = store.count_by_label().to_dict()
            while samples_this_session < args.num_samples:
                frame = camera.read_frame()
                detection = detector.detect(frame)
                preview = draw_hand_landmarks(frame, detection)
                count_for_label = int(saved_counts.get(label, 0))
                draw_status(
                    preview,
                    [
                        f"Sign: '{label}'  (saved: {count_for_label})",
                        f"Hands visible: {detection.num_hands}",
                        f"This session: {samples_this_session}/{args.num_samples}",
                        last_message,
                    ],
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    new_label = input("New sign name: ").strip()
                    if new_label:
                        label = new_label
                        last_message = f"Label changed to '{label}'."
                if key == ord(" "):
                    video_path = None
                    if args.save_video:
                        video_path = resolve_path(config["paths"]["samples_dir"]) / "custom_videos" / (
                            f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        )
                    sequence = record_one_sample(
                        camera, detector, args.frames_per_sample, args.countdown, label, video_path
                    )
                    if sequence is None:
                        break
                    frames_with_hands, total_frames = count_frames_with_hands(sequence)
                    if frames_with_hands < MIN_FRACTION_FRAMES_WITH_HANDS * total_frames:
                        last_message = (
                            f"Rejected: hands seen in only {frames_with_hands}/{total_frames} frames. Try again."
                        )
                        continue
                    store.add_sample(sequence, label, args.signer_id, source=SOURCE_NAME)
                    saved_counts[label] = saved_counts.get(label, 0) + 1
                    samples_this_session += 1
                    last_message = f"Saved sample ({frames_with_hands}/{total_frames} frames had hands)."
    except CameraError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    print(f"Recorded {samples_this_session} sample(s) this session.")
    print_sample_counts(store)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Record labelled ASL landmark samples from the webcam.")
    parser.add_argument("--sign", help="Label of the sign to record, e.g. hello.")
    parser.add_argument(
        "--signer-id",
        default="me",
        help="A name or code for the person signing. Used to split data by person.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Stop after this many saved samples.")
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=config["recognition"]["window_frames"],
        help="Frames per recording. Matching the app's window size keeps training and live data alike.",
    )
    parser.add_argument("--countdown", type=float, default=2.0, help="Seconds to get ready before recording.")
    parser.add_argument("--camera", type=int, default=config["camera"]["index"], help="Camera index.")
    parser.add_argument("--save-video", action="store_true", help="ALSO save the raw video (off by default).")
    parser.add_argument("--list", action="store_true", help="Only print how many samples exist per sign.")
    args = parser.parse_args(argv)

    if args.list:
        print_sample_counts(SampleStore(resolve_path(config["paths"]["samples_dir"])))
        return 0
    return run_collector(args, config)


if __name__ == "__main__":
    sys.exit(main())
