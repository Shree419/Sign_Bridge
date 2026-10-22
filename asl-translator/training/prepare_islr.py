"""
prepare_islr.py

Converts the Kaggle "Google - Isolated Sign Language Recognition" dataset
(competition name: asl-signs) into this project's sample format.

About the dataset:
    - About 94,000 short clips of 250 isolated ASL signs, performed by
      21 different people (identified by participant_id).
    - The videos are NOT included. Google already ran MediaPipe Holistic on
      them and published only the landmark coordinates, as one .parquet file
      per clip. That is exactly the kind of data our model needs, and it
      lets us split the data by signer.
    - License: CC BY 4.0 (Creative Commons Attribution). The data was created
      by the Deaf Professional Arts Network and the Georgia Institute of
      Technology. You may use and adapt it as long as you credit them; see
      "Dataset license and attribution" in README.md.
    - Access: you need a Kaggle account and must accept the competition's
      rules on the Kaggle website before Kaggle lets you download.
    - Layout after download and unzip:
          train.csv                     columns: path, participant_id, sequence_id, sign
          train_landmark_files/<participant_id>/<sequence_id>.parquet
      Each parquet file has one row per landmark per frame, with columns
      frame, row_id, type (face / left_hand / pose / right_hand),
      landmark_index, x, y, z.  Landmarks that were not detected are NaN.

What this script does:
    1. Reads train.csv and keeps only the signs listed in config.yaml.
    2. For each clip, keeps only the left_hand and right_hand rows (our live
       app tracks hands only, so face and pose data would not be available
       at prediction time and must not be used for training either).
    3. Builds an array of shape (num_frames, 2, 21, 3) and saves it with
       SampleStore. Re-running it skips clips that were already converted.

Run:
    python -m training.prepare_islr
    python -m training.prepare_islr --dataset-dir D:/datasets/asl-signs --max-per-sign 200

Used by: you, once, before training (python -m training.train).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from training.dataset import SampleStore
from utils.config import load_config, resolve_path
from vision.preprocessing import LEFT_HAND_SLOT, NUM_HAND_LANDMARKS, RIGHT_HAND_SLOT, empty_frame_landmarks

SOURCE_NAME = "islr"
HAND_TYPE_TO_SLOT = {"left_hand": LEFT_HAND_SLOT, "right_hand": RIGHT_HAND_SLOT}


def parquet_to_landmark_sequence(parquet_path: Path) -> np.ndarray:
    """
    Read one clip's parquet file and return its hand landmarks.

    Args:
        parquet_path: Path to <sequence_id>.parquet.

    Returns:
        Array of shape (num_frames, 2, 21, 3); NaN where a hand was not found.
    """
    clip = pd.read_parquet(parquet_path)
    # Some copies of the dataset call the column "landmark_idx" instead.
    landmark_column = "landmark_index" if "landmark_index" in clip.columns else "landmark_idx"
    required = {"frame", "type", landmark_column, "x", "y", "z"}
    if not required.issubset(clip.columns):
        raise ValueError(f"{parquet_path} is missing columns {sorted(required - set(clip.columns))}")

    frame_numbers = np.sort(clip["frame"].unique())
    sequence = np.repeat(empty_frame_landmarks()[np.newaxis], len(frame_numbers), axis=0)

    hand_rows = clip[clip["type"].isin(list(HAND_TYPE_TO_SLOT))]
    hand_rows = hand_rows[hand_rows[landmark_column].between(0, NUM_HAND_LANDMARKS - 1)]
    # np.searchsorted converts each frame number into its position 0..num_frames-1.
    frame_positions = np.searchsorted(frame_numbers, hand_rows["frame"].to_numpy())
    slots = hand_rows["type"].map(HAND_TYPE_TO_SLOT).to_numpy()
    landmark_numbers = hand_rows[landmark_column].to_numpy().astype(int)
    sequence[frame_positions, slots, landmark_numbers] = hand_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return sequence


def prepare_islr(
    dataset_dir: Path, samples_dir: Path, signs: List[str], max_per_sign: Optional[int] = None
) -> int:
    """
    Convert the selected signs of the Kaggle dataset into the project format.

    Args:
        dataset_dir: Folder containing train.csv and train_landmark_files/.
        samples_dir: Where SampleStore keeps the converted samples.
        signs: Which sign labels to import.
        max_per_sign: Optional limit per sign (handy for a quick first try).

    Returns:
        The number of newly converted clips.
    """
    train_csv = dataset_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Could not find {train_csv}.\n"
            "Download the dataset from https://www.kaggle.com/competitions/asl-signs/data "
            "(after accepting the competition rules), unzip it, and either place it at that "
            "path or pass --dataset-dir."
        )

    clips = pd.read_csv(train_csv)
    available_signs = set(clips["sign"].unique())
    unknown_signs = [sign for sign in signs if sign not in available_signs]
    if unknown_signs:
        raise ValueError(
            f"These signs are not in the dataset: {unknown_signs}. "
            f"Some available signs: {sorted(available_signs)[:40]} ..."
        )

    clips = clips[clips["sign"].isin(signs)]
    if max_per_sign:
        # Take clips from as many different people as possible by sampling
        # randomly (with a fixed seed so it is repeatable).
        clips = clips.groupby("sign", group_keys=False).apply(
            lambda rows: rows.sample(n=min(len(rows), max_per_sign), random_state=0)
        )

    store = SampleStore(samples_dir)
    already_converted = store.existing_sample_ids()
    new_index_rows = []
    failures = 0
    newly_converted = 0

    for clip_number, clip in enumerate(clips.itertuples(index=False), start=1):
        sample_id = f"{SOURCE_NAME}_{clip.sequence_id}"
        if sample_id in already_converted:
            continue
        try:
            sequence = parquet_to_landmark_sequence(dataset_dir / clip.path)
        except (OSError, ValueError) as error:
            failures += 1
            print(f"  Skipping {clip.path}: {error}")
            continue
        relative_path = Path(SOURCE_NAME) / f"{sample_id}.npy"
        (samples_dir / relative_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(samples_dir / relative_path, sequence)
        new_index_rows.append(
            {
                "sample_id": sample_id,
                "label": clip.sign,
                "signer_id": f"{SOURCE_NAME}_{clip.participant_id}",
                "source": SOURCE_NAME,
                "file_path": relative_path.as_posix(),
                "num_frames": int(sequence.shape[0]),
                "created_at": "",
            }
        )
        newly_converted += 1
        # Write the index in chunks so an interrupted run keeps its progress.
        if len(new_index_rows) >= 500:
            store.append_index_rows(new_index_rows)
            new_index_rows.clear()
        if clip_number % 200 == 0:
            print(f"  processed {clip_number}/{len(clips)} clips")

    store.append_index_rows(new_index_rows)
    print(f"Done. Newly converted clips: {newly_converted}. Failed: {failures}.")
    print("Samples per sign now available:")
    print(store.count_by_label().to_string())
    return newly_converted


def main(argv: Optional[List[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Convert the Kaggle asl-signs dataset into project samples.")
    parser.add_argument("--dataset-dir", type=Path, default=resolve_path(config["paths"]["islr_dataset_dir"]))
    parser.add_argument("--max-per-sign", type=int, default=None, help="Limit clips per sign (optional).")
    parser.add_argument(
        "--signs", nargs="+", default=config["vocabulary"]["signs"], help="Signs to import (default: config.yaml)."
    )
    args = parser.parse_args(argv)
    try:
        prepare_islr(args.dataset_dir, resolve_path(config["paths"]["samples_dir"]), args.signs, args.max_per_sign)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
