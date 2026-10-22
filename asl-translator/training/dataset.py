"""
dataset.py

Stores, checks, splits and loads the labelled training samples.

Key machine learning words used here:
    Sample   - one example: a recorded clip of one sign.
    Features - the numbers the model reads (here: processed hand landmarks).
    Label    - the correct answer for a sample (e.g. "hello"). Training
               data must have labels so the model can learn by comparing
               its guesses with the right answers ("supervised learning").
    Training / validation / test sets - see split_by_signer().

Storage format (documented in data/README.md):
    data/samples/index.csv                 one row per sample
    data/samples/<source>/<sample_id>.npy  that sample's landmark sequence,
                                           shape (num_frames, 2, 21, 3)

Main pieces:
    SampleStore          - saves new samples and reads the index.
    validate_index()     - finds missing files, unknown labels, duplicates.
    split_by_signer()    - makes train/validation/test sets without leakage.
    SignSequenceDataset  - feeds samples to PyTorch during training.
    augment_canonical_landmarks() - random variations for training only.

Used by: training/prepare_islr.py, training/train.py, training/evaluate.py,
data_collection/collect_samples.py and app.py (saving corrected examples).
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from translation.vocabulary import Vocabulary
from vision.preprocessing import (
    InsufficientHandDataError,
    canonical_landmarks_to_features,
    prepare_canonical_landmarks,
    validate_landmark_sequence,
)

INDEX_FILENAME = "index.csv"
INDEX_COLUMNS = ["sample_id", "label", "signer_id", "source", "file_path", "num_frames", "created_at"]


# ---------------------------------------------------------------------
# Saving and reading samples
# ---------------------------------------------------------------------
class SampleStore:
    """
    A folder of samples plus a CSV index describing them.

    Using one shared format for downloaded data AND your own recordings
    means the training code does not care where a sample came from.
    """

    def __init__(self, samples_dir: Union[str, Path]):
        self.samples_dir = Path(samples_dir)
        self.index_path = self.samples_dir / INDEX_FILENAME

    def load_index(self) -> pd.DataFrame:
        """
        Read index.csv into a pandas DataFrame (a table with named columns).

        Returns an empty table with the right columns if no samples exist yet.
        """
        if not self.index_path.exists():
            return pd.DataFrame(columns=INDEX_COLUMNS)
        index = pd.read_csv(self.index_path, dtype={"sample_id": str, "label": str, "signer_id": str, "source": str})
        missing_columns = set(INDEX_COLUMNS) - set(index.columns)
        if missing_columns:
            raise ValueError(f"{self.index_path} is missing columns: {sorted(missing_columns)}")
        return index

    def existing_sample_ids(self) -> set:
        """Return the IDs of every sample already in the index."""
        return set(self.load_index()["sample_id"])

    def add_sample(
        self,
        landmark_sequence: np.ndarray,
        label: str,
        signer_id: str,
        source: str = "custom",
        sample_id: Optional[str] = None,
    ) -> str:
        """
        Save one labelled landmark sequence and add it to the index.

        Args:
            landmark_sequence: Array of shape (num_frames, 2, 21, 3).
            label: The sign shown in the sample, e.g. "hello".
            signer_id: Who performed it. Used to split data by person.
            source: Which collection it came from ("custom", "islr", ...).
            sample_id: Optional unique ID; one is generated if omitted.

        Returns:
            The sample_id.
        """
        label = label.strip()
        signer_id = signer_id.strip()
        if not label:
            raise ValueError("A sample needs a non-empty label.")
        if not signer_id:
            raise ValueError("A sample needs a signer_id so data can be split by person.")
        sequence = validate_landmark_sequence(landmark_sequence)
        sample_id = sample_id or f"{source}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        relative_path = Path(source) / f"{sample_id}.npy"
        absolute_path = self.samples_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(absolute_path, sequence)
        self.append_index_rows(
            [
                {
                    "sample_id": sample_id,
                    "label": label,
                    "signer_id": signer_id,
                    "source": source,
                    "file_path": relative_path.as_posix(),
                    "num_frames": int(sequence.shape[0]),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            ]
        )
        return sample_id

    def append_index_rows(self, rows: List[Dict[str, object]]) -> None:
        """Append rows to index.csv, writing the header if the file is new."""
        if not rows:
            return
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        new_rows = pd.DataFrame(rows, columns=INDEX_COLUMNS)
        write_header = not self.index_path.exists()
        new_rows.to_csv(self.index_path, mode="a", header=write_header, index=False)

    def load_landmarks(self, file_path: str) -> np.ndarray:
        """Load one sample's landmark array using the file_path stored in the index."""
        return np.load(self.samples_dir / file_path)

    def count_by_label(self) -> pd.Series:
        """How many samples exist for each label, most common first."""
        index = self.load_index()
        if index.empty:
            return pd.Series(dtype=int)
        return index["label"].value_counts()


# ---------------------------------------------------------------------
# Checking the data
# ---------------------------------------------------------------------
@dataclass
class ValidationReport:
    """What validate_index() found. Problems are printed before training."""

    total_rows: int = 0
    kept_rows: int = 0
    missing_files: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    unknown_labels: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Samples in index: {self.total_rows}, usable: {self.kept_rows}"]
        if self.missing_files:
            lines.append(f"  {len(self.missing_files)} sample file(s) missing, e.g. {self.missing_files[:3]}")
        if self.duplicate_ids:
            lines.append(f"  {len(self.duplicate_ids)} duplicate sample id(s) removed")
        if self.unknown_labels:
            lines.append(f"  Ignored labels not in the vocabulary: {self.unknown_labels}")
        return "\n".join(lines)


def validate_index(
    index: pd.DataFrame, samples_dir: Union[str, Path], vocabulary: Vocabulary
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Remove rows that cannot or should not be used for training.

    Checks: the .npy file exists, the sample ID is not duplicated, the label
    is in the vocabulary, and a signer ID is present.

    Returns:
        (clean_index, report)
    """
    samples_dir = Path(samples_dir)
    report = ValidationReport(total_rows=len(index))
    clean = index.dropna(subset=["sample_id", "label", "signer_id", "file_path"]).copy()

    duplicated = clean["sample_id"].duplicated(keep="first")
    report.duplicate_ids = clean.loc[duplicated, "sample_id"].tolist()
    clean = clean[~duplicated]

    in_vocabulary = clean["label"].isin(vocabulary.class_names)
    report.unknown_labels = clean.loc[~in_vocabulary, "label"].value_counts().to_dict()
    clean = clean[in_vocabulary]

    file_exists = clean["file_path"].map(lambda relative: (samples_dir / relative).exists()).astype(bool)
    report.missing_files = clean.loc[~file_exists, "file_path"].tolist()
    clean = clean[file_exists]

    report.kept_rows = len(clean)
    return clean.reset_index(drop=True), report


# ---------------------------------------------------------------------
# Splitting the data
# ---------------------------------------------------------------------
def split_by_signer(
    index: pd.DataFrame, validation_fraction: float, test_fraction: float, random_seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Divide samples into training, validation and test sets.

    Training data teaches the model (its weights are adjusted on it).
    Validation data is checked after every epoch to watch for overfitting
        and to choose when to stop. The model never learns from it directly,
        but our choices (like when to stop) are influenced by it.
    Test data is used ONCE at the end, to measure performance on examples
        that played no part in training or tuning.

    Splitting by signer (person) instead of by sample:
        If the same person appears in both training and test sets, the test
        score mostly measures "can it recognize THIS person again", which
        overestimates how well it works for a new user. Keeping each person
        in only one set prevents this "data leakage" and tests whether the
        model generalizes to people it has never seen.

    If there are fewer than 3 signers, a signer split is impossible, so we
    fall back to a per-label random split and say so clearly - in that case
    the test score only tells you how well it works for the same people.

    Args:
        index: Clean sample table (from validate_index).
        validation_fraction, test_fraction: Approximate share of data for each.
        random_seed: Makes the split repeatable.

    Returns:
        ({"train": df, "validation": df, "test": df}, method_description)
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1 and sum to less than 1.")
    random_generator = np.random.default_rng(random_seed)
    signers = np.array(sorted(index["signer_id"].unique()))

    if len(signers) >= 3:
        random_generator.shuffle(signers)
        samples_per_signer = index["signer_id"].value_counts()
        total_samples = len(index)
        # Start with one signer in test, one in validation, the rest in train,
        # so every set is guaranteed at least one person.
        assignment: Dict[str, str] = {signers[0]: "test", signers[1]: "validation"}
        assignment.update({signer: "train" for signer in signers[2:]})
        samples_in_split = {"test": samples_per_signer[signers[0]], "validation": samples_per_signer[signers[1]]}
        train_signers = list(signers[2:])
        # Move more whole signers out of training until each set holds roughly
        # the requested share of SAMPLES (people can have very different
        # amounts of data), always leaving at least one signer for training.
        for split_name, fraction in (("test", test_fraction), ("validation", validation_fraction)):
            while samples_in_split[split_name] < fraction * total_samples and len(train_signers) > 1:
                moved_signer = train_signers.pop(0)
                assignment[moved_signer] = split_name
                samples_in_split[split_name] += samples_per_signer[moved_signer]
        split_names = index["signer_id"].map(assignment)
        method = f"signer split ({len(signers)} signers; no person appears in more than one set)"
    else:
        split_names = pd.Series("train", index=index.index)
        for _, label_rows in index.groupby("label"):
            shuffled_positions = random_generator.permutation(label_rows.index.to_numpy())
            num_test = max(1, int(round(len(shuffled_positions) * test_fraction)))
            num_validation = max(1, int(round(len(shuffled_positions) * validation_fraction)))
            split_names.loc[shuffled_positions[:num_test]] = "test"
            split_names.loc[shuffled_positions[num_test : num_test + num_validation]] = "validation"
        method = (
            f"random per-label split (only {len(signers)} signer(s) - the test score measures "
            "performance on the SAME people, not on new people)"
        )

    splits = {name: index[split_names == name].reset_index(drop=True) for name in ("train", "validation", "test")}
    return splits, method


# ---------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------
def augment_canonical_landmarks(canonical_sequence: np.ndarray, random_generator: np.random.Generator) -> np.ndarray:
    """
    Create a slightly different version of a training sample.

    Data augmentation shows the model realistic variations of each example
    (a slightly tilted hand, a differently shaped camera image, a sign
    started a little late), which helps it generalize instead of memorizing
    the exact training clips. It is applied ONLY to training data, never to
    validation or test data, so those measurements stay honest.

    Augmentations (each applied with some probability):
        - mirror left/right: makes the model work for either handedness
          convention and for camera images that are or are not mirrored
        - small rotation (about +/- 12 degrees) around the image centre
        - horizontal stretch: imitates cameras with different aspect ratios
        - trim a few frames from the start/end: imitates imperfect timing
        - tiny random noise: imitates landmark jitter

    Args:
        canonical_sequence: Output of prepare_canonical_landmarks().
        random_generator: NumPy random generator (so results are repeatable).

    Returns:
        A new array of the same layout (it may have a few frames fewer).
    """
    augmented = canonical_sequence.copy()
    x_coordinates = augmented[..., 0]
    y_coordinates = augmented[..., 1]

    if random_generator.random() < 0.5:
        x_coordinates[:] = 1.0 - x_coordinates

    if random_generator.random() < 0.7:
        # Rotation formula around the centre (0.5, 0.5) by angle theta:
        #   x' = cx + (x - cx) * cos(theta) - (y - cy) * sin(theta)
        #   y' = cy + (x - cx) * sin(theta) + (y - cy) * cos(theta)
        angle = np.deg2rad(random_generator.uniform(-12.0, 12.0))
        centered_x, centered_y = x_coordinates - 0.5, y_coordinates - 0.5
        rotated_x = 0.5 + centered_x * np.cos(angle) - centered_y * np.sin(angle)
        rotated_y = 0.5 + centered_x * np.sin(angle) + centered_y * np.cos(angle)
        x_coordinates[:], y_coordinates[:] = rotated_x, rotated_y

    if random_generator.random() < 0.5:
        x_coordinates[:] = 0.5 + (x_coordinates - 0.5) * random_generator.uniform(0.85, 1.15)

    num_frames = augmented.shape[0]
    if num_frames > 10 and random_generator.random() < 0.5:
        max_trim = max(1, num_frames // 10)
        trim_start = int(random_generator.integers(0, max_trim + 1))
        trim_end = int(random_generator.integers(0, max_trim + 1))
        augmented = augmented[trim_start : num_frames - trim_end]

    # NaN (missing hand) + noise stays NaN, so missing hands stay missing.
    augmented = augmented + random_generator.normal(0.0, 0.002, size=augmented.shape).astype(np.float32)
    return augmented.astype(np.float32)


# ---------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------
class SignSequenceDataset(Dataset):
    """
    Gives PyTorch one (features, label_index) pair at a time.

    A PyTorch Dataset must implement __len__ (how many samples) and
    __getitem__ (return sample number i). A DataLoader then groups samples
    into shuffled batches for training.

    All landmark files are loaded and canonicalized once, when the dataset
    is created, so each training epoch does not have to re-read the disk.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        store: SampleStore,
        vocabulary: Vocabulary,
        sequence_length: int,
        min_frames_with_hands: int = 1,
        augment: bool = False,
        random_seed: int = 0,
    ):
        """
        Args:
            index: Rows describing the samples in this split.
            store: Where to load the landmark files from.
            vocabulary: Converts labels into class indices.
            sequence_length: Time steps per model input.
            min_frames_with_hands: Samples with fewer hand frames are skipped.
            augment: Apply random augmentation (training set only!).
            random_seed: Seed for the augmentation randomness.
        """
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.augment = augment
        self.random_generator = np.random.default_rng(random_seed)

        self.canonical_sequences: List[np.ndarray] = []
        self.label_indices: List[int] = []
        self.sample_ids: List[str] = []
        self.skipped_samples: List[Tuple[str, str]] = []

        for row in index.itertuples(index=False):
            try:
                canonical = prepare_canonical_landmarks(store.load_landmarks(row.file_path), min_frames_with_hands)
            except (InsufficientHandDataError, ValueError, OSError) as error:
                self.skipped_samples.append((row.sample_id, str(error)))
                continue
            self.canonical_sequences.append(canonical)
            self.label_indices.append(vocabulary.label_to_index(row.label))
            self.sample_ids.append(row.sample_id)

    def __len__(self) -> int:
        return len(self.canonical_sequences)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: float tensor of shape (sequence_length, 132)
            label:    integer tensor holding the class index
        """
        canonical = self.canonical_sequences[position]
        if self.augment:
            canonical = augment_canonical_landmarks(canonical, self.random_generator)
        features = canonical_landmarks_to_features(canonical, self.sequence_length)
        return torch.from_numpy(features), torch.tensor(self.label_indices[position], dtype=torch.long)

    def class_counts(self) -> np.ndarray:
        """Number of samples per class index (used for class weights)."""
        return np.bincount(np.array(self.label_indices, dtype=int), minlength=self.vocabulary.num_classes)
