"""Tests for training/dataset.py - storing, validating and splitting samples."""

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import make_sequence
from training.dataset import (
    SampleStore,
    SignSequenceDataset,
    augment_canonical_landmarks,
    split_by_signer,
    validate_index,
)
from translation.vocabulary import Vocabulary
from vision.preprocessing import FEATURES_PER_FRAME, empty_frame_landmarks, prepare_canonical_landmarks

VOCABULARY = Vocabulary(["horizontal", "vertical"])


def fill_store(store, signers=("a", "b", "c", "d"), per_label=3):
    for signer in signers:
        for label in VOCABULARY.class_names:
            for _ in range(per_label):
                store.add_sample(make_sequence(label, 20), label, signer)


def test_store_saves_and_counts_samples(tmp_path):
    store = SampleStore(tmp_path)
    sample_id = store.add_sample(make_sequence("horizontal", 12), "horizontal", "alex")
    store.add_sample(make_sequence("vertical", 12), "vertical", "alex")
    index = store.load_index()
    assert len(index) == 2
    assert sample_id in store.existing_sample_ids()
    assert store.count_by_label().to_dict() == {"horizontal": 1, "vertical": 1}
    loaded = store.load_landmarks(index.iloc[0]["file_path"])
    assert loaded.shape == (12, 2, 21, 3)


def test_store_requires_label_and_signer(tmp_path):
    store = SampleStore(tmp_path)
    with pytest.raises(ValueError):
        store.add_sample(make_sequence("horizontal"), "", "alex")
    with pytest.raises(ValueError):
        store.add_sample(make_sequence("horizontal"), "horizontal", " ")


def test_validation_removes_bad_rows(tmp_path):
    store = SampleStore(tmp_path)
    fill_store(store, signers=("a",), per_label=1)
    store.add_sample(make_sequence("curl"), "curl", "a")  # label not in vocabulary
    index = store.load_index()
    index = pd.concat([index, index.iloc[[0]]])  # duplicate row
    (tmp_path / index.iloc[1]["file_path"]).unlink()  # missing file
    clean, report = validate_index(index, tmp_path, VOCABULARY)
    assert len(clean) == 1
    assert report.unknown_labels == {"curl": 1}
    assert len(report.missing_files) == 1
    assert len(report.duplicate_ids) == 1


def test_signer_split_has_no_overlap(tmp_path):
    store = SampleStore(tmp_path)
    fill_store(store, signers=("a", "b", "c", "d", "e"))
    splits, method = split_by_signer(store.load_index(), 0.2, 0.2, random_seed=1)
    signer_sets = {name: set(rows["signer_id"]) for name, rows in splits.items()}
    assert "signer split" in method
    assert all(signer_sets.values()), "every split needs at least one signer"
    assert not signer_sets["train"] & signer_sets["test"]
    assert not signer_sets["train"] & signer_sets["validation"]
    assert not signer_sets["validation"] & signer_sets["test"]
    assert sum(len(rows) for rows in splits.values()) == len(store.load_index())


def test_split_with_exactly_three_signers_fills_every_set(tmp_path):
    store = SampleStore(tmp_path)
    fill_store(store, signers=("a", "b", "c"))
    splits, _ = split_by_signer(store.load_index(), 0.4, 0.4, random_seed=0)
    assert all(len(rows) > 0 for rows in splits.values())


def test_single_signer_falls_back_to_random_split(tmp_path):
    store = SampleStore(tmp_path)
    fill_store(store, signers=("only",), per_label=10)
    splits, method = split_by_signer(store.load_index(), 0.2, 0.2)
    assert "SAME people" in method
    ids = [set(rows["sample_id"]) for rows in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for rows in splits.values():
        assert set(rows["label"]) == set(VOCABULARY.class_names)


def test_dataset_returns_tensors_and_skips_unusable_samples(tmp_path):
    store = SampleStore(tmp_path)
    fill_store(store, signers=("a",), per_label=2)
    store.add_sample(np.repeat(empty_frame_landmarks()[None], 10, axis=0), "vertical", "a")
    dataset = SignSequenceDataset(store.load_index(), store, VOCABULARY, sequence_length=16, augment=True)
    assert len(dataset) == 4
    assert len(dataset.skipped_samples) == 1
    features, label = dataset[0]
    assert features.shape == (16, FEATURES_PER_FRAME)
    assert int(label) in (0, 1)
    assert dataset.class_counts().tolist() == [2, 2]


def test_augmentation_keeps_layout_and_missing_hands():
    canonical = prepare_canonical_landmarks(make_sequence("horizontal", 30))
    augmented = augment_canonical_landmarks(canonical, np.random.default_rng(3))
    assert augmented.shape[1:] == (2, 21, 3)
    assert 24 <= augmented.shape[0] <= 30
    assert np.isnan(augmented[:, 1]).all()  # the absent second hand stays absent
    assert np.isfinite(augmented[:, 0]).all()
