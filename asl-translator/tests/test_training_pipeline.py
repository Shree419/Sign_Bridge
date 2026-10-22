"""
End-to-end smoke test: prepare data -> train -> save -> evaluate -> predict.

Uses tiny SYNTHETIC data (see tests/synthetic.py) and a very small model,
only to check that the pipeline runs and produces its output files. It
does not measure real sign recognition quality.
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from models.predictor import SignPredictor
from tests.synthetic import make_sequence
from training.dataset import SampleStore
from training.prepare_islr import parquet_to_landmark_sequence, prepare_islr
from training.train import TrainingDataError, train
from utils.config import load_config

LABELS = ["horizontal", "vertical", "curl"]


def make_test_config(tmp_path):
    config = copy.deepcopy(load_config())
    config["paths"] = {
        "samples_dir": str(tmp_path / "samples"),
        "islr_dataset_dir": str(tmp_path / "islr"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "reports_dir": str(tmp_path / "reports"),
    }
    config["vocabulary"] = {"signs": LABELS, "display_names": {}}
    config["model"].update(hidden_size=16, num_layers=1)
    config["features"].update(sequence_length=16, min_frames_with_hands=4)
    config["training"].update(epochs=3, batch_size=8, early_stopping_patience=2, min_samples_per_class=3)
    return config


def test_train_evaluate_and_predict(tmp_path):
    config = make_test_config(tmp_path)
    store = SampleStore(config["paths"]["samples_dir"])
    rng = np.random.default_rng(0)
    for signer in ("s1", "s2", "s3", "s4", "s5"):
        for label in LABELS:
            for _ in range(4):
                store.add_sample(make_sequence(label, 20, noise=0.003, rng=rng), label, signer)

    checkpoint_path = train(config)

    assert checkpoint_path.exists()
    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "labels.json").exists()
    splits = pd.read_csv(checkpoint_dir / "splits.csv")
    train_signers = set(splits.loc[splits["split"] == "train", "signer_id"])
    test_signers = set(splits.loc[splits["split"] == "test", "signer_id"])
    assert not train_signers & test_signers

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "training_curves.png").exists()
    assert (reports_dir / "confusion_matrix.png").exists()
    report = json.loads((reports_dir / "evaluation_report.json").read_text())
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["latency"]["mean_ms"] > 0

    predictor = SignPredictor.from_checkpoint(checkpoint_path)
    result = predictor.predict(make_sequence("vertical", 25))
    assert result.is_ok and result.label in LABELS


def test_training_refuses_too_little_data(tmp_path):
    config = make_test_config(tmp_path)
    store = SampleStore(config["paths"]["samples_dir"])
    store.add_sample(make_sequence("horizontal"), "horizontal", "s1")
    with pytest.raises(TrainingDataError):
        train(config)


def write_fake_islr_clip(path, num_frames=6, right_hand_missing_frames=(2,)):
    """Write a parquet file laid out like the Kaggle asl-signs files."""
    rows = []
    hand = make_sequence("horizontal", num_frames)[:, 1]
    for frame in range(num_frames):
        for landmark in range(21):
            for hand_type in ("left_hand", "right_hand"):
                if hand_type == "right_hand" and frame not in right_hand_missing_frames:
                    x, y, z = hand[frame, landmark]
                else:
                    x = y = z = np.nan
                rows.append((frame + 100, f"{frame}-{hand_type}-{landmark}", hand_type, landmark, x, y, z))
        rows.append((frame + 100, f"{frame}-face-0", "face", 0, 0.5, 0.5, 0.0))  # ignored
    table = pd.DataFrame(rows, columns=["frame", "row_id", "type", "landmark_index", "x", "y", "z"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)


def test_islr_parquet_conversion(tmp_path):
    clip_path = tmp_path / "islr" / "train_landmark_files" / "7" / "123.parquet"
    write_fake_islr_clip(clip_path)
    sequence = parquet_to_landmark_sequence(clip_path)
    assert sequence.shape == (6, 2, 21, 3)
    assert np.isnan(sequence[:, 0]).all()  # left hand never present
    assert np.isnan(sequence[2, 1]).all()  # right hand missing in frame 2
    assert np.isfinite(sequence[0, 1]).all()

    pd.DataFrame(
        [{"path": "train_landmark_files/7/123.parquet", "participant_id": 7, "sequence_id": 123, "sign": "hello"}]
    ).to_csv(tmp_path / "islr" / "train.csv", index=False)
    samples_dir = tmp_path / "samples"
    assert prepare_islr(tmp_path / "islr", samples_dir, ["hello"]) == 1
    assert prepare_islr(tmp_path / "islr", samples_dir, ["hello"]) == 0  # already converted
    index = SampleStore(samples_dir).load_index()
    assert index.iloc[0]["signer_id"] == "islr_7"
    with pytest.raises(ValueError):
        prepare_islr(tmp_path / "islr", samples_dir, ["not_a_sign"])
