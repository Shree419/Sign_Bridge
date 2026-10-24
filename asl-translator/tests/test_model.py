"""Tests for models/model.py and models/predictor.py."""

import numpy as np
import pytest
import torch

from models.model import ModelConfig, SignSequenceClassifier, load_checkpoint, save_checkpoint
from models.predictor import (
    STATUS_NOT_ENOUGH_HAND_DATA,
    STATUS_INVALID_INPUT,
    ModelNotAvailableError,
    SignPredictor,
)
from tests.synthetic import make_sequence
from translation.vocabulary import Vocabulary
from vision.preprocessing import FEATURES_PER_FRAME, empty_frame_landmarks

VOCABULARY = Vocabulary(["alpha", "beta", "gamma"])


def make_model(**overrides):
    settings = dict(input_size=FEATURES_PER_FRAME, num_classes=3, hidden_size=16, num_layers=1)
    settings.update(overrides)
    return SignSequenceClassifier(ModelConfig(**settings))


def test_output_shape_is_batch_by_classes():
    model = make_model()
    logits = model(torch.zeros(4, 32, FEATURES_PER_FRAME))
    assert logits.shape == (4, 3)


def test_probabilities_sum_to_one():
    model = make_model(num_layers=2, bidirectional=False).eval()
    probabilities = model.predict_probabilities(torch.randn(2, 10, FEATURES_PER_FRAME))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


@pytest.mark.parametrize(
    "bad_input",
    [torch.zeros(32, FEATURES_PER_FRAME), torch.zeros(1, 32, FEATURES_PER_FRAME + 1), torch.zeros(1, 0, FEATURES_PER_FRAME)],
)
def test_invalid_input_raises_value_error(bad_input):
    with pytest.raises(ValueError):
        make_model()(bad_input)


def test_checkpoint_round_trip_gives_identical_predictions(tmp_path):
    model = make_model().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, VOCABULARY.to_dict(), {"sequence_length": 16, "min_frames_with_hands": 2})
    loaded_model, checkpoint = load_checkpoint(path)
    features = torch.randn(3, 16, FEATURES_PER_FRAME)
    torch.testing.assert_close(model(features), loaded_model(features))
    assert checkpoint["vocabulary"]["class_names"] == ["alpha", "beta", "gamma"]


def test_load_rejects_unrelated_files(tmp_path):
    path = tmp_path / "not_a_model.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_missing_checkpoint_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotAvailableError) as error_info:
        SignPredictor.from_checkpoint(tmp_path / "missing.pt")
    assert "python -m training.train" in str(error_info.value)


def test_predictor_returns_labels_and_handles_bad_input(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, make_model().eval(), VOCABULARY.to_dict(), {"sequence_length": 16, "min_frames_with_hands": 3})
    predictor = SignPredictor.from_checkpoint(path)

    result = predictor.predict(make_sequence("horizontal", 20))
    assert result.is_ok
    assert result.label in VOCABULARY
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_predictions) == 3
    assert result.top_predictions[0][0] == result.label
    np.testing.assert_allclose(result.probabilities.sum(), 1.0, rtol=1e-5)

    no_hands = np.repeat(empty_frame_landmarks()[None], 20, axis=0)
    assert predictor.predict(no_hands).status == STATUS_NOT_ENOUGH_HAND_DATA
    assert predictor.predict(np.zeros((5, 3))).status == STATUS_INVALID_INPUT


def test_predictor_rejects_mismatched_vocabulary():
    with pytest.raises(ValueError):
        SignPredictor(make_model(num_classes=2), VOCABULARY, sequence_length=16)
