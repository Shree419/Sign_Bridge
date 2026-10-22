"""
predictor.py

Uses a trained model to recognize the sign in a sequence of hand landmarks.
Using a trained model to make predictions on new data is called INFERENCE.

Why this file exists:
    Training and live recognition need the same preprocessing and the same
    label mapping. SignPredictor bundles the trained network, its Vocabulary
    and its preprocessing settings, all loaded from one checkpoint file, so
    the app only has to call predict(landmark_sequence).

How camera data becomes a prediction:
    webcam frame -> MediaPipe landmarks (vision/landmarks.py)
      -> recent frames collected into a sequence (the app's rolling buffer)
      -> preprocessing into features (vision/preprocessing.py)
      -> neural network -> logits -> softmax probabilities
      -> highest probability = predicted sign, its probability = confidence

Main pieces:
    ModelNotAvailableError - raised with setup instructions if no model exists.
    PredictionResult       - what one prediction returns.
    SignPredictor          - loads the model ONCE and reuses it for every frame.

Used by: app.py and training/evaluate.py.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from models.model import SignSequenceClassifier, load_checkpoint
from translation.vocabulary import Vocabulary
from vision.preprocessing import InsufficientHandDataError, landmarks_to_model_input

# Status codes a PredictionResult can carry.
STATUS_OK = "ok"
STATUS_NOT_ENOUGH_HAND_DATA = "not_enough_hand_data"
STATUS_INVALID_INPUT = "invalid_input"

MISSING_MODEL_HELP = (
    "No trained model was found at {path}.\n\n"
    "To create one:\n"
    "  1. Get training data - either download the Kaggle 'asl-signs' dataset and run\n"
    "       python -m training.prepare_islr\n"
    "     or record your own examples with\n"
    "       python -m data_collection.collect_samples --sign hello --signer-id me\n"
    "  2. Train the model:   python -m training.train\n"
    "See README.md for step-by-step instructions."
)


class ModelNotAvailableError(FileNotFoundError):
    """Raised when the trained model file is missing, with instructions to create it."""


@dataclass
class PredictionResult:
    """
    The outcome of one prediction.

    Attributes:
        status: STATUS_OK, STATUS_NOT_ENOUGH_HAND_DATA or STATUS_INVALID_INPUT.
        label: Predicted sign label (None unless status is ok).
        confidence: Probability the model gave the predicted sign (0-1).
        probabilities: One probability per class, in vocabulary order.
        top_predictions: The best few (label, probability) pairs, best first.
        latency_ms: How long preprocessing + the model took, in milliseconds.
        message: Human-readable explanation for non-ok statuses.
    """

    status: str
    label: Optional[str] = None
    confidence: float = 0.0
    probabilities: Optional[np.ndarray] = None
    top_predictions: List[Tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    message: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_OK


class SignPredictor:
    """
    Wraps a trained SignSequenceClassifier for real-time use.

    Create it once with SignPredictor.from_checkpoint(path) and keep it:
    loading weights from disk takes far longer than a single prediction.
    """

    def __init__(
        self,
        model: SignSequenceClassifier,
        vocabulary: Vocabulary,
        sequence_length: int,
        min_frames_with_hands: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        if model.config.num_classes != vocabulary.num_classes:
            raise ValueError(
                f"The model has {model.config.num_classes} outputs but the vocabulary has "
                f"{vocabulary.num_classes} signs; they must match."
            )
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.min_frames_with_hands = min_frames_with_hands
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ) -> "SignPredictor":
        """
        Load a predictor from a checkpoint written by training/train.py.

        Raises:
            ModelNotAvailableError: if the file does not exist (the message
                explains how to train a model).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ModelNotAvailableError(MISSING_MODEL_HELP.format(path=checkpoint_path))
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        feature_settings = checkpoint["feature_settings"]
        return cls(
            model=model,
            vocabulary=Vocabulary.from_dict(checkpoint["vocabulary"]),
            sequence_length=int(feature_settings["sequence_length"]),
            min_frames_with_hands=int(feature_settings.get("min_frames_with_hands", 1)),
            device=device,
        )

    @torch.inference_mode()
    def predict_features(self, feature_batch: np.ndarray) -> np.ndarray:
        """
        Run the network on already-preprocessed features.

        Args:
            feature_batch: Array of shape (batch_size, sequence_length, num_features).

        Returns:
            Probabilities of shape (batch_size, num_classes).

        torch.inference_mode() tells PyTorch we will not train here, so it can
        skip the bookkeeping needed for backpropagation and run faster.
        """
        features_tensor = torch.as_tensor(feature_batch, dtype=torch.float32, device=self.device)
        return self.model.predict_probabilities(features_tensor).cpu().numpy()

    def predict(self, landmark_sequence: np.ndarray, top_k: int = 3) -> PredictionResult:
        """
        Predict which sign appears in a sequence of hand landmarks.

        Args:
            landmark_sequence: Raw landmarks of shape (num_frames, 2, 21, 3),
                e.g. the most recent webcam frames.
            top_k: How many of the most likely signs to report.

        Returns:
            A PredictionResult. Invalid or hand-less input returns a result
            with a non-ok status instead of raising, so the live app never
            crashes because a hand briefly left the picture.
        """
        start_time = time.perf_counter()
        try:
            features = landmarks_to_model_input(
                landmark_sequence, self.sequence_length, self.min_frames_with_hands
            )
        except InsufficientHandDataError as error:
            return PredictionResult(status=STATUS_NOT_ENOUGH_HAND_DATA, message=str(error))
        except ValueError as error:
            return PredictionResult(status=STATUS_INVALID_INPUT, message=str(error))

        # The model expects a batch, so add a batch dimension of size 1:
        # (32, 132) -> (1, 32, 132).
        probabilities = self.predict_features(features[np.newaxis])[0]
        best_indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
        top_predictions = [(self.vocabulary.index_to_label(i), float(probabilities[i])) for i in best_indices]
        best_label, best_confidence = top_predictions[0]

        return PredictionResult(
            status=STATUS_OK,
            label=best_label,
            confidence=best_confidence,
            probabilities=probabilities,
            top_predictions=top_predictions,
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
