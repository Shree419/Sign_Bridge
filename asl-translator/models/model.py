"""
model.py

Defines the neural network that recognizes isolated ASL signs from
sequences of hand landmarks, and how to save and load it.

What the model does:
    It performs CLASSIFICATION: given one short clip of hand movement, it
    chooses which one of a fixed list of signs (the classes) the clip shows.
    It recognizes ONE isolated sign at a time. It does not understand ASL
    grammar, facial expressions or continuous signing.

Why a GRU (a kind of recurrent neural network)?
    Many signs are defined by MOVEMENT, not just by a hand shape. For example
    "yes" is a fist nodding up and down. A single photo of a fist cannot tell
    "yes" from the letter "S"; you need to see how the hand moves over time.
    A static image classifier (or an MLP looking at one frame) therefore
    fails on movement-based signs.
    A GRU reads the frames one after another and keeps an internal "memory"
    (the hidden state) that summarises what it has seen so far, so it can
    learn patterns that unfold over time. GRUs are smaller and simpler than
    Transformers and train well on modest datasets, which makes them a
    sensible first choice here.

Input and output shapes:
    input  : (batch_size, sequence_length, num_features)
             e.g. (64, 32, 132) = 64 clips, 32 time steps, 132 numbers per step
             (the 132 features are described in vision/preprocessing.py).
    output : (batch_size, num_classes) raw scores called "logits".
             The highest logit is the predicted class. Applying softmax turns
             logits into probabilities that add up to 1.

Main pieces:
    ModelConfig            - the size settings needed to rebuild the network.
    SignSequenceClassifier - the PyTorch network.
    save_checkpoint() / load_checkpoint() - store and restore trained weights.

Used by: training/train.py, training/evaluate.py and models/predictor.py.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

# Increase this if the checkpoint format changes in an incompatible way.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class ModelConfig:
    """
    Everything needed to rebuild the network's structure.

    Attributes:
        input_size: Numbers per time step (132 with the default features).
        num_classes: How many signs the model chooses between.
        hidden_size: Size of the GRU's memory vector.
        num_layers: How many GRU layers are stacked.
        dropout: Fraction of values randomly zeroed during training.
        bidirectional: Whether the GRU also reads the clip backwards.
    """

    input_size: int
    num_classes: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True


class SignSequenceClassifier(nn.Module):
    """
    A GRU-based classifier for sequences of hand landmark features.

    Data flow for one clip:
        features per time step (132)
          -> LayerNorm            (put each time step's numbers on a similar scale)
          -> Linear + ReLU        (learn useful combinations of the raw numbers)
          -> GRU                  (read the time steps in order, tracking motion)
          -> average over time    (one summary vector for the whole clip)
          -> Linear               (one score per sign)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.input_size <= 0 or config.num_classes <= 0:
            raise ValueError("input_size and num_classes must be positive.")
        self.config = config

        # LayerNorm rescales the numbers within each time step to have an
        # average of 0 and a spread of 1 (then applies a learned scale and
        # shift). Neural networks train more reliably when inputs are
        # similarly sized.
        self.input_norm = nn.LayerNorm(config.input_size)

        # A Linear layer computes weighted sums of its inputs:
        #     output = input x W + b
        # W (weights) and b (bias) start random and are LEARNED in training.
        # Here it maps the 132 raw features to hidden_size learned features.
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # ReLU (Rectified Linear Unit) replaces negative numbers with 0.
        # Without a "nonlinearity" like this, stacked linear layers would
        # collapse into one linear layer and could only learn very simple
        # relationships between hand positions and signs.
        self.activation = nn.ReLU()

        # Dropout randomly sets some values to zero during TRAINING only.
        # This stops the network relying too heavily on any single feature
        # and reduces overfitting (memorizing the training data).
        self.dropout = nn.Dropout(config.dropout)

        # The GRU (Gated Recurrent Unit) processes the sequence step by step.
        # At each step it combines the current input with its memory of the
        # previous steps, using learned "gates" that decide what to keep and
        # what to forget. batch_first=True means the input shape is
        # (batch, time, features).
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            # PyTorch only applies dropout BETWEEN stacked GRU layers.
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        # A bidirectional GRU produces a forward and a backward memory vector
        # at every step, so its output is twice as wide.
        gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # The output layer produces one score (logit) per sign.
        self.output_layer = nn.Linear(gru_output_size, config.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute class scores for a batch of clips.

        Args:
            features: Tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Logits of shape (batch_size, num_classes).

        Raises:
            ValueError: if the input has the wrong number of dimensions or features.
        """
        if features.dim() != 3:
            raise ValueError(
                f"Expected input of shape (batch, time, features), got {tuple(features.shape)}."
            )
        if features.shape[2] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} features per time step, got {features.shape[2]}. "
                "Was the model trained with different preprocessing settings?"
            )
        if features.shape[1] == 0:
            raise ValueError("Input sequence has no time steps.")

        hidden = self.input_norm(features)
        hidden = self.dropout(self.activation(self.input_projection(hidden)))

        # gru_outputs has shape (batch, time, gru_output_size): one memory
        # vector per time step. We ignore the second return value (the final
        # hidden state) because we average all steps instead.
        gru_outputs, _ = self.gru(hidden)

        # Averaging over time ("mean pooling") gives every part of the sign
        # an equal say in the final decision.
        clip_summary = gru_outputs.mean(dim=1)
        return self.output_layer(self.dropout(clip_summary))

    @torch.no_grad()
    def predict_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities with softmax.

        Softmax formula: probability_i = exp(logit_i) / sum_j exp(logit_j)
            It makes every score positive and rescales them to add up to 1.
            The result is the model's CONFIDENCE in each sign. A confidence
            of 0.9 does not guarantee the answer is right - models can be
            confidently wrong, especially on people or lighting they have
            never seen.

        Returns:
            Tensor of shape (batch_size, num_classes).
        """
        return torch.softmax(self.forward(features), dim=1)


def save_checkpoint(
    path: Union[str, Path],
    model: SignSequenceClassifier,
    vocabulary_dict: Dict[str, Any],
    feature_settings: Dict[str, Any],
    training_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save the trained weights together with everything needed to use them.

    Args:
        path: Where to write the .pt file.
        model: The trained network.
        vocabulary_dict: Vocabulary.to_dict(), the label <-> index mapping.
        feature_settings: Preprocessing settings the model was trained with
            (e.g. sequence_length). Live recognition must use the same ones.
        training_summary: Optional plain-Python information about training.

    "Model weights" are all the learned numbers (W and b in every layer).
    model.state_dict() collects them into a dictionary of tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary_dict,
            "feature_settings": feature_settings,
            "training_summary": training_summary or {},
        },
        path,
    )


def load_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> Tuple[SignSequenceClassifier, Dict[str, Any]]:
    """
    Rebuild a trained model from a checkpoint file.

    Args:
        path: The .pt file written by save_checkpoint().
        device: "cpu" or "cuda" (a GPU).

    Returns:
        (model, checkpoint_dict). The model is in evaluation mode, ready for
        inference. checkpoint_dict holds the vocabulary and feature settings.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a compatible checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    # weights_only=True only allows tensors and basic Python types to be
    # loaded, so a malicious .pt file cannot run code on your computer.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required_keys = {"model_config", "model_state_dict", "vocabulary", "feature_settings"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError(f"{path} is not a checkpoint created by this project.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format {checkpoint.get('format_version')} is not supported "
            f"(expected {CHECKPOINT_FORMAT_VERSION}). Please retrain the model."
        )

    model = SignSequenceClassifier(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # eval() switches off training-only behaviour such as dropout, so the
    # same input always gives the same output.
    model.eval()
    return model, checkpoint
