"""
train.py

Trains the sign recognition model on the labelled samples in data/samples.

How a neural network learns (the short version):
    1. FORWARD PASS: the model looks at a batch of training samples and
       produces a score for every sign.
    2. LOSS: a loss function measures how wrong those scores are compared
       with the correct labels. Lower loss = better predictions.
    3. BACKPROPAGATION: PyTorch works out, for every weight in the network,
       whether nudging it up or down would reduce the loss (the "gradient").
    4. OPTIMIZER STEP: the optimizer nudges every weight a small amount in
       the helpful direction.
    Repeating this over many batches and EPOCHS (an epoch = one full pass
    through the training data) gradually improves the model.

Overfitting and early stopping:
    A model can get better and better on training data simply by memorizing
    it, while getting WORSE on new data. This is overfitting. We watch the
    loss on the validation set (data the model never trains on) after every
    epoch, keep the weights from the epoch with the lowest validation loss,
    and stop once it has not improved for `early_stopping_patience` epochs.

Steps this script performs:
    1. Load the sample index and check the data is valid.
    2. Split into training / validation / test sets (by signer when possible).
    3. Convert samples into model inputs (SignSequenceDataset).
    4. Train with early stopping, recording training and validation loss.
    5. Save the best model, the label mapping and the split assignment.
    6. Plot the training curves and evaluate on the test set.

Run:
    python -m training.train
    python -m training.train --epochs 5      (quick experiment)

Outputs (in models/checkpoints/ and reports/):
    best_model.pt, labels.json, splits.csv, training_history.csv,
    training_curves.png, plus the evaluation reports from evaluate.py.
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # draw graphs to files without needing a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.model import ModelConfig, SignSequenceClassifier, save_checkpoint
from training.dataset import SampleStore, SignSequenceDataset, split_by_signer, validate_index
from translation.vocabulary import Vocabulary
from utils.config import load_config, resolve_path
from vision.preprocessing import FEATURES_PER_FRAME

CHECKPOINT_FILENAME = "best_model.pt"
LABELS_FILENAME = "labels.json"
SPLITS_FILENAME = "splits.csv"
HISTORY_FILENAME = "training_history.csv"


class TrainingDataError(RuntimeError):
    """Raised when the data is not suitable for training (with an explanation)."""


def set_random_seed(seed: int) -> None:
    """Make training repeatable: the same seed gives the same random choices."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    """Use an NVIDIA GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    """
    Pass once over a dataset. If an optimizer is given, the model also learns.

    Args:
        model: The network.
        data_loader: Gives batches of (features, labels).
        loss_function: Measures prediction error.
        device: CPU or GPU.
        optimizer: Pass it for TRAINING; leave as None for VALIDATION/TEST.

    Returns:
        (average_loss, accuracy) over the whole dataset.
    """
    is_training = optimizer is not None
    # train() turns dropout ON, eval() turns it OFF.
    model.train(is_training)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Gradients are only needed when learning; skipping them saves time and memory.
    with torch.set_grad_enabled(is_training):
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass: compute a score for every sign.
            logits = model(features)
            loss = loss_function(logits, labels)

            if is_training:
                # Gradients add up by default, so clear the previous batch's.
                optimizer.zero_grad()
                # Backpropagation: compute how each weight affected the loss.
                loss.backward()
                # Gradient clipping limits very large updates that can make
                # recurrent networks like GRUs unstable.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Optimizer step: adjust the weights to reduce the loss.
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss / total_samples, correct_predictions / total_samples


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """
    Save a graph of loss and accuracy per epoch.

    How to read it: if training loss keeps falling while validation loss
    rises, the model is overfitting.
    """
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history["epoch"], history["train_loss"], label="training")
    loss_axis.plot(history["epoch"], history["validation_loss"], label="validation")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss (lower is better)")
    loss_axis.set_title("Loss")
    loss_axis.legend()
    accuracy_axis.plot(history["epoch"], history["train_accuracy"], label="training")
    accuracy_axis.plot(history["epoch"], history["validation_accuracy"], label="validation")
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def load_and_split_data(config: Dict, store: SampleStore, vocabulary: Vocabulary) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Steps 1-2: load the index, validate it and split it.

    Raises:
        TrainingDataError: with an explanation if there is not enough data.
    """
    index = store.load_index()
    if index.empty:
        raise TrainingDataError(
            f"No samples found in {store.samples_dir}. Run `python -m training.prepare_islr` "
            "or record samples with `python -m data_collection.collect_samples` first."
        )
    clean_index, report = validate_index(index, store.samples_dir, vocabulary)
    print(report.summary())

    counts = clean_index["label"].value_counts()
    minimum = int(config["training"]["min_samples_per_class"])
    too_few = {label: int(counts.get(label, 0)) for label in vocabulary.class_names if counts.get(label, 0) < minimum}
    if too_few:
        raise TrainingDataError(
            f"These signs have fewer than {minimum} samples: {too_few}. Record or import more "
            "examples, or remove those signs from vocabulary.signs in config.yaml."
        )

    training_config = config["training"]
    return split_by_signer(
        clean_index,
        float(training_config["validation_fraction"]),
        float(training_config["test_fraction"]),
        int(training_config["random_seed"]),
    )


def train(config: Dict, epochs_override: Optional[int] = None) -> Path:
    """
    Train a model using the settings in config and save the best checkpoint.

    Args:
        config: The loaded config.yaml.
        epochs_override: Optional number of epochs (overrides the config).

    Returns:
        The path of the saved checkpoint.
    """
    training_config = config["training"]
    feature_config = config["features"]
    set_random_seed(int(training_config["random_seed"]))
    device = choose_device()

    vocabulary = Vocabulary(config["vocabulary"]["signs"], config["vocabulary"].get("display_names"))
    store = SampleStore(resolve_path(config["paths"]["samples_dir"]))
    checkpoint_dir = resolve_path(config["paths"]["checkpoint_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- Steps 1-2: load, validate and split -----------------------------
    splits, split_method = load_and_split_data(config, store, vocabulary)
    print(f"Split method: {split_method}")
    for split_name, split_rows in splits.items():
        print(f"  {split_name:<10} {len(split_rows):>6} samples, {split_rows['signer_id'].nunique()} signer(s)")
    # Saving which sample went where lets evaluate.py use exactly the same
    # test set later, and lets you check that no sample is in two sets.
    pd.concat([rows.assign(split=name) for name, rows in splits.items()]).to_csv(
        checkpoint_dir / SPLITS_FILENAME, index=False
    )

    # ---- Step 3: build datasets -------------------------------------------
    sequence_length = int(feature_config["sequence_length"])
    min_frames = int(feature_config["min_frames_with_hands"])
    seed = int(training_config["random_seed"])
    datasets = {
        name: SignSequenceDataset(
            rows,
            store,
            vocabulary,
            sequence_length,
            min_frames,
            # Augmentation is applied ONLY to the training set.
            augment=bool(training_config["augment"]) and name == "train",
            random_seed=seed,
        )
        for name, rows in splits.items()
    }
    for name, dataset in datasets.items():
        if dataset.skipped_samples:
            print(f"  {name}: skipped {len(dataset.skipped_samples)} sample(s) with too little hand data")
    if len(datasets["train"]) == 0 or len(datasets["validation"]) == 0:
        raise TrainingDataError("The training or validation set is empty after removing unusable samples.")

    batch_size = int(training_config["batch_size"])
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=batch_size)

    # ---- Step 4: model, loss function and optimizer -----------------------
    model_config = ModelConfig(
        input_size=FEATURES_PER_FRAME,
        num_classes=vocabulary.num_classes,
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    )
    model = SignSequenceClassifier(model_config).to(device)
    num_weights = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model has {num_weights:,} trainable weights. Training on {device}.")

    # Cross-entropy loss is the standard loss for classification. For each
    # sample it equals -log(probability the model gave the CORRECT sign):
    # near 0 when the model is confidently right, large when it is
    # confidently wrong. Optional class weights make mistakes on rare signs
    # count more, so the model does not just favour the most common signs.
    class_weights = None
    if training_config.get("use_class_weights", False):
        counts = datasets["train"].class_counts().astype(np.float64)
        if (counts == 0).any():
            missing = [vocabulary.index_to_label(i) for i in np.flatnonzero(counts == 0)]
            raise TrainingDataError(f"No usable training samples for: {missing}")
        class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW is a popular optimizer. It adapts the step size for each weight
    # individually based on recent gradients, which usually trains faster
    # and more reliably than plain gradient descent. weight_decay gently
    # shrinks weights toward zero to discourage overfitting.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    # Halve the learning rate if validation loss stops improving for 3 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # ---- Training loop with early stopping --------------------------------
    num_epochs = int(epochs_override or training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_one_epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss, validation_accuracy = run_one_epoch(model, validation_loader, loss_function, device)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:>3}/{num_epochs}  train loss {train_loss:.4f} acc {train_accuracy:.3f}  |  "
            f"validation loss {validation_loss:.4f} acc {validation_accuracy:.3f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            # deepcopy so later training does not change the saved weights.
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no validation improvement for {patience} epochs.")
                break

    print(f"Training took {time.time() - start_time:.0f} s. Best epoch: {best_epoch}.")

    # ---- Step 5: save the best model and label mapping --------------------
    model.load_state_dict(best_state)
    history_table = pd.DataFrame(history)
    history_table.to_csv(checkpoint_dir / HISTORY_FILENAME, index=False)
    best_row = history_table.loc[history_table["epoch"] == best_epoch].iloc[0]
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    save_checkpoint(
        checkpoint_path,
        model,
        vocabulary.to_dict(),
        feature_settings={"sequence_length": sequence_length, "min_frames_with_hands": min_frames},
        training_summary={
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_row["validation_loss"]),
            "best_validation_accuracy": float(best_row["validation_accuracy"]),
            "split_method": split_method,
            "num_training_samples": len(datasets["train"]),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    vocabulary.save_json(checkpoint_dir / LABELS_FILENAME)
    print(f"Saved model to {checkpoint_path}")

    # ---- Step 6: graphs and test-set evaluation ---------------------------
    plot_training_history(history_table, reports_dir / "training_curves.png")
    print(f"Saved training curves to {reports_dir / 'training_curves.png'}")

    if len(datasets["test"]) > 0:
        # Imported here to avoid a circular import at module load time.
        from training.evaluate import evaluate_checkpoint

        evaluate_checkpoint(config, checkpoint_path)
    return checkpoint_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the ASL sign recognition model.")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config.yaml path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        train(config, epochs_override=args.epochs)
    except TrainingDataError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
