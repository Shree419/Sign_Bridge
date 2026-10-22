"""
evaluate.py

Measures how well a trained model performs on the TEST set - samples that
were not used for training or for choosing when to stop training.

Metrics reported (explained in more detail in README.md):
    Accuracy  - fraction of test samples whose predicted sign is correct.
    Precision - for one sign: of all the times the model SAID "hello",
                what fraction really were "hello"? (Low = false alarms.)
    Recall    - for one sign: of all the real "hello" samples, what
                fraction did the model find? (Low = missed signs.)
    F1 score  - a single number combining precision and recall
                (their harmonic mean); it is only high when both are high.
    "Macro" averages treat every sign equally; "weighted" averages give
    common signs more influence.
    Confusion matrix - a table showing, for each real sign (row), how often
                it was predicted as each sign (column). Off-diagonal cells
                reveal which signs get mixed up.
    Inference latency - how long one prediction (preprocessing + model)
                takes, which decides whether real-time use is possible.

All numbers are computed from the real test data each time you run it.

Run:
    python -m training.evaluate
    python -m training.evaluate --checkpoint models/checkpoints/best_model.pt

Outputs (in reports/):
    evaluation_report.json, per_class_metrics.csv, confusion_matrix.png
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from models.predictor import ModelNotAvailableError, SignPredictor
from training.dataset import SampleStore, SignSequenceDataset
from utils.config import load_config, resolve_path

SPLITS_FILENAME = "splits.csv"
LATENCY_SAMPLES = 200


def plot_confusion_matrix(matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
    """
    Save the confusion matrix as a colour-coded image.

    Each row is normalized to add up to 1, so a cell shows "what fraction of
    the real <row sign> samples were predicted as <column sign>". A perfect
    model has 1.0 on the diagonal and 0 everywhere else.
    """
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0)
    size = max(6, 0.55 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=60, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("predicted sign")
    axis.set_ylabel("true sign")
    axis.set_title("Confusion matrix (each row sums to 1)")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            if matrix[row, column]:
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if normalized[row, column] > 0.5 else "black",
                )
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def measure_latency(predictor: SignPredictor, store: SampleStore, test_rows: pd.DataFrame) -> Dict[str, float]:
    """
    Time single-sample predictions the way the live app makes them.

    Each timing includes preprocessing (landmarks -> features) and the
    model's forward pass, for one sample at a time on the predictor's device.
    It does not include the camera or MediaPipe, which add their own time.
    """
    rows = test_rows.head(LATENCY_SAMPLES)
    landmark_sequences = [store.load_landmarks(path) for path in rows["file_path"]]
    if landmark_sequences:
        predictor.predict(landmark_sequences[0])  # warm-up run, not timed
    timings_ms = []
    for sequence in landmark_sequences:
        start = time.perf_counter()
        predictor.predict(sequence)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    if not timings_ms:
        return {}
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "num_timed_samples": len(timings_ms),
        "device": str(predictor.device),
    }


def evaluate_checkpoint(config: Dict, checkpoint_path: Path) -> Dict:
    """
    Evaluate a checkpoint on the test split saved by training.

    Args:
        config: The loaded config.yaml.
        checkpoint_path: The model to evaluate.

    Returns:
        The report dictionary (also written to reports/evaluation_report.json).
    """
    predictor = SignPredictor.from_checkpoint(checkpoint_path)
    vocabulary = predictor.vocabulary
    splits_path = Path(checkpoint_path).parent / SPLITS_FILENAME
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} not found. It is created by training/train.py.")

    split_table = pd.read_csv(splits_path, dtype={"sample_id": str, "label": str, "signer_id": str})
    test_rows = split_table[split_table["split"] == "test"].reset_index(drop=True)
    if test_rows.empty:
        raise ValueError("The saved split contains no test samples.")

    store = SampleStore(resolve_path(config["paths"]["samples_dir"]))
    test_dataset = SignSequenceDataset(
        test_rows,
        store,
        vocabulary,
        predictor.sequence_length,
        predictor.min_frames_with_hands,
        augment=False,
    )
    if len(test_dataset) == 0:
        raise ValueError("No usable test samples (all had too little hand data or missing files).")

    # Predict the whole test set in batches.
    features = np.stack([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    true_indices = np.array(test_dataset.label_indices)
    probabilities = np.concatenate(
        [predictor.predict_features(features[start : start + 256]) for start in range(0, len(features), 256)]
    )
    predicted_indices = probabilities.argmax(axis=1)

    class_indices = list(range(vocabulary.num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_indices, predicted_indices, labels=class_indices, zero_division=0
    )
    macro = precision_recall_fscore_support(
        true_indices, predicted_indices, labels=class_indices, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        true_indices, predicted_indices, labels=class_indices, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(true_indices, predicted_indices, labels=class_indices)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    per_class = pd.DataFrame(
        {
            "sign": vocabulary.class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_test_samples": support,
        }
    )
    per_class.to_csv(reports_dir / "per_class_metrics.csv", index=False)
    plot_confusion_matrix(matrix, vocabulary.class_names, reports_dir / "confusion_matrix.png")

    report = {
        "checkpoint": str(checkpoint_path),
        "num_test_samples": int(len(test_dataset)),
        "num_test_signers": int(test_rows["signer_id"].nunique()),
        "skipped_test_samples": len(test_dataset.skipped_samples),
        "accuracy": float(accuracy_score(true_indices, predicted_indices)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "latency": measure_latency(predictor, store, test_rows),
    }
    (reports_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Test-set evaluation ===")
    print(f"Test samples: {report['num_test_samples']} from {report['num_test_signers']} signer(s)")
    print(f"Accuracy:      {report['accuracy']:.3f}")
    print(f"Macro F1:      {report['macro_f1']:.3f}   (precision {report['macro_precision']:.3f}, "
          f"recall {report['macro_recall']:.3f})")
    print(f"Weighted F1:   {report['weighted_f1']:.3f}")
    if report["latency"]:
        print(f"Latency:       {report['latency']['mean_ms']:.2f} ms mean, "
              f"{report['latency']['p95_ms']:.2f} ms 95th percentile ({report['latency']['device']})")
    print("\nPer-sign results:")
    print(per_class.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nReports written to {reports_dir}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    config = load_config()
    default_checkpoint = resolve_path(config["paths"]["checkpoint_dir"]) / "best_model.pt"
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    args = parser.parse_args(argv)
    try:
        evaluate_checkpoint(config, args.checkpoint)
    except (ModelNotAvailableError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
