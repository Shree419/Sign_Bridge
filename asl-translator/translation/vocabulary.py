"""
vocabulary.py

Maps between sign labels (like "hello") and the numbers the model uses.

Why this file exists:
    A neural network cannot output the word "hello". For classification it
    outputs one score per class, in a fixed order, so "class number 0" might
    mean "bye" and "class number 1" might mean "hello". This mapping must be
    EXACTLY the same during training and during live recognition, or every
    prediction would be shown with the wrong word. The Vocabulary is saved
    inside the model checkpoint so the two can never drift apart.

Main class:
    Vocabulary - label <-> index conversion and display names.

Used by: training/train.py, training/dataset.py, training/evaluate.py,
models/predictor.py and the app.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


class Vocabulary:
    """
    The ordered list of signs a model can recognize.

    Example:
        vocabulary = Vocabulary(["hello", "thankyou"], {"thankyou": "thank you"})
        vocabulary.label_to_index("hello")   # 0
        vocabulary.index_to_label(1)         # "thankyou"
        vocabulary.display_name("thankyou")  # "thank you"
    """

    def __init__(self, class_names: Iterable[str], display_names: Optional[Dict[str, str]] = None):
        """
        Args:
            class_names: The labels, in the order used for model outputs.
            display_names: Optional prettier text for some labels.

        Raises:
            ValueError: if the list is empty or contains duplicates/blank labels.
        """
        cleaned_names = [str(name).strip() for name in class_names]
        if not cleaned_names:
            raise ValueError("A vocabulary needs at least one sign.")
        if any(not name for name in cleaned_names):
            raise ValueError("Sign labels cannot be empty.")
        if len(set(cleaned_names)) != len(cleaned_names):
            raise ValueError(f"Sign labels must be unique, got: {cleaned_names}")
        self.class_names: List[str] = cleaned_names
        self.display_names: Dict[str, str] = dict(display_names or {})
        self._index_by_label = {name: index for index, name in enumerate(self.class_names)}

    @property
    def num_classes(self) -> int:
        """How many signs (= how many model outputs)."""
        return len(self.class_names)

    def label_to_index(self, label: str) -> int:
        """Return the output index of a label. Raises KeyError for unknown labels."""
        if label not in self._index_by_label:
            raise KeyError(f"'{label}' is not in the vocabulary {self.class_names}")
        return self._index_by_label[label]

    def index_to_label(self, index: int) -> str:
        """Return the label for a model output index. Raises IndexError if out of range."""
        if not 0 <= int(index) < self.num_classes:
            raise IndexError(f"Class index {index} is out of range for {self.num_classes} classes.")
        return self.class_names[int(index)]

    def display_name(self, label: str) -> str:
        """Return the text to show for a label (falls back to the label itself)."""
        return self.display_names.get(label, label)

    def __contains__(self, label: str) -> bool:
        return label in self._index_by_label

    def __len__(self) -> int:
        return self.num_classes

    def to_dict(self) -> Dict[str, object]:
        """Convert to plain Python types (for saving in JSON or a checkpoint)."""
        return {"class_names": list(self.class_names), "display_names": dict(self.display_names)}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        """Rebuild a Vocabulary saved with to_dict()."""
        return cls(data["class_names"], data.get("display_names") or {})

    def save_json(self, path: Union[str, Path]) -> None:
        """Write the label mapping to a human-readable JSON file."""
        mapping = self.to_dict()
        mapping["index_to_label"] = {str(index): name for index, name in enumerate(self.class_names)}
        Path(path).write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Vocabulary":
        """Read a label mapping written by save_json()."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
