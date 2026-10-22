"""
sentence_builder.py

Collects accepted signs into a growing line of text that the user can
undo, clear, edit and copy.

IMPORTANT - what this output is and is not:
    The model recognizes isolated signs, one at a time. Joining them gives a
    GLOSS: a word-for-word record of which signs were recognized, in order,
    e.g. "WATER DRINK WHERE". ASL has its own grammar (word order, facial
    grammar, use of space, classifiers...), so a gloss is NOT a grammatical
    English translation. Turning a gloss into fluent English would need a
    separate ASL-to-English translation model, which this project does not
    include. The class below is designed so such a model could be plugged in
    later (see translation/postprocessing.py) without changing this file.

Main classes:
    RecognizedSign  - one accepted sign plus the information needed to debug it.
    SentenceBuilder - the editable word list and a log of raw predictions.

Used by: app.py.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from translation.vocabulary import Vocabulary


@dataclass
class RecognizedSign:
    """
    One word in the output.

    Attributes:
        text: What is shown for this word (may have been edited by the user).
        label: The model's original label, or None for words the user typed.
        confidence: The smoothed confidence when it was accepted (None if typed).
        timestamp: When it was added (seconds since 1970).
        edited: True if the user changed or typed this word.
    """

    text: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    edited: bool = False


class SentenceBuilder:
    """
    Keeps the list of recognized signs and a debugging log of raw predictions.
    """

    def __init__(self, vocabulary: Optional[Vocabulary] = None, max_prediction_log: int = 200):
        """
        Args:
            vocabulary: Used to turn labels into display text (e.g.
                "thankyou" -> "thank you"). Optional.
            max_prediction_log: How many raw predictions to remember for debugging.
        """
        self.vocabulary = vocabulary
        self.words: List[RecognizedSign] = []
        # Raw model predictions are kept separately from the words, so you
        # can always see what the model actually said even after editing.
        self.prediction_log: Deque[Dict[str, object]] = deque(maxlen=max_prediction_log)

    def add_sign(self, label: str, confidence: Optional[float] = None) -> RecognizedSign:
        """
        Append an accepted sign.

        Args:
            label: The model's label for the sign.
            confidence: Its smoothed confidence (0-1).

        Returns:
            The RecognizedSign that was added.
        """
        display_text = self.vocabulary.display_name(label) if self.vocabulary else label
        word = RecognizedSign(text=display_text, label=label, confidence=confidence)
        self.words.append(word)
        return word

    def log_prediction(self, label: Optional[str], confidence: float, status: str) -> None:
        """Record a raw (not necessarily accepted) prediction for debugging."""
        self.prediction_log.append(
            {"time": time.time(), "label": label, "confidence": round(float(confidence), 4), "status": status}
        )

    def undo_last(self) -> Optional[RecognizedSign]:
        """Remove and return the last word, or None if there are no words."""
        return self.words.pop() if self.words else None

    def clear(self) -> None:
        """Remove all words (the raw prediction log is kept)."""
        self.words.clear()

    def replace_with_text(self, edited_text: str) -> None:
        """
        Replace the word list with text the user has edited by hand.

        The unchanged words at the start keep their original label and
        confidence. From the first difference onwards, every word is marked
        as edited (with no model label), so the output never pretends the
        model recognized a word that the user typed.
        """
        new_tokens = edited_text.split()
        kept_words: List[RecognizedSign] = []
        position = 0
        for word in self.words:
            # A display name can be several tokens long ("thank you").
            word_tokens = word.text.split()
            if new_tokens[position : position + len(word_tokens)] != word_tokens:
                break
            kept_words.append(word)
            position += len(word_tokens)
        typed_words = [RecognizedSign(text=token, edited=True) for token in new_tokens[position:]]
        self.words = kept_words + typed_words

    def gloss_text(self) -> str:
        """Return the words joined with spaces, e.g. 'hello thank you'."""
        return " ".join(word.text for word in self.words)

    def original_labels(self) -> List[Optional[str]]:
        """Return the model labels behind each word (None for typed words)."""
        return [word.label for word in self.words]

    def __len__(self) -> int:
        return len(self.words)
