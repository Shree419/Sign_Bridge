"""
postprocessing.py

Optional, conservative cleanup of the recognized-sign text, plus the place
where a future ASL-to-English translation model would plug in.

Why this file exists:
    It keeps "how text is presented" separate from "which signs were
    recognized". The rules here only change FORMATTING (capitals, spacing,
    punctuation). They never add, remove, reorder or substitute words,
    because doing that would silently change what the signer said.

Main pieces:
    format_gloss()           - ASL-gloss style: uppercase words.
    format_readable_text()   - sentence-case formatting for easier reading.
    GlossTranslator          - the interface a future translation model must follow.
    FormattingOnlyTranslator - the current implementation: formatting only,
                               no real translation.

Used by: app.py.
"""

import re
from typing import List, Protocol


def _normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace into single spaces and trim the ends."""
    return re.sub(r"\s+", " ", text).strip()


def format_gloss(words: List[str]) -> str:
    """
    Format words in the common ASL gloss convention (uppercase).

    Example: ["water", "drink"] -> "WATER DRINK"
    """
    return _normalize_spaces(" ".join(words)).upper()


def format_readable_text(words: List[str], add_final_period: bool = False) -> str:
    """
    Make the word sequence easier to read without changing its words.

    Capitalizes the first letter (and the word "i"), optionally adds a final
    period. Example: ["hello", "thank you"] -> "Hello thank you"

    This is still a sequence of recognized signs, not a grammatical
    English translation.
    """
    text = _normalize_spaces(" ".join(words))
    if not text:
        return ""
    text = re.sub(r"\bi\b", "I", text)
    text = text[0].upper() + text[1:]
    if add_final_period and text[-1] not in ".!?":
        text += "."
    return text


class GlossTranslator(Protocol):
    """
    The interface a future ASL-gloss-to-English model should implement.

    A real translator (for example a sequence-to-sequence language model
    trained on ASL gloss / English pairs) would take the recognized glosses
    and return an English sentence. Any such model must be shown to the user
    ALONGSIDE the original gloss, never instead of it, so mistakes and
    invented words can be spotted.
    """

    def translate(self, glosses: List[str]) -> str:
        ...


class FormattingOnlyTranslator:
    """
    Placeholder "translator" used today: it ONLY formats the glosses.

    It exists so the app already has a translation step in the right place,
    while being honest that no translation is performed.
    """

    is_real_translation = False

    def translate(self, glosses: List[str]) -> str:
        return format_readable_text(glosses)
