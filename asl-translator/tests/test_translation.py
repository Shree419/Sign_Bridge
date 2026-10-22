"""Tests for the translation package: vocabulary, sentence builder and formatting."""

import pytest

from translation.postprocessing import FormattingOnlyTranslator, format_gloss, format_readable_text
from translation.sentence_builder import SentenceBuilder
from translation.vocabulary import Vocabulary


def test_vocabulary_round_trip(tmp_path):
    vocabulary = Vocabulary(["hello", "thankyou"], {"thankyou": "thank you"})
    assert vocabulary.label_to_index("thankyou") == 1
    assert vocabulary.index_to_label(0) == "hello"
    assert vocabulary.display_name("thankyou") == "thank you"
    assert vocabulary.display_name("hello") == "hello"
    path = tmp_path / "labels.json"
    vocabulary.save_json(path)
    loaded = Vocabulary.load_json(path)
    assert loaded.class_names == vocabulary.class_names
    assert loaded.display_names == vocabulary.display_names


def test_vocabulary_rejects_bad_input():
    with pytest.raises(ValueError):
        Vocabulary([])
    with pytest.raises(ValueError):
        Vocabulary(["a", "a"])
    with pytest.raises(KeyError):
        Vocabulary(["a"]).label_to_index("b")
    with pytest.raises(IndexError):
        Vocabulary(["a"]).index_to_label(3)


def test_builder_add_undo_clear():
    builder = SentenceBuilder(Vocabulary(["hello", "thankyou"], {"thankyou": "thank you"}))
    builder.add_sign("hello", 0.9)
    builder.add_sign("thankyou", 0.8)
    assert builder.gloss_text() == "hello thank you"
    assert builder.original_labels() == ["hello", "thankyou"]
    assert builder.undo_last().label == "thankyou"
    assert builder.gloss_text() == "hello"
    builder.clear()
    assert builder.gloss_text() == ""
    assert builder.undo_last() is None


def test_editing_keeps_unchanged_words_and_marks_typed_ones():
    builder = SentenceBuilder(Vocabulary(["hello", "thankyou", "water"], {"thankyou": "thank you"}))
    for label in ("hello", "thankyou", "water"):
        builder.add_sign(label, 0.9)
    builder.replace_with_text("hello thank you milk please")
    assert builder.gloss_text() == "hello thank you milk please"
    assert builder.original_labels() == ["hello", "thankyou", None, None]
    assert [word.edited for word in builder.words] == [False, False, True, True]


def test_prediction_log_is_kept_separately():
    builder = SentenceBuilder()
    builder.log_prediction("hello", 0.42, "uncertain")
    builder.clear()
    assert len(builder.prediction_log) == 1


def test_formatting_never_changes_the_words():
    words = ["water", "drink", "i"]
    assert format_gloss(words) == "WATER DRINK I"
    assert format_readable_text(words) == "Water drink I"
    assert format_readable_text(words, add_final_period=True) == "Water drink I."
    assert format_readable_text([]) == ""
    translator = FormattingOnlyTranslator()
    assert translator.is_real_translation is False
    assert translator.translate(["hello"]) == "Hello"
