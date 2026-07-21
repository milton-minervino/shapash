"""Unit tests for the model capability layer (no torch/transformers required).

Uses a tiny deterministic fake model implementing the capability mixins, so the interface and the
``has_capabilities`` discovery helper are exercised without a real transformer.
"""

import unittest

import numpy as np

from shapash.model.base import (
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
    has_capabilities,
    is_word_token,
)


class PredictOnlyModel(TextModel):
    """Minimal model implementing only ``predict``."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))


class FullModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """Model implementing every capability."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.3, 0.7], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return ["a", "b"], np.eye(2)

    def embed(self, texts):
        return np.zeros((len(texts), 2))

    def token_gradients(self, text, target_class):
        toks = text.split()
        return toks, np.ones((len(toks), 2))


class TestCapabilities(unittest.TestCase):
    def test_predict_only_lacks_optional_capabilities(self):
        m = PredictOnlyModel()
        self.assertIsInstance(m, TextModel)
        self.assertFalse(has_capabilities(m, SupportsGradients))
        self.assertFalse(has_capabilities(m, SupportsEmbeddings))
        self.assertFalse(has_capabilities(m, SupportsTokenization))

    def test_full_model_has_all_capabilities(self):
        m = FullModel()
        self.assertTrue(has_capabilities(m, SupportsTokenization))
        self.assertTrue(has_capabilities(m, SupportsEmbeddings))
        self.assertTrue(has_capabilities(m, SupportsGradients))
        self.assertTrue(has_capabilities(m, SupportsGradients, SupportsEmbeddings))

    def test_predict_shape_and_n_classes(self):
        m = FullModel()
        probs = m.predict(["x", "y", "z"])
        self.assertEqual(probs.shape, (3, 2))
        self.assertEqual(m.n_classes, 2)

    def test_n_classes_none_without_labels(self):
        m = FullModel()
        m.label_names = None
        self.assertIsNone(m.n_classes)


class _SchemeModel(TextModel, SupportsTokenization):
    """Tokenizes by whitespace, then applies a subword-marker convention to each word.

    Stands in for the three families the real adapters meet: WordPiece-style continuation marking
    (no prefix), byte-level BPE (``Ġ``) and SentencePiece (``▁``).
    """

    def __init__(self, marker=""):
        super().__init__(label_names=["neg", "pos"])
        self.marker = marker
        self.tokenize_calls = 0

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))

    def tokenize(self, text):
        self.tokenize_calls += 1
        return [f"{self.marker}{w}" for w in text.split()]

    def detokenize(self, tokens):
        return " ".join(t.removeprefix(self.marker) for t in tokens)


class TestWordStartMarker(unittest.TestCase):
    """The marker is probed from what the tokenizer emits, not from a class name or model list."""

    def test_detects_no_marker_for_continuation_scheme(self):
        self.assertIsNone(_SchemeModel().word_start_marker())

    def test_detects_byte_bpe_marker(self):
        self.assertEqual(_SchemeModel(marker="Ġ").word_start_marker(), "Ġ")

    def test_detects_sentencepiece_marker(self):
        self.assertEqual(_SchemeModel(marker="▁").word_start_marker(), "▁")

    def test_probe_runs_once_and_is_memoized(self):
        model = _SchemeModel(marker="▁")
        for _ in range(5):
            model.word_start_marker()
        self.assertEqual(model.tokenize_calls, 1)

    def test_unprobeable_tokenizer_degrades_to_no_marker(self):
        # A tokenizer that cannot run the probe must not break word-hood entirely.
        class _Broken(_SchemeModel):
            def tokenize(self, text):
                raise RuntimeError("tokenizer unavailable")

        model = _Broken()
        self.assertIsNone(model.word_start_marker())
        self.assertTrue(model.is_substitutable("happy"))


class TestIsSubstitutable(unittest.TestCase):
    """Word-hood must follow the model's own scheme — a single ``isalpha`` rule is wrong for two of three.

    ``"▁good"`` is not alphabetic and ``"Ġ"`` (U+0120) *is*, so a bare ``isalpha`` check rejects every
    SentencePiece content token and accepts every mid-word byte-BPE fragment.
    """

    def test_continuation_scheme_accepts_bare_words(self):
        m = _SchemeModel()
        self.assertTrue(m.is_substitutable("happy"))
        self.assertFalse(m.is_substitutable("##ing"))  # WordPiece continuation
        self.assertFalse(m.is_substitutable("[CLS]"))
        self.assertFalse(m.is_substitutable("1b"))
        self.assertFalse(m.is_substitutable("!"))

    def test_sentencepiece_scheme_accepts_marked_words(self):
        m = _SchemeModel(marker="▁")
        self.assertTrue(m.is_substitutable("▁good"))  # the regression: was False for every token
        self.assertFalse(m.is_substitutable("good"))  # bare -> mid-word piece under this scheme
        self.assertFalse(m.is_substitutable("▁"))  # marker alone is not a word
        self.assertFalse(m.is_substitutable("▁123"))
        self.assertFalse(m.is_substitutable("<s>"))

    def test_byte_bpe_scheme_accepts_marked_words(self):
        m = _SchemeModel(marker="Ġ")
        self.assertTrue(m.is_substitutable("Ġgood"))
        self.assertFalse(m.is_substitutable("good"))  # the other regression: was True (mid-word piece)
        self.assertFalse(m.is_substitutable("Ġ"))
        self.assertFalse(m.is_substitutable("Ċ"))

    def test_is_word_token_is_the_marker_free_default(self):
        self.assertTrue(is_word_token("happy"))
        self.assertFalse(is_word_token("##ing"))
        self.assertFalse(is_word_token("▁good"))  # why it cannot be used alone


if __name__ == "__main__":
    unittest.main()
