"""Unit tests for ``NlpShapBackend`` — subword aggregation and explainer wiring.

``_aggregate_subwords`` is pure numpy (no live shap.Explainer needed). ``run_explainer`` is exercised
with a fake explainer injected via ``explainer_args={"explainer": ...}`` so the test stays offline.
"""

import unittest

import numpy as np

from shapash.backend import NlpShapBackend, get_backend_cls_from_name
from shapash.backend.nlp_shap_backend import _aggregate_subwords


class TestNlpShapBackend(unittest.TestCase):
    def test_registered_by_name(self):
        self.assertIs(get_backend_cls_from_name("nlp_shap"), NlpShapBackend)

    def test_backend_name_attribute(self):
        self.assertEqual(NlpShapBackend.name, "nlp_shap")


class TestAggregateSubwords(unittest.TestCase):
    """``_aggregate_subwords`` merges SHAP's whitespace-delimited subword fragments into words."""

    def test_merges_subwords_and_drops_specials(self):
        # SHAP's Text masker: a subword glued to the previous piece carries no trailing space;
        # the piece that ends the word does (see shap.maskers.Text.token_segments).
        tokens = ["", "i ", "am ", "hap", "py ", ""]
        contribs = np.array(
            [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0], [6.0, 5.0]], dtype=float
        )
        base = np.array([10.0, 20.0])

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        self.assertEqual(words, ["i", "am", "happy"])
        # "happy" = "hap" + "py ": contributions summed.
        np.testing.assert_allclose(word_contribs, [[2.0, 1.0], [3.0, 2.0], [9.0, 7.0]])
        # Leading/trailing blank (CLS/SEP-equivalent) attribution folded into the baseline.
        np.testing.assert_allclose(new_base, [10.0 + 1.0 + 6.0, 20.0 + 0.0 + 5.0])

    def test_merges_punctuation_glued_contraction(self):
        # "don't" is tokenized as "don" + "'" + "t " — none carry inter-token whitespace.
        tokens = ["", "i ", "don", "'", "t ", "feel ", "sad", ""]
        contribs = np.arange(8 * 2, dtype=float).reshape(8, 2)
        base = np.zeros(2)

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        self.assertEqual(words, ["i", "don't", "feel", "sad"])
        np.testing.assert_allclose(word_contribs[1], contribs[2] + contribs[3] + contribs[4])

    def test_preserves_completeness(self):
        tokens = ["", "great ", "mov", "ie ", ""]
        contribs = np.random.default_rng(0).normal(size=(5, 3))
        base = np.array([0.5, -0.5, 1.0])

        _, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        # base + Σ over words must equal the original base + Σ over every token (nothing lost).
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), base + contribs.sum(axis=0))

    def test_all_special_returns_empty_words(self):
        tokens = ["", ""]
        contribs = np.array([[1.0, 2.0], [3.0, 4.0]])
        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(2))
        self.assertEqual(words, [])
        self.assertEqual(word_contribs.shape, (0, 2))
        np.testing.assert_allclose(new_base, [4.0, 6.0])

    def test_last_token_without_trailing_space_is_flushed(self):
        # The final real word before the closing special token often has no trailing space.
        tokens = ["", "hello ", "world", ""]
        contribs = np.array([[0.0], [1.0], [2.0], [0.0]])
        words, word_contribs, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
        self.assertEqual(words, ["hello", "world"])
        np.testing.assert_allclose(word_contribs, [[1.0], [2.0]])


class FakeShapExplanation:
    def __init__(self, data, values, base_values):
        self.data = data
        self.values = values
        self.base_values = base_values


class FakeShapExplainer:
    def __call__(self, x, **kwargs):
        return FakeShapExplanation(
            data=[["", "up", "dating ", ""]],
            values=[np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])],
            base_values=np.array([[0.1, 0.2]]),
        )


class TestRunExplainer(unittest.TestCase):
    def test_aggregates_subwords_end_to_end(self):
        backend = NlpShapBackend(
            model=lambda texts: np.zeros((len(texts), 2)),
            label_names=["neg", "pos"],
            explainer_args={"explainer": FakeShapExplainer},
        )

        raw = backend.run_explainer(["updating"])

        self.assertEqual(raw.data, [["updating"]])
        np.testing.assert_allclose(raw.contributions[0], [[5.0, 3.0]])
        # CLS ([1,0]) + SEP ([4,3]) folded into the base.
        np.testing.assert_allclose(raw.base_values, [[0.1 + 1.0 + 4.0, 0.2 + 0.0 + 3.0]])


if __name__ == "__main__":
    unittest.main()
