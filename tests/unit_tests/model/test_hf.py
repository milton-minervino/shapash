"""Unit tests for ``HFPipelineModel``'s label-order resolution.

Uses a fake ``pipeline`` callable (plain function, no ``transformers``/``torch``) so these stay fast
unit tests: only the label-name/column-order contract of ``predict`` is under test here.
"""

import unittest

from shapash.model.hf import HFPipelineModel


class FakePipeline:
    """Mimics a HuggingFace ``text-classification`` pipeline with ``top_k=None``.

    Returns, per text, a list of ``{"label", "score"}`` dicts in a fixed (non-alphabetical,
    non-id) order to make sure column alignment is name-based, not position-based.
    """

    def __init__(self, rows: list[list[dict]]) -> None:
        self.rows = rows
        self.tokenizer = None

    def __call__(self, texts: list[str], top_k=None):
        assert len(texts) <= len(self.rows)
        return self.rows[: len(texts)]


def _rows(*label_score_pairs_per_row):
    return [[{"label": label, "score": score} for label, score in row] for row in label_score_pairs_per_row]


class TestHFPipelineModelLabelOrder(unittest.TestCase):
    def test_infers_order_and_label_names_from_pipeline_when_none_given(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline)

        probs = model.predict(["great movie"])

        self.assertEqual(model.label_names, ["negative", "positive"])
        self.assertEqual(list(probs[0]), [0.1, 0.9])

    def test_provided_label_names_become_the_column_order(self):
        # Pipeline's own row order is "negative" first; the caller asserts the opposite order.
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["positive", "negative"])

        probs = model.predict(["great movie"])

        self.assertEqual(model.label_names, ["positive", "negative"])
        self.assertEqual(list(probs[0]), [0.9, 0.1])

    def test_order_and_label_names_stay_in_sync_after_predict(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["positive", "negative"])

        model.predict(["great movie"])

        self.assertEqual(model._order, model.label_names)

    def test_mismatched_label_names_raise_instead_of_silently_diverging(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["negative", "possitive"])  # typo

        with self.assertRaises(ValueError):
            model.predict(["great movie"])

    def test_label_names_proper_subset_raises(self):
        # Pipeline emits 3 classes; caller only declares 2 — must not silently drop the third.
        pipeline = FakePipeline(_rows([("negative", 0.1), ("neutral", 0.2), ("positive", 0.7)]))
        model = HFPipelineModel(pipeline, label_names=["negative", "positive"])

        with self.assertRaises(ValueError):
            model.predict(["great movie"])

    def test_label_names_valid_regardless_of_first_row_score_sort_order(self):
        # Different rows may come back sorted by score (per-row), not by a stable id order; the
        # caller's declared order must still win via name-based lookup.
        pipeline = FakePipeline(
            _rows(
                [("positive", 0.9), ("negative", 0.1)],
                [("negative", 0.8), ("positive", 0.2)],
            )
        )
        model = HFPipelineModel(pipeline, label_names=["negative", "positive"])

        probs = model.predict(["good", "bad"])

        self.assertEqual(list(probs[0]), [0.1, 0.9])
        self.assertEqual(list(probs[1]), [0.8, 0.2])


if __name__ == "__main__":
    unittest.main()
