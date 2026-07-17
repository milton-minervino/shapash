"""Unit tests for the similar-example retriever (no torch/transformers required).

A tiny fake ``SupportsActivations`` model returns fixed activation vectors per text, so the retrieval
math (cosine, top-k ordering), label plumbing, corpus/layer validation and the ``.npy`` bank cache are
exercised deterministically without a real transformer.
"""

import unittest

import numpy as np

from shapash.compute.retrieval.similar_examples import (
    Neighbor,
    SimilarExampleRetriever,
    _l2_normalize,
)
from shapash.model.base import SupportsActivations, TextModel, has_capabilities

# Fixed 2-D activations: each text maps to a point; cosine ranks by angle from the origin.
_VECTORS = {
    "happy joy": [1.0, 0.0],
    "joyful glad": [0.9, 0.1],
    "sad down": [0.0, 1.0],
    "miserable": [0.1, 0.9],
    "neutral": [1.0, 1.0],
}


class FakeActivationModel(TextModel, SupportsActivations):
    """Model that returns a fixed activation vector per known text (unknown texts -> zeros)."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])
        self.calls = 0

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))

    @property
    def default_activation_layer(self):
        return "fake_layer"

    def activations(self, texts, layer=None):
        self.calls += 1
        return np.array([_VECTORS.get(t, [0.0, 0.0]) for t in texts], dtype=float)


class PredictOnlyModel(TextModel):
    """No activation capability."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.model = FakeActivationModel()
        self.texts = ["happy joy", "joyful glad", "sad down", "miserable"]
        self.labels = ["pos", "pos", "neg", "neg"]
        self.retriever = SimilarExampleRetriever(self.model, self.texts, self.labels)

    def test_defaults_layer_from_model(self):
        self.assertEqual(self.retriever.layer, "fake_layer")
        self.assertEqual(self.retriever.size, 4)

    def test_query_ranks_by_cosine_similarity(self):
        neighbors = self.retriever.query("happy joy", top_k=2)
        self.assertEqual([n.text for n in neighbors], ["happy joy", "joyful glad"])
        # Cosine to itself is 1.0; scores are sorted descending.
        self.assertAlmostEqual(neighbors[0].score, 1.0, places=6)
        self.assertGreaterEqual(neighbors[0].score, neighbors[1].score)

    def test_query_returns_labels_and_neighbor_type(self):
        neighbors = self.retriever.query("miserable", top_k=1)
        self.assertIsInstance(neighbors[0], Neighbor)
        self.assertEqual(neighbors[0].text, "miserable")
        self.assertEqual(neighbors[0].label, "neg")

    def test_top_k_clamped_to_corpus_size(self):
        neighbors = self.retriever.query("neutral", top_k=99)
        self.assertEqual(len(neighbors), 4)

    def test_bank_built_once_and_reused(self):
        self.retriever.query("happy joy", top_k=1)
        calls_after_first = self.model.calls
        self.retriever.query("sad down", top_k=1)
        # Two queries => one bank build + two query activations = 3 calls total (bank not rebuilt).
        self.assertEqual(self.model.calls, calls_after_first + 1)

    def test_labels_optional(self):
        retriever = SimilarExampleRetriever(self.model, self.texts, reference_labels=None)
        neighbors = retriever.query("happy joy", top_k=1)
        self.assertIsNone(neighbors[0].label)

    def test_mismatched_labels_length_raises(self):
        with self.assertRaises(ValueError):
            SimilarExampleRetriever(self.model, self.texts, reference_labels=["pos"])

    def test_incompatible_model_raises(self):
        self.assertFalse(has_capabilities(PredictOnlyModel(), SupportsActivations))
        with self.assertRaises(TypeError):
            SimilarExampleRetriever(PredictOnlyModel(), self.texts, self.labels)

    def test_bank_cached_to_disk_and_reloaded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            r1 = SimilarExampleRetriever(self.model, self.texts, self.labels, cache_dir=d)
            r1.build()
            cache_file = r1._cache_file()
            self.assertTrue(cache_file.exists())

            # A fresh retriever over the same corpus/layer loads the bank from disk (no activations call).
            fresh_model = FakeActivationModel()
            r2 = SimilarExampleRetriever(fresh_model, self.texts, self.labels, cache_dir=d)
            neighbors = r2.query("happy joy", top_k=1)
            self.assertEqual(fresh_model.calls, 1)  # only the query activation, bank came from disk
            self.assertEqual(neighbors[0].text, "happy joy")


class TestL2Normalize(unittest.TestCase):
    def test_unit_norm_rows(self):
        out = _l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_zero_row_stays_zero(self):
        out = _l2_normalize(np.array([[0.0, 0.0], [0.0, 5.0]]))
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
