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


if __name__ == "__main__":
    unittest.main()
