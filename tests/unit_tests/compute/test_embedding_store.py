"""Unit tests for the shared embedding cache (no torch/transformers required).

The store's whole job is *not recomputing*, and doing so under a key that cannot collide across
models, spaces or corpora. A counting fake model makes both directly observable: an assertion on
``model.calls`` says whether a cache was hit, and comparing keys says whether two configurations
would have shared a file.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shapash.compute.embedding_store import EmbeddingStore
from shapash.model.base import EmbeddingSource

TEXTS = ["alpha", "beta", "gamma"]


class FakeModel:
    """Minimal ``EmbeddingSource``: deterministic vectors, counted calls, configurable identity/space."""

    def __init__(self, model_id="fake:v1", space="decision"):
        self._model_id = model_id
        self.space = space
        self.calls = 0
        self.requested_spaces = []

    @property
    def model_id(self):
        return self._model_id

    def resolve_space(self, space=None):
        return space if space is not None else self.space

    def embed(self, texts, space=None):
        self.calls += 1
        self.requested_spaces.append(space)
        return np.array([[float(len(t)), 1.0] for t in texts])


class TestProtocolConformance(unittest.TestCase):
    def test_fake_satisfies_embedding_source(self):
        """The protocol describes a real structural surface, not just documentation."""
        self.assertIsInstance(FakeModel(), EmbeddingSource)


class TestVectors(unittest.TestCase):
    def test_embeds_once_and_memoizes(self):
        model = FakeModel()
        store = EmbeddingStore(model, TEXTS)
        first = store.vectors()
        np.testing.assert_allclose(store.vectors(), first)
        self.assertEqual(model.calls, 1)

    def test_a_space_change_invalidates_the_in_memory_memo(self):
        """The memo is keyed by ``key``, which folds in the space — not by tag alone.

        A tag-only memo answered from the previous space after a live ``embedding_space`` switch and
        never even consulted the (correctly keyed) file on disk.
        """
        model = FakeModel()
        store = EmbeddingStore(model, TEXTS)
        store.vectors()
        model.space = "pooled"
        store.vectors()
        self.assertEqual(model.calls, 2)  # re-embedded rather than replaying the decision-space memo

    def test_the_memo_still_holds_within_one_space(self):
        """Guard on the fix above: invalidating on a space change must not disable memoization."""
        model = FakeModel()
        store = EmbeddingStore(model, TEXTS)
        for _ in range(3):
            store.vectors()
        self.assertEqual(model.calls, 1)

    def test_a_space_change_reloads_from_disk_rather_than_recomputing(self):
        """Once both spaces are on disk, switching back and forth costs no forward pass."""
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel()
            store = EmbeddingStore(model, TEXTS, cache_dir=d)
            store.vectors()
            model.space = "pooled"
            store.vectors()
            fresh = EmbeddingStore(FakeModel(), TEXTS, cache_dir=d)
            fresh.model.space = "pooled"
            fresh.vectors()
            fresh.model.space = "decision"
            fresh.vectors()
            self.assertEqual(fresh.model.calls, 0)  # both spaces served from their own files

    def test_embeds_in_the_models_default_space(self):
        """The store never picks a space of its own: it calls embed() bare and keys on what that means."""
        model = FakeModel(space="pooled")
        store = EmbeddingStore(model, TEXTS)
        store.vectors()
        self.assertEqual(model.requested_spaces, [None])
        self.assertIn("pooled", store.space_key)

    def test_without_cache_dir_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            store = EmbeddingStore(FakeModel(), TEXTS)
            store.vectors()
            self.assertIsNone(store.path("emb"))
            self.assertEqual(list(Path(d).iterdir()), [])


class TestDiskCache(unittest.TestCase):
    def test_second_store_loads_from_disk(self):
        with tempfile.TemporaryDirectory() as d:
            EmbeddingStore(FakeModel(), TEXTS, cache_dir=d).vectors()
            fresh = FakeModel()
            vectors = EmbeddingStore(fresh, TEXTS, cache_dir=d).vectors()
            self.assertEqual(fresh.calls, 0)  # served entirely from disk
            np.testing.assert_allclose(vectors, [[5.0, 1.0], [4.0, 1.0], [5.0, 1.0]])

    def test_derived_arrays_cache_beside_the_embeddings(self):
        """A projection is cached under its own tag but the same key, so both travel together."""
        with tempfile.TemporaryDirectory() as d:
            store = EmbeddingStore(FakeModel(), TEXTS, cache_dir=d)
            fits = []

            def project():
                fits.append(1)
                return store.vectors()[:, :1]

            store.cached_array("pca.proj", project)
            fresh = EmbeddingStore(FakeModel(), TEXTS, cache_dir=d)
            fresh.cached_array("pca.proj", project)
            self.assertEqual(len(fits), 1)  # the reducer did not re-fit
            self.assertTrue(store.path("pca.proj").exists())
            self.assertTrue(store.path("emb").exists())

    def test_clear_drops_every_tag_under_the_key(self):
        """Half-clearing would leave a projection derived from vectors that no longer exist."""
        with tempfile.TemporaryDirectory() as d:
            store = EmbeddingStore(FakeModel(), TEXTS, cache_dir=d)
            store.vectors()
            store.cached_array("pca.proj", lambda: np.zeros((3, 2)))
            store.clear()
            self.assertFalse(store.path("emb").exists())
            self.assertFalse(store.path("pca.proj").exists())

            model = FakeModel()
            EmbeddingStore(model, TEXTS, cache_dir=d).vectors()
            self.assertEqual(model.calls, 1)  # really recomputed, not silently reloaded


class TestKeySeparation(unittest.TestCase):
    """Each ingredient of the key must, on its own, produce a different cache entry."""

    def test_different_models_do_not_collide(self):
        a = EmbeddingStore(FakeModel(model_id="checkpoint-a"), TEXTS)
        b = EmbeddingStore(FakeModel(model_id="checkpoint-b"), TEXTS)
        self.assertNotEqual(a.key, b.key)

    def test_different_spaces_do_not_collide(self):
        a = EmbeddingStore(FakeModel(space="decision"), TEXTS)
        b = EmbeddingStore(FakeModel(space="pooled"), TEXTS)
        self.assertNotEqual(a.key, b.key)

    def test_different_corpora_do_not_collide(self):
        model = FakeModel()
        self.assertNotEqual(EmbeddingStore(model, TEXTS).key, EmbeddingStore(model, [*TEXTS, "delta"]).key)

    def test_corpus_order_matters(self):
        """Rows are positional — a reordered corpus is a different bank, not the same one."""
        model = FakeModel()
        self.assertNotEqual(EmbeddingStore(model, TEXTS).key, EmbeddingStore(model, list(reversed(TEXTS))).key)

    def test_same_configuration_collides_on_purpose(self):
        model = FakeModel()
        self.assertEqual(EmbeddingStore(model, TEXTS).key, EmbeddingStore(FakeModel(), list(TEXTS)).key)


if __name__ == "__main__":
    unittest.main()
