"""Cached embeddings for a fixed corpus in a model's current representation space.

Embedding a corpus is the expensive, repeated step behind more than one feature: similar-example
retrieval needs one vector per reference text, and the 2-D scatter needs one vector per compiled
text. Both are a pure function of *(model identity, effective space, corpus)*, so both belong behind
one cache with one key — otherwise each caller invents its own filename and they drift, which is how
a bank built in the ``"decision"`` space ends up reloaded for a scatter drawn in ``"pooled"``.

:class:`EmbeddingStore` is that one place. It owns the key and the on-disk layout; callers ask for
:meth:`~EmbeddingStore.vectors` and, when they derive something further from them (a 2-D projection),
park it next to the embeddings under the same key via :meth:`~EmbeddingStore.cached_array`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from shapash.model.base import EmbeddingSource

logger = logging.getLogger(__name__)


def _hash_corpus(texts: Sequence[str], key: str) -> str:
    """Stable digest over a model/space key plus the corpus texts (order-sensitive)."""
    h = hashlib.md5(usedforsecurity=False)
    h.update(f"{key}\0".encode())
    for text in texts:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


class EmbeddingStore:
    """One vector per text, computed once and cached to disk.

    Parameters
    ----------
    model : EmbeddingSource
        A model exposing ``model_id``, ``resolve_space`` and ``embed``.
    texts : sequence of str
        The corpus to embed. Fixed for the lifetime of the store — the cache key is derived from it.
    cache_dir : str or Path or None, optional
        Where cached ``.npy`` files live. When ``None`` the store still memoizes in memory but writes
        nothing, so a fresh process recomputes.

    Notes
    -----
    The key is ``model_id | resolve_space() | corpus-hash``. ``model_id`` is the model's own
    declaration of what makes it distinct (checkpoint, pooling, normalization, head weights) and
    ``resolve_space()`` is the single place that knows which space :meth:`embed` will *actually* use,
    so ``None`` (meaning "the model's default") can never collide with the default it stands for.

    Examples
    --------
    >>> store = EmbeddingStore(model, train_texts, cache_dir="cache/")
    >>> store.vectors().shape
    (5000, 384)
    """

    def __init__(
        self,
        model: EmbeddingSource,
        texts: Sequence[str],
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.texts = list(texts)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._arrays: dict[str, np.ndarray] = {}

    @property
    def space_key(self) -> str:
        """Readable ``model_id|space`` half of the key — what gets logged when a lookup misses."""
        return f"{self.model.model_id}|{self.model.resolve_space()}"

    @property
    def key(self) -> str:
        """The cache key: model identity, effective space, and corpus digest."""
        return _hash_corpus(self.texts, self.space_key)

    def path(self, tag: str) -> Path | None:
        """On-disk location of the ``tag`` artifact, or ``None`` when caching is off."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.key}.{tag}.npy"

    def vectors(self) -> np.ndarray:
        """Return ``(n_texts, hidden_dim)`` embeddings in the model's current space."""
        return self.cached_array("emb", lambda: np.asarray(self.model.embed(self.texts)))

    def cached_array(self, tag: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the ``tag`` array for this corpus, loading it or computing and storing it.

        Parameters
        ----------
        tag : str
            Names the artifact within this store's key (``"emb"`` for the embeddings themselves,
            e.g. ``"pca-1f3c.proj"`` for a projection derived from them). Anything that changes the
            array's contents but not the key must be reflected here.
        compute : callable
            Produces the array on a cache miss. Called at most once per process.

        Returns
        -------
        np.ndarray
            The cached array. Memoized in memory, so repeat calls are free.
        """
        if tag in self._arrays:
            return self._arrays[tag]
        cache_file = self.path(tag)
        if cache_file is not None and cache_file.exists():
            logger.info("Embedding store hit (%s) — loading %s", tag, cache_file)
            self._arrays[tag] = np.load(cache_file)
            return self._arrays[tag]

        logger.info("Embedding store miss (%s) — computing over %d texts (%s)", tag, len(self.texts), self.space_key)
        array = compute()
        self._arrays[tag] = array
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, array)
            logger.info("Embedding store cached (%s) to %s", tag, cache_file)
        return array

    def clear(self) -> None:
        """Drop every cached artifact for this corpus — in memory and, when caching, on disk.

        Removes *all* tags under this key, not just the embeddings, so a caller forcing a recompute
        cannot leave a projection behind that was derived from vectors no longer on disk.
        """
        self._arrays.clear()
        if self.cache_dir is None:
            return
        for stale in self.cache_dir.glob(f"{self.key}.*.npy"):
            stale.unlink(missing_ok=True)
            logger.info("Embedding store dropped %s", stale)
