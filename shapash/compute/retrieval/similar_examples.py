"""Similar-example retrieval: nearest reference examples in a layer's activation space.

For a prediction, this retrieves the reference-corpus examples whose internal representation is most
similar to the query's — the examples the model treats most alike. It is the in-house counterpart of
Captum's ``SimilarityInfluence``, reimplemented over the capability-based model layer
(:class:`~shapash.model.base.SupportsActivations`) so it needs no per-model wiring and carries none of
Captum 0.9.0's single-layer / short-final-batch quirks. The cost splits cleanly:

* **Bank** — one activation vector per reference example, computed once and cached to a single ``.npy``
  matrix (keyed by the corpus hash, layer, and model id). This is the expensive, amortizable part.
* **Query** — one activation vector for the query text, then cosine similarity against the bank and a
  top-k selection. Milliseconds, so it runs live per selection in the webapp.

This is a *representation-similarity* (nearest-neighbour) method, **not** a leave-one-out / influence-
function measure: it surfaces the examples most alike in decision space, a cheap and faithful proxy for
"what shaped this prediction", not a causal retraining attribution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shapash.model.base import SupportsActivations, has_capabilities


@dataclass(frozen=True)
class Neighbor:
    """One retrieved reference example for a query.

    Attributes
    ----------
    index : int
        Position of the example within the reference corpus.
    score : float
        Cosine similarity to the query in the chosen layer's activation space (higher = more similar).
    text : str
        The reference example's text.
    label : str or None
        The reference example's label, when the corpus was built with labels.
    """

    index: int
    score: float
    text: str
    label: str | None = None


def _hash_corpus(texts: list[str], layer: str, model_id: str) -> str:
    """Stable cache key over the corpus texts + layer + model id (order-sensitive)."""
    h = hashlib.md5(usedforsecurity=False)
    h.update(f"{model_id}\0{layer}\0".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()


class SimilarExampleRetriever:
    """Retrieve the reference examples most similar to a query in a model layer's activation space.

    Parameters
    ----------
    model : SupportsActivations
        A model exposing :meth:`~shapash.model.base.SupportsActivations.activations`.
    reference_texts : list[str]
        The corpus to retrieve from (typically the model's training set).
    reference_labels : list[str] or None, optional
        Labels aligned with ``reference_texts`` — surfaced on each :class:`Neighbor` when present.
    layer : str or None, optional
        Fully-qualified layer name to compare in. Defaults to the model's
        :attr:`~shapash.model.base.SupportsActivations.default_activation_layer`.
    cache_dir : str or Path or None, optional
        When given, the activation bank is persisted to ``<cache_dir>/<hash>.sim.npy`` and reloaded
        on later runs (keyed by the corpus + layer + model id), so only the first build pays the cost.

    Examples
    --------
    >>> retriever = SimilarExampleRetriever(model, train_texts, train_labels)
    >>> for n in retriever.query("i feel wonderful today", top_k=5):
    ...     print(n.score, n.label, n.text)
    """

    def __init__(
        self,
        model: SupportsActivations,
        reference_texts: list[str],
        reference_labels: list[str] | None = None,
        layer: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not has_capabilities(model, SupportsActivations):
            raise TypeError(
                f"{type(model).__name__} does not support layer activations "
                "(SupportsActivations); similar-example retrieval is unavailable."
            )
        if reference_labels is not None and len(reference_labels) != len(reference_texts):
            raise ValueError(
                f"reference_labels length ({len(reference_labels)}) must match "
                f"reference_texts length ({len(reference_texts)})."
            )
        self.model = model
        self.reference_texts = list(reference_texts)
        self.reference_labels = list(reference_labels) if reference_labels is not None else None
        self.layer = layer or model.default_activation_layer
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._bank: np.ndarray | None = None  # (n_reference, hidden), L2-normalized rows

    @property
    def size(self) -> int:
        """Number of reference examples."""
        return len(self.reference_texts)

    def _cache_file(self) -> Path | None:
        """Path of the on-disk bank for this corpus/layer/model, or ``None`` when caching is off."""
        if self.cache_dir is None:
            return None
        model_id = getattr(self.model, "model_id", None) or type(self.model).__name__
        return self.cache_dir / f"{_hash_corpus(self.reference_texts, self.layer, str(model_id))}.sim.npy"

    def build(self) -> None:
        """Compute (or load) and L2-normalize the reference activation bank.

        Called automatically on the first :meth:`query`; call it up front to pay the cost eagerly
        (e.g. at app start). Rows are stored L2-normalized so a query becomes a single matrix-vector
        product. Idempotent — a second call is a no-op once the bank is in memory.
        """
        if self._bank is not None:
            return
        cache_file = self._cache_file()
        if cache_file is not None and cache_file.exists():
            self._bank = np.load(cache_file)
            return
        raw = self.model.activations(self.reference_texts, self.layer)
        self._bank = _l2_normalize(raw)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._bank)

    def query(self, text: str, top_k: int = 5) -> list[Neighbor]:
        """Return the ``top_k`` reference examples most similar to ``text``, most-similar first.

        Parameters
        ----------
        text : str
            The query text (a dataset row, an edited sentence, or a counterfactual).
        top_k : int
            Number of neighbours to return (clamped to the corpus size).

        Returns
        -------
        list[Neighbor]
            Neighbours ordered by descending cosine similarity.
        """
        self.build()
        assert self._bank is not None  # noqa: S101 - build() guarantees this
        query_vec = _l2_normalize(self.model.activations([text], self.layer))[0]
        sims = self._bank @ query_vec  # cosine, rows already unit-norm
        k = max(1, min(top_k, self.size))
        # argpartition for the top-k, then sort just those descending.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            Neighbor(
                index=int(i),
                score=float(sims[i]),
                text=self.reference_texts[i],
                label=self.reference_labels[i] if self.reference_labels is not None else None,
            )
            for i in top_idx
        ]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (zero rows left as zeros)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
