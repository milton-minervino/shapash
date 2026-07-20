"""Capability-based model interface for text explainability.

``TextModel`` is the minimal contract every adapter implements: turn a list of strings into a
probability matrix. Optional *capabilities* are expressed as separate mixin ABCs so that a
component can require exactly what it needs and no more:

* ``SupportsTokenization`` — split text into tokens and rebuild text from tokens.
* ``SupportsEmbeddings`` — expose the input-embedding table and one dense vector per text, in a
  caller-selectable *representation space* (see :meth:`SupportsEmbeddings.embed`). This single
  capability backs both the 2-D projection scatter and similar-example retrieval, which agree only
  as far as they both go through it — see the caveat on :class:`SupportsEmbeddings`.
* ``SupportsGradients`` — expose per-token gradients of a target-class logit.
* ``SupportsCaptumIG`` — expose the embedding module + a logits forward pass so a layer-attribution
  method (Captum ``LayerIntegratedGradients``) can attribute through the embeddings.

A prediction-only model (e.g. a HuggingFace pipeline) implements only ``TextModel``; a raw
classifier with tokenizer access implements all three. Generators check compatibility with
``has_capabilities(model, SupportsGradients, SupportsEmbeddings)`` rather than isinstance-ing a
concrete class — this is what keeps HotFlip and friends model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np


class TextModel(ABC):
    """Abstract text-classification model — the minimal capability every adapter provides.

    Attributes
    ----------
    label_names : list[str] or None
        Human-readable class names in the same column order as ``predict`` output.
    task : str
        Task kind; only ``"classification"`` is supported for now.
    """

    task: str = "classification"

    def __init__(self, label_names: list[str] | None = None) -> None:
        self.label_names = label_names

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities for each input text.

        Parameters
        ----------
        texts : list[str]
            Raw input strings.

        Returns
        -------
        np.ndarray, shape (n_texts, n_classes)
            Row-wise class probabilities.
        """

    @property
    def n_classes(self) -> int | None:
        """Number of classes, or ``None`` when unknown before the first prediction."""
        return len(self.label_names) if self.label_names is not None else None

    @property
    def shap_masker(self) -> Any:
        """Masker SHAP should use, or ``None`` to let SHAP infer one from :attr:`shap_callable`.

        SHAP's text explainer needs a scoring function *and* a way to segment/mask the text. When
        ``shap_callable`` is a ``transformers`` pipeline SHAP infers a ``Text`` masker from it, so the
        default ``None`` is correct. Adapters whose ``shap_callable`` is a plain callable (no tokenizer
        for SHAP to find) must override this and return an explicit ``shap.maskers.Text``.
        """
        return None

    @property
    def model_id(self) -> str:
        """Stable identity for this adapter *and its configuration*, for cache keys.

        Anything that changes the vectors or scores a downstream cache stores must appear here —
        otherwise two differently-configured models collide on the same cache entry and silently reload
        each other's results. The default is the class name alone, which is only safe for an adapter with
        no configuration; :class:`~shapash.model.encoder.EncoderClassifierModel` overrides it.
        """
        return type(self).__name__


class SupportsTokenization(ABC):
    """Capability: split text into tokens and rebuild text from tokens."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the token strings for a single text (model's own tokenization)."""

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from a list of tokens."""


class SupportsEmbeddings(ABC):
    """Capability: expose the input-embedding table and one dense vector per text.

    :meth:`embed` is the single entry point for *"give me a vector per text in some representation
    space"* — the space is a parameter, not a separate capability. Everything downstream that compares
    or projects texts (the 2-D projection scatter, similar-example retrieval) is expected to go
    through it, so that changing the model's space moves them together.

    That agreement is a **convention, not a guarantee**: a caller who computes its own coordinates and
    hands them to the webapp (``run_app(scatter_xy=...)``) bypasses this entirely, and nothing detects
    it. Use :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.compute_projection`, which derives the
    scatter from this method, unless you specifically want a space of your own choosing.
    """

    @abstractmethod
    def get_embedding_table(self) -> tuple[list[str], np.ndarray]:
        """Return ``(vocab, matrix)`` where ``matrix`` has shape ``(vocab_size, hidden_dim)``."""

    def resolve_space(self, space: str | None = None) -> str:
        """Return the name of the space :meth:`embed` will actually use for ``space``.

        Everything that caches embeddings must key on the *effective* space, not on the argument it was
        given — otherwise ``None`` (meaning "the adapter's default") collides with whatever that default
        currently is, and a cache entry from one space is silently reloaded for another. This is the one
        place that resolution happens; callers must not reimplement it.

        Parameters
        ----------
        space : str or None, optional
            The space a caller asked for, or ``None`` for the adapter's default.

        Returns
        -------
        str
            A concrete space name, never ``None``. The base implementation has no notion of a default
            space and reports ``"default"``; adapters with a configurable one (e.g.
            :class:`~shapash.model.encoder.EncoderClassifierModel`) override this.
        """
        return space if space is not None else "default"

    @abstractmethod
    def embed(self, texts: list[str], space: str | None = None) -> np.ndarray:
        """Return one dense vector per text, shape ``(n_texts, hidden_dim)``.

        Parameters
        ----------
        texts : list of str
            Input strings.
        space : str or None, optional
            Which representation to return. When ``None`` (default), the adapter's configured default
            space is used. Adapters are free to define their own space names; the shared
            :class:`~shapash.model.encoder.EncoderClassifierModel` vocabulary is ``"decision"`` (input
            to the final classification linear), ``"pooled"`` (the pooled last hidden state), or any
            named submodule of the model, whose token-level output is pooled to one vector per text.

        Returns
        -------
        np.ndarray, shape (n_texts, hidden_dim)
            Dense vectors aligned to ``texts``. The dimension depends on ``space``.
        """


class SupportsGradients(ABC):
    """Capability: per-token gradients of a target-class logit w.r.t. input embeddings."""

    @abstractmethod
    def token_gradients(self, text: str, target_class: int) -> tuple[list[str], np.ndarray]:
        """Return ``(tokens, grads)`` for one text and target class.

        Parameters
        ----------
        text : str
            Input string.
        target_class : int
            Class index whose logit is differentiated.

        Returns
        -------
        tokens : list[str]
            Tokens aligned with the gradient rows (special tokens excluded).
        grads : np.ndarray, shape (n_tokens, hidden_dim)
            Gradient of the target-class logit w.r.t. each token's input embedding.
        """


class SupportsCaptumIG(ABC):
    """Capability: expose the pieces Captum ``LayerIntegratedGradients`` needs.

    A layer-attribution method attributes a target logit *through* the word-embedding module, so it
    needs three things a plain :meth:`TextModel.predict` cannot provide: the embedding ``nn.Module``
    to attribute through, a way to turn text into model input tensors (and their reference/baseline
    ids), and a raw-logits forward pass to use as the Captum ``forward_func``. Kept separate from
    :class:`SupportsGradients` — which only yields per-token gradient vectors — so a backend can
    require exactly this surface. Tensor-typed arguments/returns are annotated ``Any`` to keep this
    module free of a hard ``torch`` import.
    """

    @property
    @abstractmethod
    def embedding_layer(self) -> Any:
        """Return the word-embedding ``nn.Module`` to compute layer attributions through."""

    @abstractmethod
    def encode(self, text: str) -> tuple[Any, Any, list[str]]:
        """Return ``(input_ids, attention_mask, tokens)`` for one text.

        ``input_ids`` and ``attention_mask`` are batch-size-1 tensors on the model device; ``tokens``
        are the aligned token strings (special tokens included).
        """

    @abstractmethod
    def reference_ids(self, input_ids: Any) -> Any:
        """Return baseline input ids: non-special tokens replaced by a reference (pad/mask) id.

        Same shape/device as ``input_ids``. Special tokens (e.g. ``[CLS]``/``[SEP]``) are kept so the
        baseline is a well-formed empty-content sequence.
        """

    @abstractmethod
    def logits(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return raw classification logits, shape ``(batch, n_classes)`` (the Captum forward func)."""

    def word_alignment(self, text: str) -> tuple[list[str], list[list[int]], list[int]] | None:
        """Group a text's subword tokens into whole words, model-agnostically.

        Token-level attribution methods (LIG) attribute at *subword* granularity, but highlights read
        best at the *word* level. How subwords compose into words is a tokenizer concern — WordPiece
        marks continuations (``##``), byte-level BPE and SentencePiece mark word *starts* (``Ġ`` / ``▁``)
        — so the model owns it here rather than any backend string-guessing.

        Positions index the same subword axis as :meth:`encode` (i.e. the token/attribution arrays).

        Returns
        -------
        tuple[list[str], list[list[int]], list[int]] or None
            ``(words, word_positions, special_positions)`` where ``words[k]`` is the display string of
            the ``k``-th whole word, ``word_positions[k]`` are the subword indices composing it, and
            ``special_positions`` are the indices of special tokens (``[CLS]``/``<s>``/…) whose
            attribution a caller folds into the baseline. Returns ``None`` when exact alignment is
            unavailable (e.g. a slow tokenizer with no ``word_ids()``), so the caller falls back to a
            token-string heuristic.
        """
        return None


@runtime_checkable
class EmbeddingSource(Protocol):
    """Structural type for the surface an embedding *cache* consumes.

    Caching embeddings needs strictly more than :class:`SupportsEmbeddings`: besides ``embed`` and
    ``resolve_space`` it needs ``model_id``, which lives on :class:`TextModel`. Neither ABC alone
    describes that, and merging them would make ``model_id`` a capability concern or ``embed`` a
    base-model one — both wrong. This ``Protocol`` names the intersection without moving anything.

    It is a *typing* aid, not a gate. Components still admit a model by capability
    (``has_capabilities(model, SupportsEmbeddings)``); every adapter that passes that check is a
    ``TextModel`` too, so it satisfies this protocol by construction.
    """

    @property
    def model_id(self) -> str:
        """Stable identity of the model and its configuration (see :attr:`TextModel.model_id`)."""
        ...

    def resolve_space(self, space: str | None = None) -> str:
        """Name of the space :meth:`embed` will actually use (see :meth:`SupportsEmbeddings.resolve_space`)."""
        ...

    def embed(self, texts: list[str], space: str | None = None) -> np.ndarray:
        """One dense vector per text (see :meth:`SupportsEmbeddings.embed`)."""
        ...


def has_capabilities(model: object, *capabilities: type) -> bool:
    """Return ``True`` when ``model`` is an instance of every capability ABC given.

    Parameters
    ----------
    model : object
        The model adapter to test.
    *capabilities : type
        Capability ABCs (e.g. ``SupportsGradients``) the model must satisfy.

    Returns
    -------
    bool
        ``True`` only if ``model`` inherits from all requested capabilities.

    Examples
    --------
    >>> has_capabilities(model, SupportsGradients, SupportsEmbeddings)
    True
    """
    return all(isinstance(model, cap) for cap in capabilities)
