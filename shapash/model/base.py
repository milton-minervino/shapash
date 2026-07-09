"""Capability-based model interface for text explainability.

``TextModel`` is the minimal contract every adapter implements: turn a list of strings into a
probability matrix. Optional *capabilities* are expressed as separate mixin ABCs so that a
component can require exactly what it needs and no more:

* ``SupportsTokenization`` — split text into tokens and rebuild text from tokens.
* ``SupportsEmbeddings`` — expose the input-embedding table and mean-pooled sentence embeddings.
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
from typing import Any

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


class SupportsTokenization(ABC):
    """Capability: split text into tokens and rebuild text from tokens."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the token strings for a single text (model's own tokenization)."""

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from a list of tokens."""


class SupportsEmbeddings(ABC):
    """Capability: expose the input-embedding table and mean-pooled sentence embeddings."""

    @abstractmethod
    def get_embedding_table(self) -> tuple[list[str], np.ndarray]:
        """Return ``(vocab, matrix)`` where ``matrix`` has shape ``(vocab_size, hidden_dim)``."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one dense vector per text, shape ``(n_texts, hidden_dim)``."""


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
