"""NLP backend infrastructure — shared base for text explainability backends.

Two dataclasses form the typed pipeline between raw explainer output and the
final explanation object:

* ``NlpRawExplanation`` — returned by every concrete ``run_explainer``.
  Carries ``contributions``, ``base_values``, and ``data`` as typed fields
  instead of an untyped dict.
* ``NlpContributions`` — returned by ``get_local_contributions`` to callers.
  Adds word-importance aggregation and optional metadata (label names, index).

``NlpBackend`` is the abstract base class that owns the common ``__init__``
skeleton and the ``get_local_contributions`` implementation.  Concrete
subclasses (``NlpShapBackend``, ``NlpLimeBackend``) only need to implement
``run_explainer``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shapash.backend.backend import Backend


@dataclass
class NlpRawExplanation:
    """Typed intermediate output of ``NlpBackend.run_explainer``.

    Every concrete backend returns this instead of a plain dict so that the
    three-field contract is enforced structurally rather than by convention.
    ``get_local_contributions`` consumes it and produces ``NlpContributions``.

    Attributes
    ----------
    contributions : list[np.ndarray]
        Per-token contribution values, one array per sample.
        Shape per sample: ``(n_tokens_i, n_classes)`` or ``(n_tokens_i,)``.
    base_values : np.ndarray
        Prediction baseline for each sample — SHAP base values or LIME
        intercepts depending on the backend.  Shape ``(n_samples, n_classes)``
        or ``(n_samples,)``.
    data : list[list[str]]
        Token or word strings per sample (variable length).
    """

    contributions: list[np.ndarray]
    base_values: np.ndarray
    data: list[list[str]]


@dataclass
class NlpContributions:
    """Token-level contributions for a batch of text samples.

    Attributes
    ----------
    token_strings : list[list[str]]
        Tokenized representation of each sample (variable length per sample).
    values : list[np.ndarray]
        Per-token contribution values, one array per sample.
        Each array has shape ``(n_tokens_i, n_classes)`` for multi-class models
        or ``(n_tokens_i,)`` for binary/regression.
    base_values : np.ndarray
        Baseline prediction for each sample, shape ``(n_samples, n_classes)``
        or ``(n_samples,)``.
    label_names : list[str] or None
        Human-readable class names; set by the caller after construction.
    index : pd.Index or None
        Row index from the source ``pd.Series``; set by the caller after construction.
    """

    token_strings: list[list[str]]
    values: list[np.ndarray]
    base_values: np.ndarray
    label_names: list[str] | None = field(default=None)
    index: pd.Index | None = field(default=None)

    _SPECIAL_RE: re.Pattern = re.compile(r"^\[.*\]$|^##|^\s*$")

    def __len__(self) -> int:
        return len(self.token_strings)

    def word_importance(
        self,
        label_idx: int,
        n_top: int = 20,
        filter_special: bool = True,
        filter_sign: str = "all",
        exclude_words: set[str] | None = None,
        sample_indices: list[int] | None = None,
    ) -> pd.Series:
        """Aggregate token contributions by word across all samples for one class.

        For each unique (stripped) token string that appears in the batch,
        computes the mean contribution across all its occurrences.

        Parameters
        ----------
        label_idx : int
            Index of the class to compute importance for.
        n_top : int
            Maximum number of words to return, sorted by ``|mean contribution|``.
        filter_special : bool
            If ``True``, skip empty strings, ``[CLS]``/``[SEP]``-style bracket
            tokens, and ``##subword`` wordpiece prefixes.
        filter_sign : {"all", "positive", "negative"}
            If ``"positive"``, keep only words with positive mean contribution.
            If ``"negative"``, keep only words with negative mean contribution.
        exclude_words : set[str], optional
            Additional set of word strings to exclude (e.g. user-selected
            stopwords or corpus words to hide).
        sample_indices : list[int], optional
            Positional indices of the samples to include in the aggregation.
            When ``None`` (default) all samples are used.

        Returns
        -------
        pd.Series
            Word → mean contribution, sorted by absolute value descending.
        """
        _exclude: set[str] = exclude_words or set()
        word_contribs: dict[str, list[float]] = {}
        iter_data = (
            ((self.token_strings[i], self.values[i]) for i in sample_indices)
            if sample_indices is not None
            else zip(self.token_strings, self.values, strict=True)
        )
        for tokens, vals in iter_data:
            vals_1d: np.ndarray = vals[:, label_idx] if vals.ndim == 2 else vals
            for tok, val in zip(tokens, vals_1d, strict=True):
                tok_clean = tok.strip()
                if filter_special and self._SPECIAL_RE.match(tok_clean):
                    continue
                if tok_clean in _exclude:
                    continue
                if tok_clean not in word_contribs:
                    word_contribs[tok_clean] = []
                word_contribs[tok_clean].append(float(val))

        importance = pd.Series({w: float(np.mean(vs)) for w, vs in word_contribs.items()})
        importance = importance.reindex(importance.abs().sort_values(ascending=False).index)

        if filter_sign == "positive":
            importance = importance[importance > 0]
        elif filter_sign == "negative":
            importance = importance[importance < 0]

        return importance.head(n_top)


class NlpBackend(Backend):
    """Abstract base class for NLP explainability backends.

    Owns the ``__init__`` skeleton shared by all text backends and the
    ``get_local_contributions`` implementation.  Concrete subclasses must
    implement ``run_explainer`` and return an ``NlpRawExplanation``.

    Parameters
    ----------
    model : callable
        Text model or pipeline accepted by the concrete backend.
    preprocessing : None
        Unused; accepted only for signature compatibility with the concrete
        subclasses' ``super().__init__`` calls.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    explainer_args : dict, optional
        Keyword arguments forwarded to the underlying explainer constructor.
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to the explainer call / ``explain_instance``.
    """

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        self.model = model
        self._classes = label_names or []
        self.explainer_args = explainer_args or {}
        self.explainer_compute_args = explainer_compute_args or {}

    def get_local_contributions(
        self, x, explain_data: NlpRawExplanation, subset: list[int] | None = None
    ) -> NlpContributions:
        """Convert a ``NlpRawExplanation`` to ``NlpContributions``.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples (not used here; kept for interface compatibility).
        explain_data : NlpRawExplanation
            Typed output of ``run_explainer``.
        subset : list[int], optional
            Positional indices to select a subset of samples.

        Returns
        -------
        NlpContributions
            Token strings, contribution values, and baseline predictions for
            each sample.
        """
        token_strings: list[list[str]] = [list(s) for s in explain_data.data]
        values: list[np.ndarray] = list(explain_data.contributions)
        base_values: np.ndarray = np.asarray(explain_data.base_values)

        if subset is not None:
            token_strings = [token_strings[i] for i in subset]
            values = [values[i] for i in subset]
            base_values = base_values[subset]

        return NlpContributions(token_strings=token_strings, values=values, base_values=base_values)
