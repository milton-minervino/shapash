"""NLP SHAP backend — token-level SHAP contributions for text classification models.

``NlpShapBackend`` wraps ``shap.Explainer`` for text inputs and implements
``run_explainer``.  All shared infrastructure (``NlpContributions`` dataclass,
``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).

SHAP's ``Text`` masker (``shap.maskers.Text.token_segments``) reconstructs each token as the slice
of the original string running up to the start of the next token, rather than the tokenizer's raw
WordPiece strings. A subword glued directly onto the previous piece (e.g. ``"up"`` + ``"dating "``
for "updating") therefore carries no trailing whitespace, while the piece that actually ends a word
carries the whitespace that follows it in the source sentence. ``_aggregate_subwords`` uses that
signal to merge subwords back into whole words — matching the word-level highlights
``nlp_captum_lig_backend`` produces — and folds special ([CLS]/[SEP]-equivalent blank) tokens'
attribution into the baseline so ``base + Σ(word contributions)`` keeps SHAP's additive guarantee.
"""

from __future__ import annotations

import re

import numpy as np
import shap

from shapash.backend.nlp_backend import NlpBackend, NlpRawExplanation

# ``[CLS]``/``[SEP]``-style bracket tokens (and blanks) SHAP's ``Text`` masker reports as empty
# strings; their attribution is folded into the baseline during word aggregation.
_SPECIAL_RE = re.compile(r"^\[.*\]$|^\s*$")


def _aggregate_subwords(
    tokens: list[str], contributions: np.ndarray, base_values: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Merge SHAP's subword fragments into whole words and fold specials into the baseline.

    Parameters
    ----------
    tokens : list[str]
        Token strings for one sample (length ``seq``), as produced by SHAP's text masker.
    contributions : np.ndarray
        Per-token contributions, shape ``(seq, n_classes)``.
    base_values : np.ndarray
        Baseline SHAP values for the sample, shape ``(n_classes,)``.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        Word strings, per-word contributions ``(n_words, n_classes)``, and the adjusted baseline
        ``(n_classes,)`` with special-token attribution folded in.
    """
    words: list[str] = []
    word_rows: list[np.ndarray] = []
    base = base_values.astype(float).copy()
    buffer_text = ""
    buffer_row: np.ndarray | None = None

    def _flush() -> None:
        nonlocal buffer_text, buffer_row
        if buffer_row is not None:
            words.append(buffer_text.strip())
            word_rows.append(buffer_row)
            buffer_text, buffer_row = "", None

    for tok, row in zip(tokens, contributions, strict=True):
        if _SPECIAL_RE.match(tok.strip()):
            _flush()
            base = base + row
            continue
        buffer_text += tok
        buffer_row = row.astype(float).copy() if buffer_row is None else buffer_row + row
        if tok != tok.rstrip():  # trailing whitespace in the source text -> word boundary
            _flush()
    _flush()

    stacked = np.stack(word_rows, axis=0) if word_rows else np.zeros((0, contributions.shape[-1]))
    return words, stacked, base


class NlpShapBackend(NlpBackend):
    """SHAP backend for text classification models (HuggingFace pipelines, etc.).

    Wraps ``shap.Explainer`` for text inputs and returns ``NlpContributions``
    via the shared ``get_local_contributions`` in ``NlpBackend``.

    Parameters
    ----------
    model : callable
        A text pipeline callable accepted by ``shap.Explainer`` (e.g. a
        ``transformers.pipeline`` with ``return_all_scores=True``).
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    masker : any, optional
        Forwarded to ``shap.Explainer`` when ``explainer_args`` is not given.
        Typically ``None`` for text (SHAP auto-selects a ``TextMasker``).
    explainer_args : dict, optional
        Keyword arguments forwarded to ``shap.Explainer.__init__``.
        Use ``{"explainer": SomeExplainerClass, ...}`` to inject a custom
        explainer class (the ``"explainer"`` key selects the class; all other
        keys are forwarded as its constructor arguments).
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to the explainer call (``__call__``).
    """

    name = "nlp_shap"

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        masker=None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.masker = masker

        if "explainer" in self.explainer_args:
            shap_params = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
            self.explainer = self.explainer_args["explainer"](**shap_params)
        elif self.explainer_args:
            self.explainer = shap.Explainer(**self.explainer_args)
        else:
            # ``masker=None`` lets SHAP auto-infer a Text masker from a transformers pipeline (the
            # HFClassifierModel/pipeline path); an explicit masker is required when ``model`` is a plain
            # scoring callable (external-head models expose one via ``TextModel.shap_masker``).
            self.explainer = shap.Explainer(model, masker=self.masker)

    def run_explainer(self, x) -> NlpRawExplanation:
        """Run the SHAP text explainer and return all explanation components.

        Subword tokens are merged into whole words via ``_aggregate_subwords`` before being
        returned, so callers (word importance, sentence/token highlight plots) always see
        word-level contributions.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpRawExplanation
            Ragged list of value arrays, baseline predictions, and token
            strings per sample.
        """
        shap_explanation = self.explainer(x, **self.explainer_compute_args)

        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []
        for tokens, values, base in zip(
            shap_explanation.data, shap_explanation.values, shap_explanation.base_values, strict=True
        ):
            words, word_contribs, word_base = _aggregate_subwords(list(tokens), np.asarray(values), np.asarray(base))
            data.append(words)
            contributions.append(word_contribs)
            base_values.append(word_base)

        return NlpRawExplanation(
            contributions=contributions,
            base_values=np.stack(base_values, axis=0),
            data=data,
        )
