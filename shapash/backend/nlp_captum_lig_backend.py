"""NLP Captum backend — token-level Layer Integrated Gradients contributions.

``NlpCaptumLigBackend`` is a second attribution method alongside ``NlpShapBackend``: it attributes a
target-class logit through the model's word-embedding layer with Captum's
``LayerIntegratedGradients`` (LIG), then sums over the embedding dimension to obtain one scalar per
token — the same per-token, per-class shape SHAP produces, so the webapp and the
``plot_sentence_highlight`` / ``plot_token_highlight`` renderers consume it unchanged.

LIG attributes at the tokenizer's *subword* granularity (``[CLS]``, ``[SEP]``, ``##`` continuation
pieces). To match the word-level highlights the SHAP backend produces, ``_aggregate_subwords`` merges
each word's subwords into a single contribution and folds the special tokens' attribution into the
baseline (see below) — so the output carries whole words, not ``##`` fragments.

LIG is a *completion* method: for each class ``c`` the token attributions sum to
``logits(x)[c] - logits(baseline)[c]``. We therefore report ``base_values[c] = logits(baseline)[c]``
so the additive ``base + Σ = total`` summary in ``plot_sentence_highlight`` stays consistent (values
live in logit space, like the additive SHAP output).

Unlike the SHAP/LIME backends (which wrap a plain text callable), this backend needs the embedding
module and a logits forward pass, so it consumes a :class:`~shapash.model.base.TextModel` that
implements :class:`~shapash.model.base.SupportsCaptumIG` (e.g. ``HFClassifierModel``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from shapash._optional import import_optional_module
from shapash.backend.nlp_backend import NlpBackend, NlpRawExplanation
from shapash.model.base import SupportsCaptumIG, has_capabilities

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# ``[CLS]``/``[SEP]``/``[PAD]``-style bracket tokens (and blanks); their attribution is folded into the
# baseline during word aggregation so it is neither shown nor lost.
_SPECIAL_RE = re.compile(r"^\[.*\]$|^\s*$")


def _aggregate_subwords(
    tokens: list[str], contributions: np.ndarray, base_values: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Merge WordPiece subwords into whole words and fold special tokens into the baseline.

    LIG attributes at the tokenizer's subword granularity, so raw output carries ``[CLS]``/``[SEP]``
    specials and ``##``-prefixed continuation pieces. This collapses each word's subword attributions
    into a single value (matching the word-level highlights the SHAP backend produces) while
    **preserving LIG's completeness relation**: special-token attributions are added to ``base_values``
    rather than discarded, so ``base + Σ(word contributions)`` still equals ``logits(x)``.

    Parameters
    ----------
    tokens : list[str]
        Subword token strings for one sample (length ``seq``).
    contributions : np.ndarray
        Per-subword contributions, shape ``(seq, n_classes)``.
    base_values : np.ndarray
        Baseline logits for the sample, shape ``(n_classes,)``.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        Word strings, per-word contributions ``(n_words, n_classes)``, and the adjusted baseline
        ``(n_classes,)`` with special-token attribution folded in.
    """
    words: list[str] = []
    word_rows: list[np.ndarray] = []
    base = base_values.astype(float).copy()
    for tok, row in zip(tokens, contributions, strict=True):
        stripped = tok.strip()
        if _SPECIAL_RE.match(stripped):
            base = base + row  # fold special-token attribution into the baseline (keep completeness)
        elif stripped.startswith("##") and words:
            words[-1] += stripped[2:]
            word_rows[-1] = word_rows[-1] + row
        else:
            words.append(stripped)
            word_rows.append(row.astype(float).copy())

    stacked = np.stack(word_rows, axis=0) if word_rows else np.zeros((0, contributions.shape[-1]))
    return words, stacked, base


class NlpCaptumLigBackend(NlpBackend):
    """Layer Integrated Gradients backend for text classification models.

    Parameters
    ----------
    model : SupportsCaptumIG
        A text model exposing the Captum attribution surface (embedding module, ``encode``,
        ``reference_ids``, ``logits``) — typically an ``HFClassifierModel``.
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    explainer_args : dict, optional
        Unused; accepted for interface parity with the other NLP backends.
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to ``LayerIntegratedGradients.attribute`` for every sample and
        class (e.g. ``{"n_steps": 50, "internal_batch_size": 16}``). ``n_steps`` defaults to 50.
    show_progress : bool, default False
        When True, wrap the per-sample attribution loop in a ``tqdm`` progress bar (LIG runs one
        integration per class per sample, so a batch is slow). Best-effort: if ``tqdm`` is not
        installed the loop runs silently. Left off by default so the library emits no stdout.

    Raises
    ------
    TypeError
        If ``model`` does not implement :class:`~shapash.model.base.SupportsCaptumIG`.
    """

    name = "nlp_captum_lig"

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
        show_progress: bool = False,
    ) -> None:
        if not has_capabilities(model, SupportsCaptumIG):
            raise TypeError(
                "NlpCaptumLigBackend requires a model implementing SupportsCaptumIG "
                "(e.g. HFClassifierModel); got a model without the Captum attribution surface."
            )
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.show_progress = show_progress
        captum_attr = import_optional_module("captum.attr", extra=_NLP_EXTRA)
        self.explainer = captum_attr.LayerIntegratedGradients(model.logits, model.embedding_layer)

    def run_explainer(self, x) -> NlpRawExplanation:
        """Attribute each text with LIG, one pass per class, and return per-token contributions.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpRawExplanation
            Ragged list of ``(n_tokens_i, n_classes)`` value arrays, per-sample baseline logits, and
            token strings per sample.
        """
        model: SupportsCaptumIG = self.model
        attribute_args = {"n_steps": 50, **self.explainer_compute_args}

        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []

        for text in self._progress_iter(list(x)):
            input_ids, attention_mask, tokens = model.encode(text)
            ref_ids = model.reference_ids(input_ids)
            base_logits = model.logits(ref_ids, attention_mask)[0].detach().cpu().numpy()
            n_classes = int(base_logits.shape[-1])

            per_class = []
            for target in range(n_classes):
                attributions = self.explainer.attribute(
                    input_ids,
                    baselines=ref_ids,
                    target=target,
                    additional_forward_args=(attention_mask,),
                    **attribute_args,
                )
                # (1, seq, hidden) -> (seq,): sum the attribution over the embedding dimension.
                token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                per_class.append(token_scores)

            # Collapse subwords to whole words (dropping special tokens) so LIG highlights read like
            # the SHAP backend's; special-token attribution is folded into base_logits to stay additive.
            word_tokens, word_contribs, base_logits = _aggregate_subwords(
                list(tokens), np.stack(per_class, axis=-1), base_logits
            )
            contributions.append(word_contribs)  # (n_words, n_classes)
            base_values.append(base_logits)
            data.append(word_tokens)

        return NlpRawExplanation(
            contributions=contributions,
            base_values=np.stack(base_values, axis=0),
            data=data,
        )

    def _progress_iter(self, items: list[str]) -> Iterable[str]:
        """Wrap ``items`` in a ``tqdm`` bar when ``show_progress`` is set, else return it unchanged.

        Best-effort and dependency-free: ``tqdm`` is imported with ``errors="ignore"`` so a missing
        install simply yields the plain list rather than raising.
        """
        if not self.show_progress:
            return items
        tqdm_mod = import_optional_module("tqdm", errors="ignore")
        if tqdm_mod is None:
            return items
        return tqdm_mod.tqdm(items, desc="LIG attribution", unit="text")
