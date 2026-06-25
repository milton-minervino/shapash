"""NLP LIME backend — word-level LIME contributions for text classification models.

``NlpLimeBackend`` wraps ``LimeTextExplainer`` and implements ``run_explainer``.
All shared infrastructure (``NlpContributions`` dataclass,
``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).

LIME works at word level (bag-of-words by default) rather than at the subword
or token level used by SHAP.  Each sample's ``token_strings`` is therefore the
list of unique vocabulary words found by the ``split_expression`` tokeniser, not
HuggingFace subword tokens.  Only the top ``num_features`` words per label
receive a non-zero weight; all others are zero in the dense matrix.
"""

from __future__ import annotations

import numpy as np

try:
    from lime.lime_text import LimeTextExplainer

    _lime_available = True
except ImportError:
    _lime_available = False

from shapash.backend.nlp_backend import NlpBackend, NlpRawExplanation


class NlpLimeBackend(NlpBackend):
    """LIME backend for text classification models.

    Wraps ``LimeTextExplainer`` and returns ``NlpContributions`` via the shared
    ``get_local_contributions`` in ``NlpBackend``.

    Parameters
    ----------
    model : callable
        A function ``f(texts: list[str]) -> np.ndarray`` of shape
        ``(n_texts, n_classes)`` returning class probabilities.
        HuggingFace pipelines require a thin wrapper — see the example below.
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
        Forwarded to ``LimeTextExplainer`` as ``class_names``.
    mask_string : str or None
        Token used to replace masked words when ``bow=False``.  Mirrors the
        ``masker`` parameter of ``NlpShapBackend``.  Defaults to
        ``'UNKWORDZ'`` inside LIME.
    explainer_args : dict, optional
        Keyword arguments forwarded to ``LimeTextExplainer.__init__``.
        Supported keys: ``kernel_width``, ``kernel``, ``verbose``,
        ``feature_selection``, ``split_expression``, ``bow``,
        ``random_state``, ``char_level``.
        Use ``{"explainer": SubclassOfLimeTextExplainer, ...rest...}`` to
        inject a custom explainer class (mirrors the SHAP escape hatch).
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to ``LimeTextExplainer.explain_instance``
        on every call.  Supported keys: ``num_features`` (default 10),
        ``num_samples`` (default 5000), ``distance_metric`` (default
        ``'cosine'``), ``model_regressor``, ``labels``, ``top_labels``.
        If neither ``labels`` nor ``top_labels`` is provided, ``labels`` is
        automatically set to ``range(len(label_names))``.

    Example
    -------
    >>> import numpy as np
    >>> from transformers import pipeline
    >>> pipe = pipeline(
    ...     "text-classification",
    ...     model="distilbert-base-uncased-finetuned-sst-2-english",
    ...     return_all_scores=True,
    ... )
    >>> def classifier_fn(texts):
    ...     return np.array([[s["score"] for s in row] for row in pipe(texts)])
    >>> backend = NlpLimeBackend(
    ...     classifier_fn,
    ...     label_names=["NEGATIVE", "POSITIVE"],
    ...     explainer_compute_args={"num_features": 15, "num_samples": 3000},
    ... )
    """

    name = "nlp_lime"

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        mask_string: str | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        if not _lime_available:
            raise ImportError("lime is required for NlpLimeBackend — pip install lime")

        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.mask_string = mask_string

        if "explainer" in self.explainer_args:
            lime_params = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
            self.explainer = self.explainer_args["explainer"](**lime_params)
        else:
            self.explainer = LimeTextExplainer(
                class_names=label_names or None,
                mask_string=self.mask_string,
                **self.explainer_args,
            )

    def _classifier_fn(self, texts: list[str]) -> np.ndarray:
        """Wrap self.model to guarantee a float numpy array of shape (n, n_classes).

        Handles three output formats:

        * ``np.ndarray`` — returned as-is.
        * ``list[list[dict]]`` — HuggingFace pipeline with ``return_all_scores=True``.
          Scores are extracted by label name using ``self._classes`` as the column
          order, so ``label_names`` must be provided when the model uses this format.
        * Anything else — coerced with ``np.array(..., dtype=float)``.

        LIME's internals index the result with ``[:, label_idx]``, so the array
        must be 2-D and numeric.
        """
        result = self.model(texts)
        if isinstance(result, np.ndarray):
            return result
        # HuggingFace pipeline with return_all_scores=True → list[list[dict]]
        if result and isinstance(result[0], list) and isinstance(result[0][0], dict):
            label_to_idx = {name: i for i, name in enumerate(self._classes)}
            matrix = np.zeros((len(result), len(self._classes)), dtype=np.float64)
            for i, preds in enumerate(result):
                for pred in preds:
                    idx = label_to_idx.get(pred["label"])
                    if idx is not None:
                        matrix[i, idx] = pred["score"]
            return matrix
        return np.array(result, dtype=np.float64)

    def run_explainer(self, x) -> NlpRawExplanation:
        """Run LimeTextExplainer on each sample and normalise output.

        Converts LIME's sparse per-label ``{label_id: [(word_id, weight)]}``
        representation into a dense ``(n_words, n_classes)`` array so that the
        returned ``NlpRawExplanation`` matches the same field shapes as
        ``NlpShapBackend``.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpRawExplanation
            Dense weight arrays, LIME intercepts as base values, and unique
            vocabulary words per sample.
        """
        texts = list(x)
        n_classes = len(self._classes)

        compute_args = dict(self.explainer_compute_args)
        if "labels" not in compute_args and "top_labels" not in compute_args and n_classes:
            compute_args["labels"] = list(range(n_classes))

        contributions: list[np.ndarray] = []
        base_values_list: list[list[float]] = []
        data: list[list[str]] = []

        for text in texts:
            exp = self.explainer.explain_instance(text, self._classifier_fn, **compute_args)

            indexed_string = exp.domain_mapper.indexed_string
            vocab: list[str] = list(indexed_string.inverse_vocab)
            n_words = len(vocab)
            effective_n_classes = n_classes or len(exp.local_exp)

            # Dense weight matrix — same shape contract as NlpShapBackend values.
            weight_matrix = np.zeros((n_words, effective_n_classes), dtype=float)
            for label_idx in range(effective_n_classes):
                if label_idx in exp.local_exp:
                    for word_id, weight in exp.local_exp[label_idx]:
                        weight_matrix[word_id, label_idx] = weight

            contributions.append(weight_matrix)
            base_values_list.append([exp.intercept.get(i, 0.0) for i in range(effective_n_classes)])
            data.append(vocab)

        return NlpRawExplanation(
            contributions=contributions,
            base_values=np.array(base_values_list),
            data=data,
        )
