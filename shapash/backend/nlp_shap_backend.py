"""NLP SHAP backend — token-level SHAP contributions for text classification models.

``NlpShapBackend`` wraps ``shap.Explainer`` for text inputs and implements
``run_explainer``.  All shared infrastructure (``NlpContributions`` dataclass,
``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).
"""

from __future__ import annotations

import shap

from shapash.backend.nlp_backend import NlpBackend, NlpRawExplanation


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
            self.explainer = shap.Explainer(model)

    def run_explainer(self, x) -> NlpRawExplanation:
        """Run the SHAP text explainer and return all explanation components.

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
        return NlpRawExplanation(
            contributions=shap_explanation.values,
            base_values=shap_explanation.base_values,
            data=shap_explanation.data,
        )
