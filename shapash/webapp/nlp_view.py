"""Read-only view over a compiled ``NlpExplainer`` for webapp components.

Prototype of the master plan's ``ExplainerView`` (Phase 5) for the text modality: webapp components
read compiled explanation data **only** through this contract, never by reaching into explainer
internals. It is deliberately read-only — live actions (re-predicting edited text, generating
counterfactuals) live on the separate :class:`~shapash.explainer.interactive.InteractiveEngine`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NlpView:
    """Read-only accessor over a compiled ``NlpExplainer``.

    Parameters
    ----------
    explainer : NlpExplainer
        A compiled explainer (``compile()`` already called).
    """

    def __init__(self, explainer) -> None:
        self._explainer = explainer

    @property
    def texts(self) -> pd.Series | None:
        """The compiled input texts."""
        return self._explainer.texts

    @property
    def contributions(self):
        """The compiled ``NlpContributions``."""
        return self._explainer.contributions

    @property
    def y_pred(self) -> pd.Series | None:
        """Predicted labels for the compiled batch."""
        return self._explainer.y_pred

    @property
    def y_prob(self) -> pd.DataFrame | None:
        """Per-class probabilities for the compiled batch, if available."""
        return getattr(self._explainer, "y_prob", None)

    @property
    def y_true(self) -> pd.Series | None:
        """Ground-truth labels, if provided at compile time."""
        return getattr(self._explainer, "y_true", None)

    @property
    def label_names(self) -> list[str] | None:
        """Class names in output-column order."""
        return self._explainer.label_names or (self.contributions.label_names if self.contributions else None)

    @property
    def has_ground_truth(self) -> bool:
        """Whether ground-truth labels are available."""
        return self.y_true is not None

    @property
    def n_classes(self) -> int:
        """Number of classes inferred from the compiled contributions."""
        sample = self.contributions.values[0]
        return sample.shape[1] if np.ndim(sample) == 2 else 1
