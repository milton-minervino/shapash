"""Interactive (live) compute contract for what-if tooling.

The master refactoring's ``ExplainerView`` is a **read-only** DTO over a *compiled* result — it
deliberately cannot run the model on new inputs. Interactive what-if (re-predicting text the user
edits, generating counterfactuals on the fly) needs the opposite: live access to the model,
explanation backend and generator.

``InteractiveEngine`` is that live counterpart, kept separate from the read-only view (mirroring LIT
keeping ``Model`` separate from data). Webapp components that mutate/regenerate inputs *require* an
engine and self-disable when one is absent (e.g. an explainer restored from a snapshot, which holds
no model). ``NlpExplainer`` implements this Protocol.

The capability flags (``can_edit`` / ``can_counterfactual``) let components check exactly what the
bound engine supports, the same way generators check model capabilities.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.generators.base import Counterfactual, Field


@runtime_checkable
class InteractiveEngine(Protocol):
    """Live compute surface for interactive text what-if components."""

    def can_edit(self) -> bool:
        """Whether live re-prediction / re-explanation of edited text is available."""
        ...

    def can_counterfactual(self) -> bool:
        """Whether automated counterfactual generation is available."""
        ...

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Return ``(predicted_label, {label: probability})`` for a single text."""
        ...

    def explain_text(self, text: str) -> tuple[NlpContributions, str, dict[str, float]]:
        """Re-explain a single (possibly edited) text.

        Returns
        -------
        contributions : NlpContributions
            Token-level contributions for the single text (one sample).
        label : str
            Predicted label.
        probabilities : dict[str, float]
            Per-class probabilities.
        """
        ...

    def generate_counterfactuals(self, text: str, config: dict | None = None) -> list[Counterfactual]:
        """Generate counterfactuals for ``text`` using the bound generator."""
        ...

    def cf_config_spec(self) -> dict[str, Field]:
        """Return the bound generator's config spec (empty when no generator)."""
        ...
