"""``WebappComponent`` contract + capability resolution for what-if panels.

A component declares the capabilities it needs via ``requires``; :func:`available_capabilities`
computes what the bound explanation + engine actually provide, and :meth:`WebappComponent.is_available`
gates mounting on ``requires <= available``. This is the mechanism that makes the What-if Lab
appear only when the explainer holds a live (and, for counterfactuals, gradient-capable) model.

Components read the immutable :class:`~shapash.explainer.nlp_explanation.NlpExplanation` directly and
never write to it: every display choice lives in a Dash ``dcc.Store`` or a callback argument, so the
artifact a component renders is the same one that was saved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shapash.compute.diagnostics.label_noise import has_usable_probabilities
from shapash.explainer.interactive import InteractiveEngine
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.model.base import SupportsGradients, has_capabilities

# Capability tokens components may require.
CAP_PREDICT = "engine:predict"
CAP_COUNTERFACTUAL = "engine:counterfactual"
CAP_GRADIENTS = "model:gradients"
CAP_SIMILAR = "engine:similar"
CAP_LABELS = "data:labels"
CAP_GROUND_TRUTH = "data:ground_truth"


def available_capabilities(explanation: NlpExplanation, engine: InteractiveEngine | None) -> frozenset[str]:
    """Return the capability tokens the given explanation + engine satisfy.

    Parameters
    ----------
    explanation : NlpExplanation
        The immutable artifact, supplying the *data* capabilities — what the compiled batch
        contains, independent of whether a live model is still attached.
    engine : InteractiveEngine or None
        Live engine, or ``None`` for a snapshot (no live capabilities).

    Returns
    -------
    frozenset[str]
        Satisfied capability tokens (e.g. ``{"engine:predict", "engine:counterfactual",
        "model:gradients"}``).
    """
    caps: set[str] = set()
    # Data capabilities are read from the compiled batch, so they survive a snapshot — they sit
    # outside the engine guard below on purpose.
    if explanation.y_true is not None:
        caps.add(CAP_GROUND_TRUTH)
        if has_usable_probabilities(explanation.y_prob):
            caps.add(CAP_LABELS)
    if engine is not None:
        if engine.can_edit():
            caps.add(CAP_PREDICT)
        if engine.can_find_similar():
            caps.add(CAP_SIMILAR)
        if engine.can_counterfactual():
            caps.add(CAP_COUNTERFACTUAL)
            # Advertise gradients only when the *bound* generator actually operates on a
            # gradient-capable model — a forward-pass-only generator (AblationFlip) must not.
            generator = getattr(engine, "cf_generator", None)
            if has_capabilities(getattr(generator, "model", None), SupportsGradients):
                caps.add(CAP_GRADIENTS)
    return frozenset(caps)


class WebappComponent(ABC):
    """Base class for a self-contained, registrable webapp panel.

    Subclasses set ``id``/``name``/``scope``/``requires`` and implement :meth:`layout` and
    :meth:`register_callbacks`. All Dash ids a component creates must be namespaced with its ``id``
    to avoid collisions.
    """

    id: str = "component"
    name: str = "Component"
    scope: str = "local"  # "global" | "local"
    requires: frozenset[str] = frozenset()

    @classmethod
    def is_available(cls, explanation: NlpExplanation, engine: InteractiveEngine | None) -> bool:
        """Whether the component's ``requires`` are satisfied by the explanation + engine."""
        return cls.requires <= available_capabilities(explanation, engine)

    @abstractmethod
    def layout(self, explanation: NlpExplanation, engine: InteractiveEngine | None = None):
        """Return the Dash layout for this component.

        ``engine`` is provided for components whose initial UI depends on live capabilities — e.g. the
        counterfactual panel renders its config controls from the generator's spec, so they must exist
        in the initial layout rather than be injected by a later callback.
        """

    @abstractmethod
    def register_callbacks(self, app, explanation: NlpExplanation, engine: InteractiveEngine, stores: dict) -> None:
        """Register this component's Dash callbacks.

        Parameters
        ----------
        app : dash.Dash
            The Dash application.
        explanation : NlpExplanation
            The immutable artifact to read (never written to).
        engine : InteractiveEngine
            Live engine for prediction / counterfactual generation.
        stores : dict
            Shared ``dcc.Store`` ids the What-if Lab wires between components
            (e.g. ``{"apply": "whatif-apply-store"}``).
        """
