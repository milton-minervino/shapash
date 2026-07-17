"""Model capability layer — modality-agnostic adapters for the shapash engine.

This package introduces a first-class ``Model`` abstraction (a peer to the planned
``shapash/data/`` layer). Explanation backends and counterfactual generators depend on
*declared capabilities* rather than a concrete model type, so a component can call
``is_compatible(model)`` and only run when the required capabilities are present.

The design deliberately adopts the *intent* of Google PAIR LIT's ``Model`` API — components
stay model-agnostic by introspecting what a model can do — but expresses it Pythonically with
abstract base classes and capability mixins (``SupportsTokenization`` / ``SupportsEmbeddings`` /
``SupportsGradients``) instead of LIT's runtime ``JsonDict``/``LitType`` specs. This matches the
typed-dataclass style already used elsewhere in shapash (e.g. ``NlpContributions``).

Currently ships text adapters for HuggingFace models. ``SentenceTransformerModel`` and
``TorchModel`` are documented extension points (see ``hf.py`` roadmap notes).
"""

from __future__ import annotations

from shapash.model.base import (
    SupportsActivations,
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
    has_capabilities,
)
from shapash.model.hf import HFClassifierModel, HFPipelineModel

__all__ = [
    "TextModel",
    "SupportsTokenization",
    "SupportsEmbeddings",
    "SupportsGradients",
    "SupportsActivations",
    "has_capabilities",
    "HFPipelineModel",
    "HFClassifierModel",
]
