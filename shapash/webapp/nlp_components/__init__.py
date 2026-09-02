"""Composable webapp components for the NLP what-if tools.

Prototype of the master plan's Phase-5b ``WebappComponent`` contract, applied first to the
interactive what-if panels (data editor + counterfactuals). Each component owns its layout and
callbacks and declares its ``requires`` capabilities, self-disabling when the bound explanation/engine
cannot satisfy them.

Notably this **extends** the Phase-5b ``requires`` idea from *data* capabilities (e.g.
``proba_values``) to *engine/model* capabilities (``engine:predict``, ``engine:counterfactual``),
so a panel can require a live, gradient-capable model — the refinement to fold back into the master
contract.
"""

from __future__ import annotations

from shapash.webapp.nlp_components.base import WebappComponent, available_capabilities
from shapash.webapp.nlp_components.counterfactual import CounterfactualComponent
from shapash.webapp.nlp_components.data_editor import DataEditorComponent
from shapash.webapp.nlp_components.datapoint import (
    datapoint_from_contributions,
    pack_datapoint,
    unpack_datapoint,
)
from shapash.webapp.nlp_components.error_analysis import ErrorAnalysisComponent
from shapash.webapp.nlp_components.label_noise import LabelNoiseComponent
from shapash.webapp.nlp_components.sentence_highlight import SentenceHighlightComponent
from shapash.webapp.nlp_components.similar_examples import SimilarExamplesComponent
from shapash.webapp.nlp_components.waterfall import WaterfallComponent

__all__ = [
    "WebappComponent",
    "available_capabilities",
    "DataEditorComponent",
    "CounterfactualComponent",
    "ErrorAnalysisComponent",
    "LabelNoiseComponent",
    "SentenceHighlightComponent",
    "SimilarExamplesComponent",
    "WaterfallComponent",
    "pack_datapoint",
    "unpack_datapoint",
    "datapoint_from_contributions",
]
