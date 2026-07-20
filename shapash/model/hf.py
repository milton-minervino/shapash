"""HuggingFace text-model adapters.

Two concrete :class:`~shapash.model.base.TextModel` implementations:

* :class:`HFPipelineModel` — wraps a ``transformers.pipeline`` and exposes **prediction only**
  (plus tokenization). This is the backward-compatible default: everything the existing prototype
  passes to ``NlpExplainer`` is a pipeline, so it is wrapped here transparently.
* :class:`HFClassifierModel` — wraps a raw ``AutoModelForSequenceClassification`` + tokenizer and
  implements **all** capabilities, which gradient-based counterfactual methods (HotFlip) and the
  Captum LIG backend need. It is a thin preset over
  :class:`~shapash.model.encoder.EncoderClassifierModel`: the classifier itself is the backbone
  (pooling and head are internal to it), so no fusing is required.

For models whose encoder body and classification head are *separate* modules (sentence-transformers,
hand-rolled PyTorch classifiers), see :class:`~shapash.model.torch_models.SentenceTransformerModel` and
:class:`~shapash.model.torch_models.TorchClassifierModel` — they share the same capability spine via a
fused backbone. Generators and the webapp need no changes because everything depends on *capabilities*,
not concrete classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shapash.model.base import (
    SupportsTokenization,
    TextModel,
)
from shapash.model.encoder import _DECISION_SPACE, EncoderClassifierModel


def _probs_from_pipeline_output(raw: list, order: list[str]) -> np.ndarray:
    """Convert ``pipeline`` output to a probability matrix aligned to ``order``.

    Handles both ``top_k=None`` output (list of per-class dict lists) and single-prediction output
    (list of dicts). Aligning by label name makes the result robust to per-row score sorting.
    """
    rows = raw if raw and isinstance(raw[0], list) else [[r] for r in raw]
    matrix = np.zeros((len(rows), len(order)), dtype=np.float64)
    for i, row in enumerate(rows):
        scores = {d["label"]: float(d["score"]) for d in row}
        for j, label in enumerate(order):
            matrix[i, j] = scores.get(label, 0.0)
    return matrix


class HFPipelineModel(TextModel, SupportsTokenization):
    """Prediction-only adapter over a HuggingFace ``text-classification`` pipeline.

    Parameters
    ----------
    pipeline : transformers.Pipeline
        A ``text-classification`` pipeline. ``top_k=None`` is requested at call time so all class
        scores are returned.
    label_names : list[str] or None
        Class names in output-column order. When ``None``, inferred from the pipeline's first
        prediction (the labels it emits). When given, validated against those labels on the first
        prediction — a mismatched set raises ``ValueError`` rather than silently misaligning columns.
    """

    def __init__(self, pipeline, label_names: list[str] | None = None) -> None:
        super().__init__(label_names=label_names)
        self.pipeline = pipeline
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        self._order: list[str] | None = None

    @property
    def shap_callable(self):
        """Callable SHAP's text explainer can wrap directly (the pipeline itself)."""
        return self.pipeline

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return ``(n_texts, n_classes)`` probabilities via the pipeline."""
        raw = self.pipeline(list(texts), top_k=None)
        if self._order is None:
            first = raw[0] if raw and isinstance(raw[0], list) else raw
            labels = [d["label"] for d in first]
            if self.label_names is not None:
                if set(self.label_names) != set(labels):
                    raise ValueError(
                        f"label_names {self.label_names!r} do not match the labels the pipeline "
                        f"emits {labels!r}; pass the exact label set (any order), or leave "
                        "label_names=None to infer it from the pipeline."
                    )
                self._order = list(self.label_names)
            else:
                self._order = labels
                self.label_names = list(self._order)
        return _probs_from_pipeline_output(raw, self._order)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize with the pipeline's tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Pipeline has no tokenizer; cannot tokenize.")
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Pipeline has no tokenizer; cannot detokenize.")
        return self.tokenizer.convert_tokens_to_string(tokens)


class HFClassifierModel(EncoderClassifierModel):
    """Full-capability adapter over a raw ``AutoModelForSequenceClassification`` + tokenizer.

    A thin preset over :class:`~shapash.model.encoder.EncoderClassifierModel`: the classifier *is* the
    backbone (it maps ids/embeds to ``.logits`` with pooling and head internal), so this class only
    resolves ``label_names`` from the model config and delegates the entire capability surface —
    prediction, tokenization, the input-embedding table, sentence embeddings in any representation
    space, per-token gradients and the Captum layer-attribution surface.

    Parameters
    ----------
    classifier : transformers.PreTrainedModel
        A sequence-classification model whose ``forward`` accepts ``inputs_embeds`` and supports
        ``output_hidden_states=True``.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching the classifier.
    label_names : list[str] or None
        Class names in class-index (id) order. When ``None``, read from ``classifier.config.id2label``.
    batch_size : int, optional
        Batch size for ``predict``, ``embed`` and the SHAP pipeline. Default 32.
    device : int or str or torch.device or None, optional
        Device for the SHAP pipeline (e.g. ``0``, ``"cuda"``, ``"cpu"``). When ``None`` (default),
        the pipeline inherits the device the classifier already lives on, so CPU-only environments
        work with no configuration. An explicit CUDA request is silently downgraded to CPU (with a
        warning) when no GPU is available.
    embedding_space : str, optional
        Representation :meth:`embed` returns (see :class:`~shapash.model.encoder.EncoderClassifierModel`).
        Default ``"decision"`` (input to the final classification linear); ``"pooled"`` restores the
        historical mask-pooled last hidden state.
    pool : {"mean", "cls", "max"} or callable, optional
        How a token-level output is reduced to one vector per text in :meth:`embed`. Default ``"mean"``.
        Note this affects the *analysis* representation only — an ``AutoModelForSequenceClassification``
        does its own pooling internally, so :meth:`predict` is unaffected.
    """

    def __init__(
        self,
        classifier,
        tokenizer,
        label_names: list[str] | None = None,
        batch_size: int = 32,
        device: int | str | object | None = None,
        *,
        embedding_space: str = _DECISION_SPACE,
        pool: Any = "mean",
    ) -> None:
        if label_names is None:
            id2label = getattr(getattr(classifier, "config", None), "id2label", None)
            if id2label:
                label_names = [id2label[i] for i in sorted(id2label)]
        super().__init__(
            classifier,
            tokenizer,
            label_names=label_names,
            batch_size=batch_size,
            device=device,
            pool=pool,
            embedding_space=embedding_space,
        )
        # ``classifier`` is the backbone; keep the historical alias for introspection and tests.
        self.classifier = classifier
