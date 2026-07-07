"""HuggingFace text-model adapters.

Two concrete :class:`~shapash.model.base.TextModel` implementations:

* :class:`HFPipelineModel` — wraps a ``transformers.pipeline`` and exposes **prediction only**
  (plus tokenization). This is the backward-compatible default: everything the existing prototype
  passes to ``NlpExplainer`` is a pipeline, so it is wrapped here transparently.
* :class:`HFClassifierModel` — wraps a raw ``AutoModelForSequenceClassification`` + tokenizer and
  implements **all** capabilities (tokenization, embeddings, gradients), which gradient-based
  counterfactual methods such as HotFlip need.

Roadmap (documented extension points, not implemented): ``SentenceTransformerModel`` (encoder +
classifier head) and ``TorchModel`` (hand-rolled ``nn.Module``) would subclass ``TextModel`` and the
relevant capability mixins the same way — generators and the webapp need no changes because they
depend on capabilities, not concrete classes.
"""

from __future__ import annotations

import warnings

import numpy as np

from shapash._optional import import_optional_module
from shapash.model.base import (
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
)

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'


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
        prediction (the labels it emits).
    """

    def __init__(self, pipeline, label_names: list[str] | None = None) -> None:
        super().__init__(label_names=label_names)
        self.pipeline = pipeline
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        self._order: list[str] | None = list(label_names) if label_names else None

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
            self._order = (
                list(self.label_names) if (self.label_names and set(self.label_names) <= set(labels)) else labels
            )
            if self.label_names is None:
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


class HFClassifierModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """Full-capability adapter over a raw ``AutoModelForSequenceClassification`` + tokenizer.

    Provides prediction, tokenization, the input-embedding table, mean-pooled sentence embeddings,
    and per-token gradients — everything gradient-based counterfactual generators require.

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
    """

    def __init__(
        self,
        classifier,
        tokenizer,
        label_names: list[str] | None = None,
        batch_size: int = 32,
        device: int | str | object | None = None,
    ) -> None:
        if label_names is None:
            id2label = getattr(getattr(classifier, "config", None), "id2label", None)
            if id2label:
                label_names = [id2label[i] for i in sorted(id2label)]
        super().__init__(label_names=label_names)
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self._pipeline = None  # lazily built for SHAP

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return ``(n_texts, n_classes)`` softmax probabilities in class-index order."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        device = self.classifier.device
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i : i + self.batch_size], padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self.classifier(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            out.append(probs.cpu().numpy())
        return np.vstack(out)

    def _resolve_pipeline_device(self):
        """Pick the pipeline device, degrading an unavailable-GPU request to CPU with a warning."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        if self.device is None:
            # Inherit whatever device the classifier already lives on (CPU-safe by construction).
            return self.classifier.device
        wants_cuda = (isinstance(self.device, str) and self.device.startswith("cuda")) or (
            isinstance(self.device, int) and self.device >= 0
        )
        if wants_cuda and not torch.cuda.is_available():
            warnings.warn(
                f"Requested device {self.device!r} but no CUDA GPU is available; "
                "falling back to CPU for the SHAP pipeline.",
                stacklevel=2,
            )
            return torch.device("cpu")
        return self.device

    @property
    def shap_callable(self):
        """A ``text-classification`` pipeline built from the classifier, for SHAP's TextMasker."""
        if self._pipeline is None:
            transformers = import_optional_module("transformers", extra=_NLP_EXTRA)
            self._pipeline = transformers.pipeline(
                "text-classification",
                model=self.classifier,
                tokenizer=self.tokenizer,
                top_k=None,
                device=self._resolve_pipeline_device(),
                batch_size=self.batch_size,
                truncation=True,
            )
        return self._pipeline

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Return the classifier tokenizer's tokens for ``text``."""
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from tokens (merges ``##`` word-pieces)."""
        return self.tokenizer.convert_tokens_to_string(tokens)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embedding_table(self) -> tuple[list[str], np.ndarray]:
        """Return ``(vocab, matrix)`` from the model's input-embedding layer."""
        weight = self.classifier.get_input_embeddings().weight
        matrix = weight.detach().cpu().numpy()
        vocab = self.tokenizer.convert_ids_to_tokens(range(matrix.shape[0]))
        return vocab, matrix

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return mean-pooled last-layer hidden states, shape ``(n_texts, hidden_dim)``.

        Padding tokens are excluded from the mean via the attention mask.
        """
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        device = self.classifier.device
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i : i + self.batch_size], padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                hidden = self.classifier(**batch, output_hidden_states=True).hidden_states[-1]
            mask = batch["attention_mask"].unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1)
            out.append(emb.cpu().numpy())
        return np.vstack(out)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def token_gradients(self, text: str, target_class: int) -> tuple[list[str], np.ndarray]:
        """Return ``(tokens, grads)`` of the target-class logit w.r.t. input embeddings.

        Special tokens (``[CLS]``/``[SEP]``/padding) are dropped so the returned tokens and gradient
        rows are aligned to substitutable content tokens only.
        """
        device = self.classifier.device  # torch is already present (classifier is a torch model)
        enc = self.tokenizer(text, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        embed_layer = self.classifier.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()

        self.classifier.zero_grad()
        logits = self.classifier(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        logits[0, target_class].backward()

        grads = inputs_embeds.grad[0].detach().cpu().numpy()  # (seq, hidden)
        ids = input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        special_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

        keep = [i for i, is_special in enumerate(special_mask) if not is_special]
        kept_tokens = [tokens[i] for i in keep]
        kept_grads = grads[keep]
        return kept_tokens, kept_grads
