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
    SupportsActivations,
    SupportsCaptumIG,
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
)

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# Pooled-representation module candidates for :attr:`HFClassifierModel.default_activation_layer`, most
# model-specific first: DistilBERT exposes ``pre_classifier``, BERT/DeBERTa a top-level ``pooler``. The
# first that exists on the classifier is the default similar-example decision-space; when none do (e.g.
# RoBERTa/XLM-R, whose head reads the raw ``<s>`` hidden state), we fall back to the sentinel below.
_POOLED_LAYER_CANDIDATES = ("pre_classifier", "pooler")
# Sentinel returned by ``default_activation_layer`` when the model has no pooled head module: it means
# "use the mean-pooled last hidden state" (what ``embed`` computes) — a representation every
# architecture provides, so similar-example retrieval works with no per-model configuration.
_LAST_HIDDEN_STATE = "__last_hidden_state__"


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


class HFClassifierModel(
    TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients, SupportsCaptumIG, SupportsActivations
):
    """Full-capability adapter over a raw ``AutoModelForSequenceClassification`` + tokenizer.

    Provides prediction, tokenization, the input-embedding table, mean-pooled sentence embeddings,
    per-token gradients, and the Captum layer-attribution surface — everything gradient-based
    counterfactual generators and ``LayerIntegratedGradients`` require.

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
    # Layer activations (similar-example retrieval)
    # ------------------------------------------------------------------

    @property
    def default_activation_layer(self) -> str:
        """The model's pooled decision-space representation, resolved per-architecture.

        Returns the first of ``pre_classifier`` (DistilBERT) / ``pooler`` (BERT, DeBERTa) that the
        classifier actually exposes — the pooled ``[CLS]`` vector the head reads, a good default
        decision-space for similar-example retrieval. When the model has no such module (e.g.
        RoBERTa/XLM-R), returns the :data:`_LAST_HIDDEN_STATE` sentinel, and :meth:`activations` falls
        back to the mean-pooled last hidden state (what :meth:`embed` computes) — a representation every
        architecture provides, so retrieval needs no per-model configuration. Override either by passing
        an explicit ``layer`` to :meth:`activations`.
        """
        modules = dict(self.classifier.named_modules())
        for name in _POOLED_LAYER_CANDIDATES:
            if name in modules:
                return name
        return _LAST_HIDDEN_STATE

    def activations(self, texts: list[str], layer: str | None = None) -> np.ndarray:
        """Return one activation vector per text at ``layer``, shape ``(n_texts, hidden_dim)``.

        A forward hook captures the named submodule's output on each batch. A pooled ``(batch, hidden)``
        output (e.g. ``pre_classifier``) is used as-is; a token-level ``(batch, seq, hidden)`` output
        (e.g. a transformer layer) is mean-pooled over non-padding tokens via the attention mask. The
        model runs under ``no_grad`` — this is inference only.
        """
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        layer_name = layer or self.default_activation_layer
        if layer_name == _LAST_HIDDEN_STATE:
            # No pooled head module on this architecture — use the universal mean-pooled last hidden
            # state (mask-aware), which embed() already computes. Keeps retrieval model-agnostic.
            return self.embed(list(texts))
        modules = dict(self.classifier.named_modules())
        if layer_name not in modules:
            raise KeyError(
                f"Layer {layer_name!r} not found on the model. Available top-level modules: "
                f"{[n for n, _ in self.classifier.named_children()]}."
            )
        module = modules[layer_name]

        captured: list = []
        handle = module.register_forward_hook(lambda _mod, _inp, out: captured.append(out.detach()))
        device = self.classifier.device
        texts = list(texts)
        out = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size], padding=True, truncation=True, return_tensors="pt"
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                captured.clear()
                with torch.no_grad():
                    self.classifier(**batch)
                act = captured[0]
                if act.dim() == 3:  # token-level layer output -> mask-aware mean pool to one vector
                    mask = batch["attention_mask"].unsqueeze(-1)
                    act = (act * mask).sum(1) / mask.sum(1)
                out.append(act.cpu().numpy())
        finally:
            handle.remove()
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

    # ------------------------------------------------------------------
    # Captum layer-attribution surface
    # ------------------------------------------------------------------

    @property
    def embedding_layer(self):
        """Return the word-embedding ``nn.Module`` (what ``LayerIntegratedGradients`` attributes through)."""
        return self.classifier.get_input_embeddings()

    def encode(self, text: str):
        """Return ``(input_ids, attention_mask, tokens)`` — batch-size-1 tensors + aligned token strings."""
        device = self.classifier.device
        enc = self.tokenizer(text, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return input_ids, attention_mask, tokens

    def reference_ids(self, input_ids):
        """Return baseline ids: content tokens replaced by the pad/mask reference, special tokens kept."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        ids = input_ids[0].tolist()
        special_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        ref_id = self.tokenizer.pad_token_id
        if ref_id is None:
            ref_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id or 0
        ref = [tid if is_special else ref_id for tid, is_special in zip(ids, special_mask, strict=True)]
        return torch.tensor([ref], device=input_ids.device)

    def logits(self, input_ids, attention_mask):
        """Return raw classification logits ``(batch, n_classes)`` (the Captum forward func)."""
        return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    def word_alignment(self, text: str) -> tuple[list[str], list[list[int]], list[int]] | None:
        """Group subwords into whole words via the fast tokenizer's ``word_ids()`` (see base ABC).

        Re-encodes ``text`` with the same ``truncation=True`` as :meth:`encode`, so the returned
        positions align with the token/attribution axis. ``word_ids()`` maps each subword to its word
        index (``None`` for special tokens), giving exact, scheme-independent grouping — no ``##`` / ``Ġ``
        / ``▁`` string-guessing. Display strings come from ``convert_tokens_to_string`` so each word is
        rebuilt correctly whatever the tokenization scheme. Returns ``None`` for a slow tokenizer (no
        ``word_ids()``), letting the caller fall back to a token-string heuristic.
        """
        if not getattr(self.tokenizer, "is_fast", False):
            return None
        enc = self.tokenizer(text, truncation=True)
        word_ids = enc.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

        # Special tokens (and any tokenless position) carry word_id None: fold their attribution into
        # the baseline. Content subwords sharing a word_id compose one word.
        special_positions = [i for i, wid in enumerate(word_ids) if wid is None]
        words: list[str] = []
        word_positions: list[list[int]] = []
        prev_wid: int | None = None
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_wid:
                word_positions.append([i])
                prev_wid = wid
            else:
                word_positions[-1].append(i)
        words = [self.tokenizer.convert_tokens_to_string([tokens[i] for i in pos]).strip() for pos in word_positions]
        return words, word_positions, special_positions
