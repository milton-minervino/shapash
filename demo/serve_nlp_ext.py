"""Serve the NLP explainer **with the What-if Lab** (live model, cached compute).

Unlike ``serve_nlp.py``, the What-if Lab needs a live model in memory — editing a
sample, re-predicting, and generating counterfactuals all call the real model, so a
model-free snapshot (``from_snapshot``) makes the lab self-disable. There is therefore
no way to serve what-if without loading the model.

What *can* be avoided on every restart is recomputing the expensive stuff around the
model. This script caches both to ``--cache-dir`` (keyed by a hash of the input texts):

* **SHAP contributions + predictions** — via the built-in ``compile(cache_dir=...)``.
* **The text embeddings and the 2-D projection** — via ``compute_projection(cache_dir=...)``. The
  library owns the embedding + caching (so the scatter and the similar-example neighbours are always
  in the same space); this script only supplies the reducer, PaCMAP.

The caching is automatic: each run **loads** the cache if one exists for these exact
texts, otherwise it **computes and writes** it. So the first run is slow; every run after
pays only the model load (a few seconds on CPU). Pass ``--recompute`` to force a fresh
compute (e.g. after changing the explainer). Iterating on the webapp itself? Don't
restart at all — keep this in a REPL and re-call ``run_app`` after reloading the webapp
modules; the model stays resident.

The model runs on CPU when no GPU is present, so this serves fine on a plain box.

Model and dataset are configurable (see ``ServeConfig``): swap ``--model-name`` for any HF
sequence-classification checkpoint (its label names are read from the model's own config) and
``--dataset-name`` for any HF ``datasets`` classification dataset (point ``--text-column`` /
``--label-column`` at the right fields if they aren't ``text`` / ``label``). The on-disk cache is
keyed by a hash of the input texts only (not the model), so each ``--model-name`` gets its own
subdirectory under ``--cache-dir`` — otherwise switching models with the same dataset/``--n``
would silently reload another model's cached contributions/embeddings.

Dataset and model label strings aren't guaranteed to match even when their class order does (e.g.
a dataset's ``ClassLabel`` names ``"neg"``/``"pos"`` vs. a model's ``config.id2label`` values
``"NEGATIVE"``/``"POSITIVE"``) — pass ``--label-map`` to rename ground-truth labels onto the
model's spelling so predicted vs. ground-truth comparisons in the webapp line up.

The sentence-highlight attribution method is selectable (a Captum-backed alternative to the default,
see the ``[nlp]`` extra):

* ``--attribution {shap,lig}`` — sentence-highlight method: ``shap`` (KernelSHAP, default) or ``lig``
  (Captum ``LayerIntegratedGradients``). The two are cached in **separate** subdirectories, so you can
  flip between them freely without ``--recompute``.

Both counterfactual generators are offered live in the What-if Lab — ``hotflip`` (gradient-based token
substitution) and ``ablation`` (Captum ``FeatureAblation`` token removal) — and are switched from a
**method dropdown in the webapp**, so there is no CLI flag for them. They run live and are not cached.

The on-disk cache mirrors these dependencies as a hierarchy under ``--cache-dir``:
``<model>/<dataset>__<split>/`` holds the (backend-independent) embedding-store artifacts — the
vectors and the projection derived from them — and a per-backend ``<...>/nlp_shap/`` or
``<...>/nlp_captum_lig/`` subdirectory holds that method's ``<hash>.pkl`` contributions.

Serving a *custom* model (external-head checkpoints)
----------------------------------------------------
Any standard ``AutoModelForSequenceClassification`` checkpoint loads automatically from ``--model-name``
alone (``load_model`` builds an ``HFClassifierModel``; label names come from ``config.id2label``). That
covers every certified encoder architecture — BERT, DistilBERT, RoBERTa, XLM-R, DeBERTa-v1, and a MiniLM
packaged as a sequence classifier.

A model whose classification **head is custom** cannot be introspected from its name — its layer
structure, weight-key convention, and input contract are non-standard (a sentence-transformer body + a
separate MLP head, a SetFit head, a hand-rolled ``nn.Module``, …). These are served through the
``MODEL_BUILDERS`` registry: a ``{--model-name: builder}`` map consulted *before* the generic path. To add
one:

1. Write a builder ``def build_x(config: ServeConfig, device: str) -> TextModel:`` that assembles the
   right adapter and returns it. Put **all** model-specific code inside it — the core ``load_model`` never
   changes. Pick the adapter by shape:

   * **sentence-transformer body + head** → :class:`~shapash.model.SentenceTransformerModel` (pass the
     ``sentence_transformers`` model + a *logits* head ``nn.Module``; drop any final ``softmax`` the head
     applies — ``predict`` softmaxes and the LIG backend needs logit space; set ``pool=`` to the body's
     pooling mode, usually ``"mean"``).
   * **raw encoder body + head** (head not fused into an ``AutoModelForSequenceClassification``) →
     :class:`~shapash.model.TorchClassifierModel` (pass ``body, head, tokenizer``; ``pool="cls"`` for a
     CLS-pooled head, ``"mean"`` otherwise).
   * **standard classifier that merely needs ``trust_remote_code``** → just call
     ``HFClassifierModel(AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True),
     tokenizer)``.

2. Register it: ``MODEL_BUILDERS["org/your-model"] = build_x``. Nothing else changes — caching, the
   What-if Lab, both attribution backends, counterfactuals, and similar-examples all work through the
   capability interface regardless of adapter.

``_build_shhossain_sentiment_model`` (registered for
``shhossain/all-MiniLM-L6-v2-sentiment-classifier``) is the worked example: a MiniLM ST body + a trained
``fc1 -> relu -> fc2 -> relu -> out`` MLP head, ``trust_remote_code`` with no HF tokenizer, so the generic
path can't touch it — but ``SentenceTransformerModel`` serves it faithfully (it reproduces the source
model's predictions exactly).

Usage
-----
    python demo/serve_nlp_ext.py [--n 100] [--cache-dir demo/nlp_ext_cache] [--port 8051]
    python demo/serve_nlp_ext.py --recompute   # ignore the cache and recompute
    python demo/serve_nlp_ext.py --attribution lig   # Captum LayerIntegratedGradients highlights
    python demo/serve_nlp_ext.py --model-name distilbert-base-uncased-finetuned-sst-2-english \\
        --dataset-name sst2 --dataset-split validation --text-column sentence
    python demo/serve_nlp_ext.py --model-name lvwerra/distilbert-imdb --dataset-name stanfordnlp/imdb \\
        --label-column label --label-map '{"neg": "NEGATIVE", "pos": "POSITIVE"}'
    # A registered custom (external-head) model — served via MODEL_BUILDERS, no other flags needed:
    python demo/serve_nlp_ext.py --model-name shhossain/all-MiniLM-L6-v2-sentiment-classifier

First run needs transformers, datasets and pacmap (the ``[nlp]`` extra). A ``SentenceTransformerModel``
custom builder additionally needs the ``sentence-transformers`` package.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pacmap
import torch
import transformers
from torch import nn

from shapash.backend import NlpCaptumLigBackend
from shapash.explainer.nlp_explainer import (
    NlpExplainer,
    _hash_texts,  # same keying as the compile cache
)
from shapash.model import HFClassifierModel, SentenceTransformerModel, TextModel

_HERE = Path(__file__).parent

logger = logging.getLogger("serve_nlp_ext")


@dataclass
class ServeConfig:
    """Everything needed to load data + model and serve the What-if webapp.

    A single object (rather than argparse's ``Namespace``) so the loading helpers below
    stay reusable from a REPL too — see the module docstring on iterating without restarting.
    """

    model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"
    dataset_name: str = "dair-ai/emotion"
    dataset_split: str = "test"
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, str] = field(default_factory=dict)
    n: int = 500
    attribution: str = "shap"  # sentence-highlight method: "shap" | "lig" (Captum LayerIntegratedGradients)
    # Similar-examples reference corpus: the split neighbours are retrieved from (the model's own
    # training split) and how many rows of it to bank. Set ``n_reference=0`` to disable the "Similar
    # Examples" panel.
    reference_split: str = "train"
    n_reference: int = 2000
    # Representation space the scatter projects *and* neighbours are ranked in — one setting for both,
    # so they can never disagree. ``None`` keeps whatever the model was built with ("decision").
    embedding_space: str | None = None
    cache_dir: Path = _HERE / "nlp_ext_cache"
    port: int = 8051
    host: str = "0.0.0.0"  # noqa: S104
    recompute: bool = False


# Short ``--attribution`` choice → the attribution backend's registered ``.name`` (used as the cache
# subdirectory *and* to pick the backend in ``build_backend``).
_ATTRIBUTION_BACKENDS = {"shap": "nlp_shap", "lig": "nlp_captum_lig"}


def _dataset_slug(config: ServeConfig) -> str:
    """Filesystem-safe ``<dataset>__<split>`` tag for the cache hierarchy."""
    return f"{config.dataset_name}__{config.dataset_split}".replace("/", "__")


def _dataset_cache_dir(config: ServeConfig) -> Path:
    """Return ``cache_dir/<model>/<dataset>__<split>`` — the level the projection is cached at.

    The compile/projection caches are keyed by a hash of the input texts only (see ``_hash_texts``),
    so different models or datasets would otherwise collide on the same cache file; the ``<model>`` and
    ``<dataset>`` path levels keep them apart. The PaCMAP projection embeds the texts with
    ``model.embed`` and never touches the attribution backend, so every ``--attribution`` choice shares
    one projection cache at *this* level (below it, contributions split per backend).
    """
    return config.cache_dir / config.model_name.replace("/", "__") / _dataset_slug(config)


def _compile_cache_dir(config: ServeConfig) -> Path:
    """Return ``<dataset-dir>/<backend>`` — the level contributions are cached at.

    Contributions *do* depend on the attribution backend, so SHAP and LIG get separate subdirectories.
    Switching ``--attribution`` then reads/writes a different ``<hash>.pkl`` instead of silently
    reusing the other method's cached highlights — no ``--recompute`` needed to swap methods.
    """
    return _dataset_cache_dir(config) / _ATTRIBUTION_BACKENDS[config.attribution]


def parse_args(argv: list[str] | None = None) -> ServeConfig:
    """Parse CLI args into a ``ServeConfig``, defaulting every flag from the dataclass."""
    defaults = ServeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", default=defaults.model_name, help="HF model id (label names are read from its config)."
    )
    parser.add_argument("--dataset-name", default=defaults.dataset_name, help="HF datasets id.")
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--text-column", default=defaults.text_column, help="Column holding the raw text.")
    parser.add_argument("--label-column", default=defaults.label_column, help="Column holding the ground-truth label.")
    parser.add_argument(
        "--label-map",
        type=json.loads,
        default=defaults.label_map,
        help=(
            "JSON object renaming dataset label strings to match the model's own label names, e.g. "
            '\'{"neg": "NEGATIVE", "pos": "POSITIVE"}\'. Needed when the dataset and model disagree on '
            "label spelling/casing even though their class order matches."
        ),
    )
    parser.add_argument("--n", type=int, default=defaults.n, help="Number of samples to load.")
    parser.add_argument(
        "--attribution",
        choices=list(_ATTRIBUTION_BACKENDS),
        default=defaults.attribution,
        help=(
            "Sentence-highlight attribution method: 'shap' (KernelSHAP, the default) or 'lig' "
            "(Captum LayerIntegratedGradients). Cached separately, so switching needs no --recompute."
        ),
    )
    parser.add_argument(
        "--reference-split",
        default=defaults.reference_split,
        help="Dataset split used as the similar-examples reference corpus (the model's training split).",
    )
    parser.add_argument(
        "--n-reference",
        type=int,
        default=defaults.n_reference,
        help="Reference rows to bank for 'Similar Examples' (0 disables the panel).",
    )
    parser.add_argument(
        "--embedding-space",
        default=defaults.embedding_space,
        help=(
            "Representation space for the 2-D scatter and similar-example retrieval: 'decision' "
            "(input to the final classification linear, class-discriminative), 'pooled' (the pooled "
            "last hidden state, semantic), or any backbone submodule name. Default: the model's own."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore any cached results and recompute SHAP + projection (overwrites the cache).",
    )
    args = parser.parse_args(argv)
    return ServeConfig(**vars(args))


def _load_split(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """Load the first ``n`` ``(texts, label_strings)`` of ``split`` from the configured dataset.

    Ground-truth label strings come from the dataset's own ``ClassLabel`` feature names when
    present (the usual shape for HF classification datasets); otherwise the raw values are
    stringified as-is. ``config.label_map`` then renames them to match the model's label
    names when the two disagree on spelling/casing (e.g. dataset ``"neg"`` vs. model
    ``"NEGATIVE"``) despite sharing the same class order.
    """
    dataset = datasets.load_dataset(config.dataset_name, split=split)
    sentences = dataset[config.text_column][:n]
    raw_labels = dataset[config.label_column][:n]
    names = getattr(dataset.features.get(config.label_column), "names", None)
    labels = [names[i] for i in raw_labels] if names is not None else [str(v) for v in raw_labels]
    if config.label_map:
        labels = [config.label_map.get(v, v) for v in labels]
    logger.info("Loaded %d samples from %s (split=%s)", len(sentences), config.dataset_name, split)
    return sentences, labels


def load_data(config: ServeConfig) -> tuple[list[str], list[str]]:
    """Load the served ``config.n`` samples + label strings from ``config.dataset_split``."""
    return _load_split(config, config.dataset_split, config.n)


def load_reference_corpus(config: ServeConfig) -> tuple[list[str], list[str]] | None:
    """Load the similar-examples reference corpus (the model's training split).

    Returns ``(texts, labels)`` from ``config.reference_split`` for the "Similar Examples" panel,
    or ``None`` when disabled (``n_reference <= 0``). This is the pool neighbours are retrieved from,
    so it is the model's *training* set rather than the served (test) split.
    """
    if config.n_reference <= 0:
        return None
    return _load_split(config, config.reference_split, config.n_reference)


def _load_tokenizer(model_name: str):
    """Load the tokenizer, preferring a fast one but falling back to the slow implementation.

    A fast tokenizer is best — it unlocks the exact ``word_ids()`` word-alignment the LIG highlights
    use — but some checkpoints ship no ``tokenizer.json``, so the fast load raises. We then retry with
    ``use_fast=False``; a slow tokenizer still works (the LIG backend degrades to its scheme-aware
    string merge). When *both* fail, the checkpoint has no standard HF tokenizer at all — typically a
    custom ``trust_remote_code`` architecture — which the ``HFClassifierModel`` adapter does not support;
    surface that as a clear error rather than a deep tokenizer stack trace.
    """
    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as fast_err:  # noqa: BLE001 — retry slow, then re-raise with a usable hint
        logger.warning(
            "Fast tokenizer unavailable for %s (%s) — retrying with use_fast=False.",
            model_name,
            type(fast_err).__name__,
        )
        try:
            return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as slow_err:  # noqa: BLE001 — no standard tokenizer; give an actionable message
            raise RuntimeError(
                f"Could not load a tokenizer for {model_name!r} in fast or slow mode. This is usually a "
                "checkpoint with a non-standard or custom (trust_remote_code) tokenizer/architecture, "
                "which the HFClassifierModel adapter does not support. Use a standard "
                "AutoModelForSequenceClassification checkpoint that ships a normal HF tokenizer."
            ) from slow_err


def _load_hf_classifier(config: ServeConfig, device: str) -> HFClassifierModel:
    """The generic default: any standard ``AutoModelForSequenceClassification`` checkpoint.

    Covers every certified encoder architecture (BERT/DistilBERT/RoBERTa/XLM-R/DeBERTa-v1, and MiniLM
    packaged as a sequence classifier). Label names come from the model's own ``config.id2label``, so a
    new ``--model-name`` brings its own classes with no extra flags.
    """
    tokenizer = _load_tokenizer(config.model_name)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_name).to(device)
    logger.info("Loaded %s as HFClassifierModel on device: %s", config.model_name, next(classifier.parameters()).device)
    return HFClassifierModel(classifier, tokenizer)


# Reconcile the shhossain checkpoint's own class label spelling with the dair-ai/emotion dataset it is
# demoed against ("sad" -> "sadness"); every other class name already matches.
_SHHOSSAIN_LABEL_ALIASES = {"sad": "sadness"}


# ── Custom-model builders ──────────────────────────────────────────────────────────────────────────
# Standard checkpoints load generically above. A model whose classification *head* is custom (its layer
# structure / weight keys / IO contract are not a standard, so they can't be introspected from a name
# alone) needs a per-model builder. Register one function per such checkpoint below; the generic path is
# never touched. Each builder takes (config, device) and returns any TextModel adapter.


def _build_shhossain_sentiment_model(config: ServeConfig, device: str) -> SentenceTransformerModel:
    """Example builder — ``shhossain/all-MiniLM-L6-v2-sentiment-classifier`` (ST body + MLP head).

    This checkpoint is a custom ``trust_remote_code`` model: a standard ``all-MiniLM-L6-v2`` body plus a
    trained ``fc1 -> relu -> fc2 -> relu -> out -> softmax`` MLP head, and it ships no HF tokenizer — so the
    generic HF path cannot load it. Everything model-specific (the head's weight-key convention, dropping
    its final softmax to recover logits, mean pooling) is contained *here*, not in the core loader.
    """
    huggingface_hub = __import__("huggingface_hub")
    sentence_transformers = __import__("sentence_transformers")
    safetensors_torch = __import__("safetensors.torch", fromlist=["load_file"])

    with open(huggingface_hub.hf_hub_download(config.model_name, "config.json")) as fh:
        cfg = json.load(fh)
    class_map = cfg["class_map"]
    labels = [class_map[str(i)] for i in range(len(class_map))]
    # This checkpoint's class_map spells one emotion as "sad" while its natural demo dataset
    # (dair-ai/emotion) names the same class "sadness"; align them so predicted-vs-ground-truth
    # comparisons in the webapp line up. (Same intent as the --label-map flag, but done here since
    # the mismatch is intrinsic to this model's own labels, not a dataset choice.)
    labels = [_SHHOSSAIN_LABEL_ALIASES.get(label, label) for label in labels]

    st_model = sentence_transformers.SentenceTransformer(cfg["embedding_model"], device=device)
    weights = safetensors_torch.load_file(huggingface_hub.hf_hub_download(config.model_name, "model.safetensors"))
    # Rebuild the trained MLP as a *logits* head (sizes from weight shapes; the checkpoint's final softmax
    # is dropped — SentenceTransformerModel.predict softmaxes and the LIG backend needs logit space).
    head = nn.Sequential(
        nn.Linear(weights["fc1.weight"].shape[1], weights["fc1.weight"].shape[0]),
        nn.ReLU(),
        nn.Linear(weights["fc2.weight"].shape[1], weights["fc2.weight"].shape[0]),
        nn.ReLU(),
        nn.Linear(weights["out.weight"].shape[1], weights["out.weight"].shape[0]),
    )
    for idx, name in ((0, "fc1"), (2, "fc2"), (4, "out")):
        head[idx].load_state_dict({"weight": weights[f"{name}.weight"], "bias": weights[f"{name}.bias"]})
    return SentenceTransformerModel(st_model, head.to(device).eval(), label_names=labels, pool="mean")


# ``--model-name`` -> a custom builder for a checkpoint whose head can't be introspected from its name.
# Anything not listed takes the generic ``_load_hf_classifier`` path. To serve a new custom model, add a
# ``build(config, device) -> TextModel`` function above and one entry here — the core loader is untouched.
# See the module docstring's "Serving a custom model" section for the per-adapter recipe.
MODEL_BUILDERS: dict[str, Callable[[ServeConfig, str], TextModel]] = {
    "shhossain/all-MiniLM-L6-v2-sentiment-classifier": _build_shhossain_sentiment_model,
}


def load_model(config: ServeConfig) -> TextModel:
    """Load the model on the best available device: a registered custom builder, else the generic HF path.

    Standard ``AutoModelForSequenceClassification`` checkpoints of any certified architecture load with no
    configuration; a checkpoint whose head is custom (so it can't be introspected from its name) is built
    by its entry in :data:`MODEL_BUILDERS`.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA device found — using CPU")

    builder = MODEL_BUILDERS.get(config.model_name)
    return builder(config, device) if builder is not None else _load_hf_classifier(config, device)


def build_backend(config: ServeConfig, model: TextModel) -> NlpCaptumLigBackend | None:
    """Return the attribution backend selected by ``--attribution``.

    Returns ``None`` for ``"shap"`` so ``NlpExplainer`` builds its default ``NlpShapBackend`` (which
    needs the model's ``shap_callable`` wiring); ``"lig"`` returns an explicit ``NlpCaptumLigBackend``.
    """
    if config.attribution == "lig":
        # LIG runs one integration per class per sample — show a progress bar over the batch.
        return NlpCaptumLigBackend(model, label_names=model.label_names, show_progress=True)
    return None


def build_reducer() -> pacmap.PaCMAP:
    """The dimensionality reducer for the scatter — a *demo* choice, not a library one.

    ``NlpExplainer.compute_projection`` owns everything that must stay consistent (which space the
    texts are embedded in, and the caching of both the vectors and the coordinates) and takes the
    reducer as an argument, because which manifold method suits your data is a modelling decision.
    PaCMAP is a good default for text clusters; swap in UMAP, t-SNE, or drop the argument entirely for
    the built-in PCA.
    """
    return pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)


def main() -> None:
    """Load config + data + model, (re)use the SHAP + projection cache, and serve the What-if webapp."""
    config = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # Quiet noisy third-party INFO logs (HF Hub freshness checks, download chatter) so the
    # device/cache messages below stand out.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sentences, y_true = load_data(config)
    dataset_dir = _dataset_cache_dir(config)  # projection lives here (shared across attribution backends)
    compile_dir = _compile_cache_dir(config)  # contributions live here (one subdir per attribution backend)

    text_hash = _hash_texts(list(sentences))
    if config.recompute:
        # Drop this text set's cached artifacts so the steps below recompute + overwrite. Only the
        # selected backend's contributions are dropped; other backends' caches are left intact.
        logger.info("--recompute: dropping cached %s contributions + projection for these texts", config.attribution)
        (compile_dir / f"{text_hash}.pkl").unlink(missing_ok=True)
        # The embeddings + projection are dropped by compute_projection(recompute=True) below, which
        # knows their key; only the contributions pickle is this script's to manage.

    model = load_model(config)
    if config.embedding_space is not None:
        # One setting drives both the scatter projection and similar-example retrieval. Assigning it
        # here (rather than threading it through every MODEL_BUILDERS entry) is safe because the
        # setter validates the name against the backbone — a typo raises now, not mid-forward-pass.
        model.embedding_space = config.embedding_space
    logger.info("Embedding space: %s (scatter projection + similar examples)", model.resolve_space())

    # Similar-examples reference corpus (the training split) + a cache dir for its embedding bank,
    # keyed by corpus/space/model so it is embedded once and reloaded on later runs.
    reference_corpus = load_reference_corpus(config)
    similar_cache_dir = dataset_dir / "similar"

    # No cf_generator is passed: NlpExplainer auto-discovers every built-in compatible with the model
    # (HotFlip + AblationFlip on an HF classifier) and offers them from the What-if Lab's method dropdown.
    # reference_corpus enables the "Similar Examples" panel (skipped when --n-reference 0).
    xpl = NlpExplainer(
        model,
        label_names=model.label_names,
        backend=build_backend(config, model),
        reference_corpus=reference_corpus,
        reference_cache_dir=similar_cache_dir,
    )
    # Both steps load from cache if one exists for these exact texts, otherwise they compute and write
    # it. So the first run is slow; later runs only pay the model load.
    compile_cache = compile_dir / f"{text_hash}.pkl"
    if compile_cache.exists():
        logger.info("Contributions cache hit (%s) — loading %s", config.attribution, compile_cache)
    else:
        logger.info(
            "Contributions cache miss (%s) — computing for %d texts (this is the slow part)",
            config.attribution,
            len(sentences),
        )
    xpl.compile(sentences, y_true=y_true, cache_dir=compile_dir)
    projected = xpl.compute_projection(reducer=build_reducer(), cache_dir=dataset_dir, recompute=config.recompute)

    if xpl.can_find_similar():
        # Build (and cache) the reference activation bank now so the first in-app lookup is instant.
        logger.info("Warming similar-examples bank over %d reference texts…", len(reference_corpus[0]))
        xpl.find_similar(sentences[0], top_k=1)

    logger.info(
        "attribution=%s | counterfactual=%s | can_edit=%s | can_counterfactual=%s | can_find_similar=%s",
        config.attribution,
        ",".join(name for name, _ in xpl.available_cf_generators()) or "none",
        xpl.can_edit(),
        xpl.can_counterfactual(),
        xpl.can_find_similar(),
    )
    logger.info("Serving on http://%s:%d (Ctrl+C to stop)", config.host, config.port)
    xpl.run_app(port=config.port, debug=False, host=config.host, scatter_xy=projected)


if __name__ == "__main__":
    main()
