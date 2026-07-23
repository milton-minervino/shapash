"""Serve the NLP explainer **with the What-if Lab** (live model, cached compute).

The What-if Lab needs a live model in memory — editing a sample, re-predicting, and generating
counterfactuals all call the real model, so a model-free snapshot (``from_snapshot``) makes the
lab self-disable. There is therefore no way to serve what-if without loading the model.

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
``--label-column`` at the right fields if they aren't ``text`` / ``label``). Cache keys cover the model
and the attribution backend as well as the texts, so switching ``--model-name`` can never reload another
model's contributions; the per-model/per-dataset subdirectories under ``--cache-dir`` below are for
human legibility (and easy pruning), not for correctness.

Dataset and model label strings aren't guaranteed to match even when their class order does (e.g.
a dataset's ``ClassLabel`` names ``"neg"``/``"pos"`` vs. a model's ``config.id2label`` values
``"NEGATIVE"``/``"POSITIVE"``) — pass ``--label-map`` to rename ground-truth labels onto the
model's spelling so predicted vs. ground-truth comparisons in the webapp line up. See "Serving a custom
dataset" below for label spaces ``--label-map`` can't fix (bucketing, dropping rows), and ``--seed`` for
how rows are sampled.

The sentence-highlight attribution method is selectable (a Captum-backed alternative to the default,
see the ``[nlp]`` extra):

* ``--attribution {shap,lig}`` — sentence-highlight method: ``shap`` (KernelSHAP, default) or ``lig``
  (Captum ``LayerIntegratedGradients``). The two are cached in **separate** subdirectories, so you can
  flip between them freely without ``--recompute``.
* ``--lig-batch-size`` — only used by ``lig``: Captum's ``internal_batch_size``, chunking each sample's
  50-step integration instead of running it through the model in one shot. Lower it (e.g. ``2`` or
  ``1``) if you hit a CUDA out-of-memory error, especially on memory-hungry architectures like DeBERTa.

``lig`` is also the method most sensitive to *truncation* actually being configured — see
``_hf_classifier_model`` for why every HF classifier here resolves an explicit ``max_length``.

Both counterfactual generators are offered live in the What-if Lab — ``hotflip`` (gradient-based token
substitution) and ``ablation`` (leave-one-out token removal) — and are switched from a
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
   * **standard classifier whose config's labels can't be trusted** (missing ``id2label``, or a
     ``label2id`` that turns out backwards from the model's real behaviour) → same call as the generic
     path, but pass ``label_names=[...]`` explicitly instead of leaving it to be read from
     ``config.id2label``. Verify the ordering against a real labeled sample (dozens+, not a handful) —
     ``almanach/camembertv2-base-cls`` was first "fixed" from a 10-example hand check that (wrongly)
     looked decisive; a proper sanity check (``demo/check_camembertv2_amazon_fr.py``) later showed
     *neither* ordering is trustworthy on that checkpoint at all. A handful of examples can't tell a
     genuinely swapped label from a model that just doesn't work.

2. Register it: ``MODEL_BUILDERS["org/your-model"] = build_x``. Nothing else changes — caching, the
   What-if Lab, both attribution backends, counterfactuals, and similar-examples all work through the
   capability interface regardless of adapter.

``_build_shhossain_sentiment_model`` (registered for
``shhossain/all-MiniLM-L6-v2-sentiment-classifier``) is the worked example: a MiniLM ST body + a trained
``fc1 -> relu -> fc2 -> relu -> out`` MLP head, ``trust_remote_code`` with no HF tokenizer, so the generic
path can't touch it — but ``SentenceTransformerModel`` serves it faithfully (it reproduces the source
model's predictions exactly). ``_build_camembertv2_cls_model`` (registered for
``almanach/camembertv2-base-cls``) is the worked example of the label-names-can't-be-trusted case: an
architecturally plain ``RobertaForSequenceClassification`` whose checkpoint config omits ``id2label``.
**Its config's labels can't be trusted, but neither can the checkpoint's own predictions** — see the
builder's docstring and ``demo/check_camembertv2_amazon_fr.py`` for a sanity check showing neither label
ordering beats a trivial majority-class baseline on gold in-domain data. It stays registered only as a
worked example of this builder pattern, not as a model whose predictions should be trusted.

Serving a *custom* dataset (label logic that isn't a 1:1 rename)
------------------------------------------------------------------
Any standard HF classification dataset loads automatically from ``--dataset-name`` alone: point
``--text-column``/``--label-column`` at the right fields if they aren't ``text``/``label``, and (rows are
shuffled with ``--seed`` before ``--n``/``--n-reference`` are taken, so this is representative even when
the dataset's on-disk order groups rows by label). Ground-truth strings come from the dataset's own
``ClassLabel`` names when present, else the raw values stringified; ``--label-map`` renames them 1:1 onto
the model's own spelling when the two disagree on casing/wording despite sharing the same class order
(e.g. dataset ``"neg"`` vs. model ``"NEGATIVE"``).

A dataset whose ground truth needs **more** than a rename — bucketing several raw values into one class,
dropping rows with no mapping, or any other dataset-specific logic — is served through the
``DATASET_LOADERS`` registry: a ``{--dataset-name: loader}`` map consulted *before* the generic path
(the exact ``MODEL_BUILDERS`` pattern above, applied to data instead of the model). To add one:

1. Write a loader ``def load_x(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]``
   that returns ``(texts, label_strings)``. Put **all** dataset-specific code inside it, including which
   raw columns it reads — a bucketing loader's logic is tied to that dataset's exact label semantics, so
   there is no generic ``--text-column``/``--label-column`` to thread through; the core ``_load_split``
   never changes.
2. Register it: ``DATASET_LOADERS["org/your-dataset"] = load_x``. Nothing else changes — caching, the
   webapp, and the reference corpus all call ``_load_split`` regardless of which path served it.

``_load_amazon_reviews_multi_binary`` (registered for ``mteb/amazon_reviews_multi``) is the worked
example: its ``label`` is a 5-star rating minus one (0-4), not the binary sentiment space a binary
classifier like ``almanach/camembertv2-base-cls`` predicts into, so it buckets rating > 3 -> ``"positive"``,
rating < 3 -> ``"negative"``, and drops the neutral 3-star (a ground-truth convention only — see
``demo/check_camembertv2_amazon_fr.py`` for whether this specific model's predictions are trustworthy).

Usage
-----
    python demo/serve_nlp.py [--n 100] [--cache-dir demo/nlp_cache] [--port 8050]
    python demo/serve_nlp.py --recompute   # ignore the cache and recompute
    python demo/serve_nlp.py --attribution lig   # Captum LayerIntegratedGradients highlights
    python demo/serve_nlp.py --model-name distilbert-base-uncased-finetuned-sst-2-english \\
        --dataset-name sst2 --dataset-split validation --text-column sentence
    # A registered custom model against a registered dataset loader (bucketed 5-star -> binary
    # sentiment ground truth; --text-column/--label-column don't apply here, see DATASET_LOADERS):
    python demo/serve_nlp.py --model-name almanach/camembertv2-base-cls \\
        --dataset-name mteb/amazon_reviews_multi --dataset-config fr --dataset-split test
    python demo/serve_nlp.py --model-name lvwerra/distilbert-imdb --dataset-name stanfordnlp/imdb \\
        --label-column label --label-map '{"neg": "NEGATIVE", "pos": "POSITIVE"}'
    # A registered custom (external-head) model — served via MODEL_BUILDERS, no other flags needed:
    python demo/serve_nlp.py --model-name shhossain/all-MiniLM-L6-v2-sentiment-classifier

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
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.model import HFClassifierModel, SentenceTransformerModel, TextModel

_HERE = Path(__file__).parent

logger = logging.getLogger("serve_nlp")


@dataclass
class ServeConfig:
    """Everything needed to load data + model and serve the What-if webapp.

    A single object (rather than argparse's ``Namespace``) so the loading helpers below
    stay reusable from a REPL too — see the module docstring on iterating without restarting.
    """

    model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"
    dataset_name: str = "dair-ai/emotion"
    dataset_config: str | None = None  # HF dataset config/subset (e.g. a language); None = dataset's default
    dataset_split: str = "test"
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, str] = field(default_factory=dict)
    n: int = 500
    # Rows are shuffled with this seed before taking --n / --n-reference (see _load_split) — several
    # HF dataset repos (e.g. mteb/amazon_reviews_multi) store rows grouped by label, so an unshuffled
    # "first n" slice can be entirely one class.
    seed: int = 0
    attribution: str = "shap"  # sentence-highlight method: "shap" | "lig" (Captum LayerIntegratedGradients)
    # Captum's LayerIntegratedGradients expands one sample into ``n_steps`` (50) scaled copies and runs
    # them through the model as a single batch unless told otherwise — on a memory-hungry architecture
    # (e.g. DeBERTa-v2/v3's disentangled attention) that batch of 50 can blow past a small GPU's memory.
    # ``internal_batch_size`` makes Captum chunk that batch instead; lower it further if you still OOM.
    lig_batch_size: int = 8
    # Similar-examples reference corpus: the split neighbours are retrieved from (the model's own
    # training split) and how many rows of it to bank. Set ``n_reference=0`` to disable the "Similar
    # Examples" panel.
    reference_split: str = "train"
    n_reference: int = 2000
    # Representation space the scatter projects *and* neighbours are ranked in — one setting for both,
    # so they can never disagree. ``None`` keeps whatever the model was built with ("decision").
    embedding_space: str | None = None
    cache_dir: Path = _HERE / "nlp_cache"
    port: int = 8050
    host: str = "0.0.0.0"  # noqa: S104
    recompute: bool = False


# Short ``--attribution`` choice → the attribution backend's registered ``.name`` (used as the cache
# subdirectory *and* to pick the backend in ``build_backend``).
_ATTRIBUTION_BACKENDS = {"shap": "nlp_shap", "lig": "nlp_captum_lig"}


def _dataset_slug(config: ServeConfig) -> str:
    """Filesystem-safe ``<dataset>__[<config>__]<split>`` tag for the cache hierarchy.

    The config segment is only included when ``--dataset-config`` is actually set, so existing cache
    directories built before that flag existed (``<dataset>__<split>``, no config with no HF dataset
    config) still resolve to the same path.
    """
    parts = [config.dataset_name, *([config.dataset_config] if config.dataset_config else []), config.dataset_split]
    return "__".join(parts).replace("/", "__")


def _dataset_cache_dir(config: ServeConfig) -> Path:
    """Return ``cache_dir/<model>/<dataset>__<split>`` — the level the projection is cached at.

    The compile and projection caches both key on the model (and, for contributions, the backend), so
    these path levels are organisational rather than load-bearing: they keep a ``--cache-dir`` readable
    and prunable per model/dataset. The PaCMAP projection embeds the texts with ``model.embed`` and
    never touches the attribution backend, so every ``--attribution`` choice shares one projection cache
    at *this* level (below it, contributions split per backend).
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
    parser.add_argument(
        "--dataset-config",
        default=defaults.dataset_config,
        help="HF dataset config/subset (e.g. a language) — needed when the dataset has more than one, "
        "such as mteb/amazon_reviews_multi (pass 'fr', 'en', ...).",
    )
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
        "--seed",
        type=int,
        default=defaults.seed,
        help="Shuffle seed applied before taking --n / --n-reference rows, so sampling is representative "
        "rather than whatever the dataset's on-disk row order puts first.",
    )
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
        "--lig-batch-size",
        type=int,
        default=defaults.lig_batch_size,
        help=(
            "Captum internal_batch_size for --attribution lig: chunks each sample's n_steps=50 "
            "integration batch instead of running it through the model in one shot. Lower this "
            "(e.g. 2 or 1) if LIG hits a CUDA out-of-memory error."
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


def _load_hf_dataset(name: str, dataset_config: str | None, split: str) -> datasets.Dataset:
    """Load one split of a HF dataset, falling back to raw per-split JSONL for legacy-script repos.

    Some older dataset repos (e.g. ``mteb/amazon_reviews_multi``) still ship a ``<name>.py`` loading
    script, which ``datasets>=4.0`` refuses to execute (``RuntimeError: Dataset scripts are no longer
    supported``). Their raw ``<config>/<split>.jsonl`` files are usually still present in the repo, so on
    exactly that error this retries against them directly via the generic ``json`` builder, bypassing the
    broken script. Any other failure (missing dataset, bad split name, ...) is not masked.
    """
    try:
        return datasets.load_dataset(name, dataset_config, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc) or dataset_config is None:
            raise
        logger.warning("%s ships a legacy loading script; falling back to raw %s/%s.jsonl", name, dataset_config, split)
        return datasets.load_dataset(
            "json", data_files=f"hf://datasets/{name}/{dataset_config}/{split}.jsonl", split="train"
        )


def _load_amazon_reviews_multi_binary(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """``mteb/amazon_reviews_multi`` — bucket its 5-star ``label`` (0-4) onto binary sentiment.

    The raw ``label`` is a star rating minus one (0 = 1-star, ..., 4 = 5-star) — not the binary
    positive/negative space a binary sentiment model (e.g. ``almanach/camembertv2-base-cls``) predicts
    into, and not something ``--label-map`` can fix: that flag only renames labels 1:1, it can't collapse
    five classes into two or drop one. Its ``label_text`` column doesn't help either — it's just
    ``str(label)`` ("0".."4"), not a description.

    Standard star-rating bucketing: rating > 3 -> ``"positive"``, rating < 3 -> ``"negative"``; the
    3-star (rating == 3) review is neutral, has no binary ground truth, and is dropped — so this can
    return fewer than ``n`` rows. This is a *ground-truth* convention only — it says nothing about
    whether ``almanach/camembertv2-base-cls`` (or any other model served against this data) predicts
    accurately; see ``demo/check_camembertv2_amazon_fr.py`` for that question answered against a
    cleaner, in-domain gold set.

    Also fixes the "all one label" symptom on this dataset specifically: the raw file is sorted by
    ``label`` in blocks of 1000 identical values, so an unshuffled ``[:n]`` slice for any ``n`` <= 1000
    is entirely 1-star. Shuffling with ``config.seed`` before selecting fixes that regardless of ``n``.
    """
    dataset = _load_hf_dataset(config.dataset_name, config.dataset_config, split)
    dataset = dataset.shuffle(seed=config.seed).select(range(min(n, len(dataset))))
    texts, labels = [], []
    for row in dataset:
        rating = row["label"] + 1
        if rating == 3:
            continue
        texts.append(row["text"])
        labels.append("positive" if rating > 3 else "negative")
    return texts, labels


# ``--dataset-name`` -> a custom loader for a dataset whose ground-truth labels need dataset-specific
# logic (bucketing, dropping rows, ...) that the generic ``--label-column``/``--label-map`` path can't
# express. Anything not listed takes the generic path in ``_load_split``. Mirrors ``MODEL_BUILDERS``: put
# all dataset-specific code inside the loader and register it here — ``_load_split`` itself never changes.
DATASET_LOADERS: dict[str, Callable[[ServeConfig, str, int], tuple[list[str], list[str]]]] = {
    "mteb/amazon_reviews_multi": _load_amazon_reviews_multi_binary,
}


def _load_split(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """Load ``n`` ``(texts, label_strings)`` of ``split`` from the configured dataset.

    A dataset registered in :data:`DATASET_LOADERS` is delegated to its loader entirely (see there for
    why ``mteb/amazon_reviews_multi`` needs one). Otherwise, the generic path applies: rows are shuffled
    with ``config.seed`` before taking the first ``n`` — so samples are representative even when the
    dataset's on-disk order groups rows by label — and ground-truth label strings come from the
    dataset's own ``ClassLabel`` feature names when present (the usual shape for HF classification
    datasets), otherwise the raw values are stringified as-is. ``config.label_map`` then renames them to
    match the model's label names when the two disagree on spelling/casing (e.g. dataset ``"neg"`` vs.
    model ``"NEGATIVE"``) despite sharing the same class order.
    """
    loader = DATASET_LOADERS.get(config.dataset_name)
    if loader is not None:
        sentences, labels = loader(config, split, n)
        logger.info(
            "Loaded %d samples from %s (split=%s) via its registered dataset loader",
            len(sentences),
            config.dataset_name,
            split,
        )
        return sentences, labels

    dataset = _load_hf_dataset(config.dataset_name, config.dataset_config, split).shuffle(seed=config.seed)
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


# transformers' sentinel for "tokenizer_config.json never set model_max_length" — effectively means
# "no limit", not a real length. Passing it straight through makes truncation=True a no-op.
_UNSET_TOKENIZER_MAX_LENGTH = 100_000


def _safe_max_length(tokenizer, default: int = 512) -> int | None:
    """The tokenizer's own truncation length, unless it's the "unset" sentinel — then ``default``.

    A checkpoint whose tokenizer config never set ``model_max_length`` reports ``~1e30`` instead, so
    ``truncation=True`` with no explicit ``max_length`` silently does nothing and a long text tokenizes
    **unbounded**. That is not merely slow or memory-hungry: once a sequence is longer than the model's
    ``max_position_embeddings``, its position ids index past the position/token-type buffers, and the
    resulting out-of-bounds gather aborts the CUDA context outright (``device-side assert triggered``).
    The abort is asynchronous, so it surfaces at whatever unrelated line next synchronises — see
    ``_hf_classifier_model``, which is why every HF classifier here is built through one place.
    """
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None or model_max_length > _UNSET_TOKENIZER_MAX_LENGTH:
        return default
    return model_max_length


def _hf_classifier_model(config: ServeConfig, device: str, **kwargs) -> HFClassifierModel:
    """Build an ``HFClassifierModel`` with truncation always resolved — the single construction point.

    Every path that produces an ``HFClassifierModel`` (the generic loader *and* every custom builder in
    :data:`MODEL_BUILDERS`) goes through here, so ``max_length`` can never be left unset by an author who
    only meant to override something else, such as ``label_names``. That is not hypothetical: this helper
    exists because a custom builder once constructed the model directly and inherited
    ``max_length=None``, which silently disabled truncation for a tokenizer reporting the "unset"
    sentinel and made any input longer than the model's position buffer abort the CUDA context.

    ``kwargs`` are forwarded to :class:`~shapash.model.HFClassifierModel` (e.g. ``label_names=``).
    """
    tokenizer = _load_tokenizer(config.model_name)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_name).to(device)
    logger.info("Loaded %s as HFClassifierModel on device: %s", config.model_name, next(classifier.parameters()).device)
    max_length = _safe_max_length(tokenizer)
    if max_length != tokenizer.model_max_length:
        logger.info(
            "%s's tokenizer reports no configured model_max_length — truncating at %d tokens instead.",
            config.model_name,
            max_length,
        )
    return HFClassifierModel(classifier, tokenizer, max_length=max_length, **kwargs)


def _load_hf_classifier(config: ServeConfig, device: str) -> HFClassifierModel:
    """The generic default: any standard ``AutoModelForSequenceClassification`` checkpoint.

    Covers every certified encoder architecture (BERT/DistilBERT/RoBERTa/XLM-R/DeBERTa-v1, and MiniLM
    packaged as a sequence classifier). Label names come from the model's own ``config.id2label``, so a
    new ``--model-name`` brings its own classes with no extra flags.
    """
    return _hf_classifier_model(config, device)


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


def _build_camembertv2_cls_model(config: ServeConfig, device: str) -> HFClassifierModel:
    """``almanach/camembertv2-base-cls`` — standard checkpoint, but don't trust its predictions either.

    Architecturally this is a plain ``RobertaForSequenceClassification`` (CamemBERT v2) that would load
    fine through the generic path — the config problem is only ``label_names``. Its ``config.json`` ships
    ``label2id`` but no ``id2label``; when only one of the pair is present, transformers regenerates the
    missing side (and *overwrites* the other) via a ``num_labels`` setter, so ``config.id2label`` resolves
    to meaningless ``{0: "LABEL_0", 1: "LABEL_1"}`` at load time — the generic ``_load_hf_classifier``
    path (which reads ``label_names`` from exactly that) would show those in the webapp.

    Passing ``label_names`` explicitly only fixes the *display* problem, not the model. A first attempt
    at picking the right order used a 10-example hand-written probe (documented order 2/10, flipped
    8/10) and shipped the flip — that was wrong to trust: ``demo/check_camembertv2_amazon_fr.py`` scores
    both orderings against 388 gold-labeled reviews from this checkpoint's own FLUE-CLS fine-tuning
    domain and finds **neither beats the trivial majority-class baseline**, and the model is confidently
    wrong about as often as confidently right (mean top-class probability ~0.79 regardless of
    correctness). The 10-example probe was noise, not signal — a small hand check can't tell a genuinely
    swapped label from a model that just doesn't work. ``label_names`` below keeps the checkpoint's own
    documented convention (no evidence favours the flip over it); treat every prediction from this model
    as illustrative, not accurate — this builder exists as a worked example of the
    label-names-can't-be-trusted registration pattern, not as a working sentiment classifier.

    Only ``label_names`` is overridden — everything else, truncation included, comes from
    :func:`_hf_classifier_model`. This checkpoint's tokenizer reports no ``model_max_length`` and its
    ``max_position_embeddings`` is 1025, so building the model directly (as this once did) left
    truncation disabled and let any review over 1024 tokens abort the CUDA context.
    """
    logger.warning(
        "%s: this checkpoint's predictions are near chance-level and confidently wrong on gold in-domain "
        "data (neither label ordering beats the majority-class baseline) — see "
        "demo/check_camembertv2_amazon_fr.py. Treat its output in this demo as illustrative only.",
        config.model_name,
    )
    return _hf_classifier_model(config, device, label_names=["negative", "positive"])


# ``--model-name`` -> a custom builder for a checkpoint whose head can't be introspected from its name.
# Anything not listed takes the generic ``_load_hf_classifier`` path. To serve a new custom model, add a
# ``build(config, device) -> TextModel`` function above and one entry here — the core loader is untouched.
# See the module docstring's "Serving a custom model" section for the per-adapter recipe.
MODEL_BUILDERS: dict[str, Callable[[ServeConfig, str], TextModel]] = {
    "shhossain/all-MiniLM-L6-v2-sentiment-classifier": _build_shhossain_sentiment_model,
    "almanach/camembertv2-base-cls": _build_camembertv2_cls_model,
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
        # internal_batch_size chunks each sample's n_steps=50 scaled-copies batch so it doesn't OOM
        # memory-hungry architectures (e.g. DeBERTa) on a small GPU — see ServeConfig.lig_batch_size.
        return NlpCaptumLigBackend(
            model,
            label_names=model.label_names,
            explainer_compute_args={"internal_batch_size": config.lig_batch_size},
            show_progress=True,
        )
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
    if config.recompute:
        # Drop this text set's cached contributions so the steps below recompute + overwrite. The
        # explainer owns the key (it covers the model and backend), so it also owns the deletion —
        # only *this* model+backend's entry goes, other backends' caches are left intact. The
        # embeddings + projection are dropped by compute_projection(recompute=True) below.
        logger.info("--recompute: dropping cached %s contributions + projection for these texts", config.attribution)
        xpl.clear_cache(sentences, compile_dir)

    # Both steps load from cache if one exists for these exact texts, otherwise they compute and write
    # it. So the first run is slow; later runs only pay the model load.
    compile_cache = xpl.cache_path(sentences, compile_dir)
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
