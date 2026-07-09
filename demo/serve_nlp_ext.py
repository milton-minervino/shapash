"""Serve the NLP explainer **with the What-if Lab** (live model, cached compute).

Unlike ``serve_nlp.py``, the What-if Lab needs a live model in memory — editing a
sample, re-predicting, and generating counterfactuals all call the real model, so a
model-free snapshot (``from_snapshot``) makes the lab self-disable. There is therefore
no way to serve what-if without loading the model.

What *can* be avoided on every restart is recomputing the expensive stuff around the
model. This script caches both to ``--cache-dir`` (keyed by a hash of the input texts):

* **SHAP contributions + predictions** — via the built-in ``compile(cache_dir=...)``.
* **The 2-D PaCMAP projection** — via a small ``<hash>.proj.npy`` written here (the
  library deliberately does not own a projection method).

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

Attribution and counterfactual methods are both selectable (Captum-backed alternatives to the
defaults, see the ``[nlp]`` extra):

* ``--attribution {shap,lig}`` — sentence-highlight method: ``shap`` (KernelSHAP, default) or ``lig``
  (Captum ``LayerIntegratedGradients``). The two are cached in **separate** subdirectories, so you can
  flip between them freely without ``--recompute``.
* ``--counterfactual {hotflip,ablation}`` — What-if generator: ``hotflip`` (gradient-based token
  substitution, default) or ``ablation`` (Captum ``FeatureAblation`` token removal). Runs live in the
  lab, so it is not cached.

The on-disk cache mirrors these dependencies as a hierarchy under ``--cache-dir``:
``<model>/<dataset>__<split>/`` holds the (backend-independent) ``<hash>.proj.npy`` projection, and a
per-backend ``<...>/nlp_shap/`` or ``<...>/nlp_captum_lig/`` subdirectory holds that method's
``<hash>.pkl`` contributions.

Usage
-----
    python demo/serve_nlp_ext.py [--n 100] [--cache-dir demo/nlp_ext_cache] [--port 8051]
    python demo/serve_nlp_ext.py --recompute   # ignore the cache and recompute
    python demo/serve_nlp_ext.py --attribution lig --counterfactual ablation   # Captum methods
    python demo/serve_nlp_ext.py --model-name distilbert-base-uncased-finetuned-sst-2-english \\
        --dataset-name sst2 --dataset-split validation --text-column sentence
    python demo/serve_nlp_ext.py --model-name lvwerra/distilbert-imdb --dataset-name stanfordnlp/imdb \\
        --label-column label --label-map '{"neg": "NEGATIVE", "pos": "POSITIVE"}'

First run needs transformers, datasets and pacmap (the ``[nlp]`` extra).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import pacmap
import torch
import transformers

from shapash.backend import NlpCaptumLigBackend
from shapash.compute.generators import AblationFlipGenerator, HotFlipGenerator
from shapash.explainer.nlp_explainer import (
    NlpExplainer,
    _hash_texts,  # same keying as the compile cache
)
from shapash.model import HFClassifierModel

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
    counterfactual: str = "hotflip"  # what-if generator: "hotflip" | "ablation" (Captum FeatureAblation)
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
        "--counterfactual",
        choices=["hotflip", "ablation"],
        default=defaults.counterfactual,
        help=(
            "What-if counterfactual generator: 'hotflip' (gradient-based token substitution, the "
            "default) or 'ablation' (Captum FeatureAblation token removal). Runs live — not cached."
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


def load_data(config: ServeConfig) -> tuple[list[str], list[str]]:
    """Load ``config.n`` samples and their display label strings from the configured dataset.

    Ground-truth label strings come from the dataset's own ``ClassLabel`` feature names when
    present (the usual shape for HF classification datasets); otherwise the raw values are
    stringified as-is. ``config.label_map`` then renames them to match the model's label
    names when the two disagree on spelling/casing (e.g. dataset ``"neg"`` vs. model
    ``"NEGATIVE"``) despite sharing the same class order.
    """
    dataset = datasets.load_dataset(config.dataset_name, split=config.dataset_split)
    sentences = dataset[config.text_column][: config.n]
    raw_labels = dataset[config.label_column][: config.n]
    names = getattr(dataset.features.get(config.label_column), "names", None)
    y_true = [names[i] for i in raw_labels] if names is not None else [str(v) for v in raw_labels]
    if config.label_map:
        y_true = [config.label_map.get(v, v) for v in y_true]
    logger.info("Loaded %d samples from %s (split=%s)", len(sentences), config.dataset_name, config.dataset_split)
    return sentences, y_true


def load_model(config: ServeConfig) -> HFClassifierModel:
    """Load the tokenizer + classifier on the best available device.

    Label names aren't passed explicitly: ``HFClassifierModel`` reads them from the model's
    own ``config.id2label`` when none are given, so a new ``--model-name`` brings its own
    classes automatically.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA device found — using CPU")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_name).to(device)
    logger.info("Loaded model %s on device: %s", config.model_name, next(classifier.parameters()).device)
    return HFClassifierModel(classifier, tokenizer)


def build_backend(config: ServeConfig, model: HFClassifierModel) -> NlpCaptumLigBackend | None:
    """Return the attribution backend selected by ``--attribution``.

    Returns ``None`` for ``"shap"`` so ``NlpExplainer`` builds its default ``NlpShapBackend`` (which
    needs the model's ``shap_callable`` wiring); ``"lig"`` returns an explicit ``NlpCaptumLigBackend``.
    """
    if config.attribution == "lig":
        # LIG runs one integration per class per sample — show a progress bar over the batch.
        return NlpCaptumLigBackend(model, label_names=model.label_names, show_progress=True)
    return None


def build_cf_generator(config: ServeConfig, model: HFClassifierModel) -> HotFlipGenerator | AblationFlipGenerator:
    """Return the counterfactual generator selected by ``--counterfactual`` (both fit an HF classifier)."""
    if config.counterfactual == "ablation":
        return AblationFlipGenerator(model)
    return HotFlipGenerator(model)


def load_or_project(sentences, model, cache_dir: Path, n_components: int = 2) -> np.ndarray:
    """Return a 2-D PaCMAP projection, reading from / writing to ``cache_dir``.

    Cached alongside the ``compile`` results under the same text hash so a single
    ``cache_dir`` holds everything the app needs after the first run.
    """
    cache_file = cache_dir / f"{_hash_texts(list(sentences))}.proj.npy"
    if cache_file.exists():
        logger.info("Projection cache hit — loading %s", cache_file)
        return np.load(cache_file)

    logger.info("Projection cache miss — computing PaCMAP projection (embedding %d texts)", len(sentences))
    projector = pacmap.PaCMAP(n_components=n_components, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)
    projected = projector.fit_transform(model.embed(sentences), init="pca")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, projected)
    logger.info("Projection computed and cached to %s", cache_file)
    return projected


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
        (dataset_dir / f"{text_hash}.proj.npy").unlink(missing_ok=True)

    model = load_model(config)

    xpl = NlpExplainer(
        model,
        label_names=model.label_names,
        backend=build_backend(config, model),
        cf_generator=build_cf_generator(config, model),
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
    projected = load_or_project(sentences, model, dataset_dir)

    logger.info(
        "attribution=%s | counterfactual=%s | can_edit=%s | can_counterfactual=%s",
        config.attribution,
        config.counterfactual,
        xpl.can_edit(),
        xpl.can_counterfactual(),
    )
    logger.info("Serving on http://%s:%d (Ctrl+C to stop)", config.host, config.port)
    xpl.run_app(port=config.port, debug=False, host=config.host, scatter_xy=projected)


if __name__ == "__main__":
    main()
