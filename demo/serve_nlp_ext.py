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

Usage
-----
    python demo/serve_nlp_ext.py [--n 100] [--cache-dir demo/nlp_ext_cache] [--port 8051]
    python demo/serve_nlp_ext.py --recompute   # ignore the cache and recompute
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

from shapash.compute.generators import HotFlipGenerator
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
    cache_dir: Path = _HERE / "nlp_ext_cache"
    port: int = 8051
    host: str = "0.0.0.0"  # noqa: S104
    recompute: bool = False


def _model_cache_dir(config: ServeConfig) -> Path:
    """Return ``config.cache_dir`` namespaced by model.

    The compile/projection caches are keyed by a hash of the input texts only (see
    ``_hash_texts``), so two different models run over the same dataset/``--n`` would
    otherwise collide on the same cache file. A per-model subdirectory keeps them apart.
    """
    return config.cache_dir / config.model_name.replace("/", "__")


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
    cache_dir = _model_cache_dir(config)

    text_hash = _hash_texts(list(sentences))
    if config.recompute:
        # Drop this text set's cached artifacts so the steps below recompute + overwrite.
        logger.info("--recompute: dropping cached SHAP + projection for these texts")
        (cache_dir / f"{text_hash}.pkl").unlink(missing_ok=True)
        (cache_dir / f"{text_hash}.proj.npy").unlink(missing_ok=True)

    model = load_model(config)

    xpl = NlpExplainer(model, label_names=model.label_names, cf_generator=HotFlipGenerator(model))
    # Both steps load from cache_dir if a cache for these exact texts exists, otherwise they
    # compute and write it. So the first run is slow; later runs only pay the model load.
    compile_cache = cache_dir / f"{text_hash}.pkl"
    if compile_cache.exists():
        logger.info("SHAP cache hit — loading %s", compile_cache)
    else:
        logger.info("SHAP cache miss — computing contributions for %d texts (this is the slow part)", len(sentences))
    xpl.compile(sentences, y_true=y_true, cache_dir=cache_dir)
    projected = load_or_project(sentences, model, cache_dir)

    logger.info("can_edit=%s | can_counterfactual=%s", xpl.can_edit(), xpl.can_counterfactual())
    logger.info("Serving on http://%s:%d (Ctrl+C to stop)", config.host, config.port)
    xpl.run_app(port=config.port, debug=False, host=config.host, scatter_xy=projected)


if __name__ == "__main__":
    main()
