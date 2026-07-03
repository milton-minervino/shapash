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

Usage
-----
    python demo/serve_nlp_ext.py [--n 100] [--cache-dir demo/nlp_ext_cache] [--port 8051]
    python demo/serve_nlp_ext.py --recompute   # ignore the cache and recompute

First run needs transformers, datasets and pacmap (the ``[nlp]`` extra).
"""

from __future__ import annotations

import argparse
import logging
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
_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
_LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

logger = logging.getLogger("serve_nlp_ext")


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
    """Load the model, (re)use the SHAP + projection cache, and serve the What-if webapp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of samples to load.")
    parser.add_argument("--cache-dir", type=Path, default=_HERE / "nlp_ext_cache")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore any cached results and recompute SHAP + projection (overwrites the cache).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # Quiet noisy third-party INFO logs (HF Hub freshness checks, download chatter) so the
    # device/cache messages below stand out.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    dataset = datasets.load_dataset("dair-ai/emotion", split="train")
    sentences = dataset["text"][: args.n]
    labels = dataset["label"][: args.n]
    y_true = [_LABEL_NAMES[i] for i in labels]
    logger.info("Loaded %d samples from dair-ai/emotion", len(sentences))

    text_hash = _hash_texts(list(sentences))
    if args.recompute:
        # Drop this text set's cached artifacts so the steps below recompute + overwrite.
        logger.info("--recompute: dropping cached SHAP + projection for these texts")
        (args.cache_dir / f"{text_hash}.pkl").unlink(missing_ok=True)
        (args.cache_dir / f"{text_hash}.proj.npy").unlink(missing_ok=True)

    # Select the device and report it — GPU when available, CPU otherwise.
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA device found — using CPU")

    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME).to(device)
    # Confirm the weights actually landed on the intended device.
    logger.info("Loaded model %s on device: %s", _MODEL_NAME, next(classifier.parameters()).device)

    # Full-capability adapter: predict + embeddings + gradients (what HotFlip needs).
    model = HFClassifierModel(classifier, tokenizer, label_names=_LABEL_NAMES)

    xpl = NlpExplainer(model, label_names=_LABEL_NAMES, cf_generator=HotFlipGenerator(model))
    # Both steps load from cache_dir if a cache for these exact texts exists, otherwise they
    # compute and write it. So the first run is slow; later runs only pay the model load.
    compile_cache = args.cache_dir / f"{text_hash}.pkl"
    if compile_cache.exists():
        logger.info("SHAP cache hit — loading %s", compile_cache)
    else:
        logger.info("SHAP cache miss — computing contributions for %d texts (this is the slow part)", len(sentences))
    xpl.compile(sentences, y_true=y_true, cache_dir=args.cache_dir)
    projected = load_or_project(sentences, model, args.cache_dir)

    logger.info("can_edit=%s | can_counterfactual=%s", xpl.can_edit(), xpl.can_counterfactual())
    logger.info("Serving on http://%s:%d (Ctrl+C to stop)", args.host, args.port)
    xpl.run_app(port=args.port, debug=False, host=args.host, scatter_xy=projected)


if __name__ == "__main__":
    main()
