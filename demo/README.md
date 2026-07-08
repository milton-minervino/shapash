# Shapash NLP Explainer — Demo

Interactive webapp that explains a DistilBERT emotion classifier at the token level using SHAP values.

The base app ([below](#nlp-explainer)) is served read-only from a snapshot. An optional
[**Extended**](#nlp-explainer-extended) variant adds a live-model What-if Lab on top.

| Variant | Port | Live model? | Image | What it adds |
|---|---|---|---|---|
| **NLP Explainer** | 8050 | No — served from a snapshot | Lightweight (no torch/GPU) | Explore precomputed SHAP explanations |
| **NLP Explainer Extended** | 8051 | Yes — model in memory (CPU) | Heavier; model + caches baked in | The above **plus** live editing, re-prediction, and counterfactual suggestions |

---

# NLP Explainer

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- The snapshot file `demo/explainer_snapshot.pkl` (see Step 1 below)

---

## Step 1 — Generate the snapshot (one-time, requires GPU)

This step runs the model and computes SHAP values for 100 samples.  It only needs to be done once; the result is saved as a self-contained file.

```bash
# From the repo root
pip install "shapash[nlp]"
python demo/test_webapp_nlp.py
```

This writes `demo/explainer_snapshot.pkl`.  The file bundles texts, predictions, SHAP contributions, and the 2-D scatter projection — everything the webapp needs.

> **Already have the snapshot?** Skip to Step 2.

---

## Step 2 — Build and run the app

Run from the **repo root** (the build context needs access to the `shapash/` package):

```bash
docker compose -f demo/docker-compose.yml up --build shapash-nlp
```

The image installs only the lightweight serving dependencies (Dash, Plotly, pandas …).  No model, no GPU, no heavy ML libraries.  Drop `--build` on subsequent runs if nothing has changed.

Then open **http://localhost:8050** in your browser.

Press `Ctrl+C` to stop.

---

## Webapp overview

| Panel | Description |
|---|---|
| **Global word importance** | Mean SHAP contribution per word for the selected class. Use the top-K slider, sign filter, and word exclusion list to explore. |
| **Sample scatter** | 2D PaCMAP projection of text embeddings. Draw a box, lasso or click to filter the table and word importance to a subset. |
| **Dataset table** | All samples with predicted label, ground truth, and predicted probability. Click a row to inspect it. Click a bar in the word importance chart to filter to samples containing that word. |
| **Local contributions** | Token-level SHAP highlight for the selected sample. Toggle to a waterfall chart for a ranked view. |

---

# NLP Explainer Extended

Adds a **Data Editor** tab: edit a sample's text for a live prediction + token-contribution highlight, and a **Counterfactuals** tab to
get auto-generated minimal token flips (counterfactuals) that change the prediction.

This needs the live model in memory, so it can't use the snapshot — but SHAP, the projection, and the
model/dataset downloads are all cached, so only the first run pays for it. Runs on **CPU**.

**New webapp layout** with three panels:
- Left: control panel with **Dataset** (Dash AG Grid to explore the dataset and click on texts), **Embeddings** (Scatter plot, used to lasso/box select, hihglight word importance), and **Data Editor**
- Upper Right: **Word Importance** (Global explainability) tab and **Counterfactuals** tab.
- Lower Right: **Local contributions** divided as sentence highlights tab and waterfall plot tab.

Same idea as the snapshot: compute the heavy stuff (SHAP) **once, up front**, then Docker just loads
it. The model + dataset are downloaded by the Docker build itself, so you don't stage them.

**`demo/serve_nlp_ext.py` is the serving script**, and everything below runs it with its defaults
(the emotion model/dataset). Model, dataset, split, columns, sample count, and a ground-truth
label-renaming map are all configurable via `ServeConfig` in that file — see its module docstring
for `--model-name`/`--dataset-name`/`--dataset-split`/`--text-column`/`--label-column`/`--label-map`/
`--n` and worked examples (e.g. swapping in an IMDB sentiment model/dataset). `Dockerfile.nlp_ext`
itself stays hardcoded to the emotion example on purpose (it's meant to run out of the box for
anyone cloning the repo) — its build-time warm-up (`RUN python -c "..."`) downloads exactly that
model/dataset/split, so it always matches `ServeConfig`'s defaults. Serving a different model/dataset
combo in Docker means editing that warm-up line to match, then redoing Step 1 below with the same
flags before rebuilding.

### Step 1 — Pre-compute the SHAP cache (one-time, GPU optional)

Needs the `[nlp]` extra (torch, transformers, datasets, pacmap). Run the script once — it computes SHAP
+ the projection (a few minutes on CPU, faster on GPU), writes the cache, then starts serving. Once it
prints the local URL, the cache is on disk — press `Ctrl+C`:

```bash
pip install "shapash[nlp]"
python demo/serve_nlp_ext.py   # Ctrl+C once it starts serving
```

This writes `demo/nlp_ext_cache/` — SHAP contributions, predictions, and the 2-D projection, keyed by a
hash of the input texts. The Docker build copies it in, so this expensive step never runs in Docker.

> **Already have `demo/nlp_ext_cache/`?** Skip to Step 2.

### Step 2 — Build and run

Run from the **repo root**:

```bash
docker compose -f demo/docker-compose.yml up --build shapash-nlp-ext
```

The build downloads the model + dataset into the image and copies in the SHAP cache; at **run** time it
serves fully offline — no downloads, no SHAP, no GPU. (The build fails fast if `demo/nlp_ext_cache/` is
missing — do Step 1 first.)

Then open **http://localhost:8051** in your browser. Press `Ctrl+C` to stop.

## Run without Docker

Just run the same script — it serves on http://127.0.0.1:8051 and reuses the `demo/nlp_ext_cache/`
from Step 1 (recomputing only if the input texts change, or if you pass `--recompute`):

```bash
python demo/serve_nlp_ext.py             # ServeConfig defaults (500 samples, dair-ai/emotion test split)
python demo/serve_nlp_ext.py --n 1000    # more samples (recomputes + re-caches)
```

It uses the GPU if present, otherwise CPU.

> **Iterating on the webapp code?** Don't restart the process — run `serve_nlp_ext.py`'s body in a REPL
> once, then re-call `xpl.run_app(...)` after reloading the changed webapp modules. The model stays
> resident, so you never pay the reload.
