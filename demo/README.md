# Shapash NLP Explainer — Demo

Interactive webapp that explains a DistilBERT emotion classifier at the token level using SHAP values.

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

## Step 2 — Build the Docker image

Run from the **repo root** (the build context needs access to the `shapash/` package):

```bash
docker compose -f demo/docker-compose.yml build
```

The image installs only the lightweight serving dependencies (Dash, Plotly, pandas …).  No model, no GPU, no heavy ML libraries.

---

## Step 3 — Run the app

```bash
docker compose -f demo/docker-compose.yml up
```

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
