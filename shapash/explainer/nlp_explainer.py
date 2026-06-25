"""Prototype NLP explainer for text classification models.

This module is a **bridge prototype** toward Phase 6 of the refactoring plan,
where token-level NLP explanations will be integrated into ``SmartExplainer``
via ``TextDataset`` and ``ExplanationSession``. Once those are in place, users
will call ``SmartExplainer.compile(x=TextDataset(texts))`` and this class will
be removed.

For now, ``NlpExplainer`` provides a minimal compile → plot → webapp workflow
for text without touching the tabular path in ``SmartExplainer.compile()``.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd
from plotly import graph_objs as go

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_shap_backend import NlpShapBackend
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.webapp.nlp_app import NlpWebApp


def _hash_texts(text_list: list[str]) -> str:
    h = hashlib.md5(usedforsecurity=False)
    for t in text_list:
        h.update(t.encode())
    return h.hexdigest()


def _cache_file(data_hash: str, cache_dir: Path) -> Path:
    return cache_dir / f"{data_hash}.pkl"


class NlpExplainer:
    """Minimal explainer for text classification — token-level contributions.

    Uses ``NlpShapBackend`` by default.  Pass a pre-built backend instance to
    switch to a different explainer (e.g. ``NlpLimeBackend``).

    Parameters
    ----------
    model : callable
        Text pipeline or callable accepted by the chosen backend.
    label_names : list[str], optional
        Class names in the same order as the model output columns.
        Used in plot titles and the webapp class selector.
    backend : NlpBackend, optional
        Pre-built backend instance.  When provided, ``explainer_args`` and
        ``explainer_compute_args`` are ignored — configure the backend directly
        before passing it in.  Defaults to ``NlpShapBackend``.
    explainer_args : dict, optional
        Forwarded to ``NlpShapBackend.__init__`` when no ``backend`` is given.
    explainer_compute_args : dict, optional
        Forwarded to ``NlpShapBackend.__init__`` when no ``backend`` is given.

    Examples
    --------
    >>> import transformers
    >>> pipe = transformers.pipeline("text-classification", model="...", return_all_scores=True)
    >>> xpl = NlpExplainer(pipe, label_names=["sadness", "joy", "love", "anger", "fear", "surprise"])
    >>> xpl.compile(texts[:100])
    >>> xpl.run_app(port=8050)

    >>> from shapash.backend.nlp_lime_backend import NlpLimeBackend
    >>> lime_backend = NlpLimeBackend(classifier_fn, label_names=[...], explainer_compute_args={"num_features": 15})
    >>> xpl = NlpExplainer(classifier_fn, label_names=[...], backend=lime_backend)
    """

    def __init__(
        self,
        model,
        label_names: list[str] | None = None,
        backend: NlpBackend | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        self.model = model
        self.label_names = label_names
        self.backend: NlpBackend = backend or NlpShapBackend(
            model=model,
            label_names=label_names,
            explainer_args=explainer_args or {},
            explainer_compute_args=explainer_compute_args or {},
        )
        self.contributions: NlpContributions | None = None
        self.texts: pd.Series | None = None
        self.y_pred: pd.Series | None = None
        self.y_true: pd.Series | None = None
        self._data_hash: str | None = None

    def compile(
        self,
        x: list[str] | pd.Series,
        y_true: list | pd.Series | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Compute token-level contributions and model predictions.

        Results are cached in memory keyed by a hash of the input texts, so
        re-calling ``compile`` with the same data on the same instance skips the
        expensive explainer run.  Passing only ``y_true`` (same ``x``) is always
        a lightweight metadata update.  For persistence across kernel restarts,
        pass a ``cache_dir`` path to enable an opt-in disk cache.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.
        y_true : list or pd.Series, optional
            Ground-truth labels for each sample. When provided, the webapp will
            show a "Ground Truth" column alongside "Prediction" in the dataset
            table, making it easy to spot misclassifications.
        cache_dir : str or Path, optional
            If provided, contributions and predictions are also persisted to
            ``<cache_dir>/<hash>.pkl`` and reloaded on subsequent calls — even
            after a kernel restart.  Disabled by default.
        """
        texts = x if isinstance(x, pd.Series) else pd.Series(x)
        text_list = texts.tolist()
        new_hash = _hash_texts(text_list)

        if new_hash != self._data_hash:
            self.texts = texts
            cache_path = _cache_file(new_hash, Path(cache_dir)) if cache_dir is not None else None

            if cache_path is not None and cache_path.exists():
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)  # noqa: S301
                self.contributions = cached["contributions"]
                self.y_pred = cached["y_pred"]
            else:
                explain_data = self.backend.run_explainer(text_list)
                self.contributions = self.backend.get_local_contributions(text_list, explain_data)
                self.y_pred = self._predict_labels(text_list)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("wb") as f:
                        pickle.dump({"contributions": self.contributions, "y_pred": self.y_pred}, f)

            self.contributions.label_names = self.label_names
            self.contributions.index = self.texts.index
            self._data_hash = new_hash

        if y_true is not None:
            self.y_true = (
                y_true
                if isinstance(y_true, pd.Series)
                else pd.Series(y_true, index=self.texts.index, name="ground_truth")
            )

    def text_plot(self, pos: int = 0, label_idx: int = 0, max_tokens: int | None = None) -> go.Figure:
        """Plot token-level contributions for one sample and one class.

        Parameters
        ----------
        pos : int
            Positional index of the sample within the compiled batch.
        label_idx : int
            Index of the class to display (matches ``label_names`` order).
        max_tokens : int, optional
            If set, show only the top-``max_tokens`` tokens by absolute
            contribution magnitude (in their original sentence order).

        Returns
        -------
        go.Figure
        """
        if self.contributions is None:
            raise RuntimeError("Call compile() before text_plot().")

        tokens = self.contributions.token_strings[pos]
        values = self.contributions.values[pos]

        if values.ndim == 2:
            values = values[:, label_idx]

        label_name = (
            self.label_names[label_idx] if self.label_names is not None and label_idx < len(self.label_names) else None
        )
        title = f"Token contributions — {label_name}" if label_name else "Token contributions"

        return plot_token_highlight(tokens=tokens, values=values, title=title, max_tokens=max_tokens)

    def run_app(self, port: int = 8050, debug: bool = False, scatter_xy=None) -> None:
        """Launch the NLP explanation webapp.

        Parameters
        ----------
        port : int
            Port for the Dash development server.
        debug : bool
            Enable Dash debug mode (hot reload, error overlay).
        scatter_xy : np.ndarray, optional
            Pre-computed 2-D projection of the text samples, shape
            ``(n_samples, 2)``.  When provided, a scatter panel appears in the
            webapp that can be used to lasso/box-select a subset of samples,
            filtering both the dataset table and the global word importance
            plot to that subset.  Compute with any projection method
            (PaCMAP, UMAP, PCA, t-SNE …) and pass the result here.
        """
        if self.contributions is None:
            raise RuntimeError("Call compile() before run_app().")
        NlpWebApp(self, scatter_xy=scatter_xy).run(port=port, debug=debug)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _predict_labels(self, text_list: list[str]) -> pd.Series:
        """Run the pipeline once to obtain predicted class labels.

        Handles both ``return_all_scores=True`` (list of lists of dicts) and
        single-prediction (list of dicts) pipeline output formats.
        """
        raw = self.model(text_list)
        if raw and isinstance(raw[0], list):
            labels = [max(preds, key=lambda p: p["score"])["label"] for preds in raw]
        else:
            labels = [p["label"] for p in raw]
        return pd.Series(labels, index=self.texts.index, name="prediction")
