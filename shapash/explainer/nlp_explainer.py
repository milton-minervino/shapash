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
from shapash.compute.generators.ablation_flip import AblationFlipGenerator
from shapash.compute.generators.base import Counterfactual, CounterfactualGenerator, Field
from shapash.compute.generators.hotflip import HotFlipGenerator
from shapash.model.base import TextModel
from shapash.model.hf import HFPipelineModel
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.webapp.nlp_app import NlpWebApp

# Built-in counterfactual generators, in preference order: HotFlip (gradient-based, richer
# substitutions) first, AblationFlip (forward-pass-only removal) as the broader fallback. Every entry
# compatible with the bound model is offered in the webapp's method selector.
_BUILTIN_CF_GENERATORS: tuple[type[CounterfactualGenerator], ...] = (HotFlipGenerator, AblationFlipGenerator)


def _looks_like_pipeline(model: object) -> bool:
    """Heuristic: a HuggingFace ``text-classification`` pipeline is callable and has a tokenizer."""
    return callable(model) and hasattr(model, "tokenizer")


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
        cf_generator: CounterfactualGenerator | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        self.model = model
        self.label_names = label_names

        # Wrap the model in a capability-aware TextModel adapter when possible. A raw HuggingFace
        # pipeline becomes an HFPipelineModel (predict-only); a TextModel is used as-is; anything
        # else (e.g. a LIME classifier_fn) leaves _text_model as None and the legacy path is kept.
        if isinstance(model, TextModel):
            self._text_model: TextModel | None = model
            if label_names is None:
                self.label_names = model.label_names
        elif _looks_like_pipeline(model):
            self._text_model = HFPipelineModel(model, label_names)
        else:
            self._text_model = None

        # The SHAP/LIME backend consumes a plain callable; a TextModel exposes it via shap_callable.
        shap_model = self._text_model.shap_callable if self._text_model is not None else model
        self.backend: NlpBackend = backend or NlpShapBackend(
            model=shap_model,
            label_names=self.label_names,
            explainer_args=explainer_args or {},
            explainer_compute_args=explainer_compute_args or {},
        )

        # Counterfactual generators: explicit > auto-discovered > none. An explicit ``cf_generator``
        # is used verbatim (no surprise additions). Otherwise every built-in compatible with the model
        # is offered — so the webapp can switch methods live — with HotFlip preferred and AblationFlip
        # (forward-pass-only) covering prediction-only models such as a plain pipeline. ``cf_generator``
        # stays the *active/default* generator for callers that don't select one.
        if cf_generator is not None:
            generators: list[CounterfactualGenerator] = [cf_generator]
        elif self._text_model is not None:
            generators = [
                cls(self._text_model) for cls in _BUILTIN_CF_GENERATORS if cls.is_compatible(self._text_model)
            ]
        else:
            generators = []
        self.cf_generators: dict[str, CounterfactualGenerator] = {g.name: g for g in generators}
        self.cf_generator: CounterfactualGenerator | None = generators[0] if generators else None

        self.contributions: NlpContributions | None = None
        self.texts: pd.Series | None = None
        self.y_pred: pd.Series | None = None
        self.y_prob: pd.DataFrame | None = None
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
                self.y_prob = cached.get("y_prob")
            else:
                explain_data = self.backend.run_explainer(text_list)
                self.contributions = self.backend.get_local_contributions(text_list, explain_data)
                pred_df = self._predict(text_list)
                self.y_pred = pred_df["prediction"]
                self.y_prob = pred_df.drop(columns=["prediction"])
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("wb") as f:
                        pickle.dump(
                            {"contributions": self.contributions, "y_pred": self.y_pred, "y_prob": self.y_prob}, f
                        )

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

    def save_snapshot(self, path: str | Path, scatter_xy=None) -> None:
        """Persist compiled results (no model or backend) for offline serving.

        The saved file contains only the data required by ``NlpWebApp`` —
        texts, contributions, predictions and the optional 2-D projection.
        It can be loaded back with :meth:`from_snapshot` without installing
        torch, transformers, or any other heavy inference dependency.

        Parameters
        ----------
        path : str or Path
            Destination ``.pkl`` file.
        scatter_xy : np.ndarray, optional
            Pre-computed 2-D projection to bundle with the snapshot (same
            array you would pass to ``run_app``).
        """
        if self.contributions is None:
            raise RuntimeError("Call compile() before save_snapshot().")
        state = {
            "texts": self.texts,
            "contributions": self.contributions,
            "y_pred": self.y_pred,
            "y_prob": self.y_prob,
            "y_true": self.y_true,
            "label_names": self.label_names,
            "scatter_xy": scatter_xy,
        }
        with Path(path).open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def from_snapshot(cls, path: str | Path) -> tuple[NlpExplainer, object]:
        """Restore a snapshot saved by :meth:`save_snapshot`.

        Returns
        -------
        explainer : NlpExplainer
            Ready-to-serve instance (``model`` and ``backend`` are ``None``).
        scatter_xy : np.ndarray or None
            The projection array bundled at save time, or ``None``.
        """
        with Path(path).open("rb") as f:
            state = pickle.load(f)  # noqa: S301
        xpl = cls.__new__(cls)
        xpl.model = None
        xpl.backend = None
        xpl._text_model = None
        xpl.cf_generator = None
        xpl.cf_generators = {}
        xpl.label_names = state.get("label_names")
        xpl.texts = state["texts"]
        xpl.contributions = state["contributions"]
        xpl.y_pred = state["y_pred"]
        xpl.y_prob = state.get("y_prob")
        xpl.y_true = state.get("y_true")
        xpl._data_hash = None
        return xpl, state.get("scatter_xy")

    def run_app(self, port: int = 8050, debug: bool = False, host: str = "127.0.0.1", scatter_xy=None) -> None:
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
        NlpWebApp(self, scatter_xy=scatter_xy).run(port=port, debug=debug, host=host)

    # ------------------------------------------------------------------
    # InteractiveEngine — live what-if surface (see explainer/interactive.py)
    # ------------------------------------------------------------------

    def can_edit(self) -> bool:
        """Whether edited text can be re-predicted and re-explained live.

        Requires a live model adapter and backend (both are ``None`` after ``from_snapshot``).
        """
        return getattr(self, "_text_model", None) is not None and getattr(self, "backend", None) is not None

    def can_counterfactual(self) -> bool:
        """Whether a counterfactual generator is bound and ready."""
        return getattr(self, "cf_generator", None) is not None

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict a single text, returning ``(label, {label: probability})``."""
        text_model = self._require_text_model()
        probs = text_model.predict([text])[0]
        names = text_model.label_names or self.label_names or [str(i) for i in range(len(probs))]
        idx = int(probs.argmax())
        return names[idx], {name: float(p) for name, p in zip(names, probs, strict=False)}

    def explain_text(self, text: str) -> tuple[NlpContributions, str, dict[str, float]]:
        """Re-explain one (possibly edited) text: contributions + prediction + probabilities.

        Bypasses the batch hash cache — this is a single, on-demand explanation of new input.
        """
        if getattr(self, "backend", None) is None:
            raise RuntimeError("explain_text() requires a live backend (unavailable on a snapshot).")
        raw = self.backend.run_explainer([text])
        contributions = self.backend.get_local_contributions([text], raw)
        contributions.label_names = self.label_names
        label, probabilities = self.predict(text)
        return contributions, label, probabilities

    def available_cf_generators(self) -> list[tuple[str, str]]:
        """Return ``(name, display_name)`` for each bound generator, in preference order.

        Drives the webapp's counterfactual-method selector; empty when no generator is configured
        (e.g. a snapshot-restored explainer).
        """
        return [(g.name, g.display_name) for g in getattr(self, "cf_generators", {}).values()]

    def generate_counterfactuals(
        self, text: str, config: dict | None = None, generator: str | None = None
    ) -> list[Counterfactual]:
        """Generate counterfactuals for ``text`` via the selected (or active) generator.

        Parameters
        ----------
        text : str
            Text to perturb.
        config : dict, optional
            Overrides for the generator's config spec.
        generator : str, optional
            ``name`` of one of :meth:`available_cf_generators`. Defaults to the active generator.
        """
        gen = self._select_cf_generator(generator)
        if gen is None:
            raise RuntimeError("No counterfactual generator is configured for this explainer.")
        return gen.generate(text, config=config)

    def cf_config_spec(self, generator: str | None = None) -> dict[str, Field]:
        """Return the tunable config spec of the selected (or active) generator, empty when none."""
        gen = self._select_cf_generator(generator)
        return gen.config_spec() if gen is not None else {}

    def _select_cf_generator(self, generator: str | None) -> CounterfactualGenerator | None:
        """Resolve a generator by ``name`` (or the active one when ``generator`` is ``None``)."""
        generators = getattr(self, "cf_generators", {})
        if generator is not None:
            if generator not in generators:
                raise KeyError(f"Unknown counterfactual generator {generator!r}; available: {list(generators)}.")
            return generators[generator]
        return getattr(self, "cf_generator", None)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _require_text_model(self) -> TextModel:
        text_model = getattr(self, "_text_model", None)
        if text_model is None:
            raise RuntimeError("Live prediction requires a TextModel-backed explainer (unavailable on a snapshot).")
        return text_model

    def _predict(self, text_list: list[str]) -> pd.DataFrame:
        """Run the pipeline and return a unified DataFrame of predictions and probabilities.

        The first column is always ``"prediction"`` (the argmax label).
        Subsequent columns hold class probabilities: one per class when the
        pipeline returns all scores (``return_all_scores=True``), or a single
        ``"probability"`` column (the winning class confidence) otherwise.

        Handles both ``return_all_scores=True`` (list of lists of dicts) and
        single-prediction (list of dicts) pipeline output formats.
        """
        text_model = getattr(self, "_text_model", None)
        if text_model is not None:
            probs = text_model.predict(text_list)
            names = text_model.label_names or self.label_names or [str(i) for i in range(probs.shape[1])]
            result = pd.DataFrame(probs, index=self.texts.index, columns=list(names))
            labels = [names[int(i)] for i in probs.argmax(axis=1)]
            result.insert(0, "prediction", pd.Series(labels, index=self.texts.index))
            return result

        raw = self.model(text_list)
        if raw and isinstance(raw[0], list):
            labels = [max(preds, key=lambda p: p["score"])["label"] for preds in raw]
            col_labels = [d["label"] for d in raw[0]]
            if self.label_names and len(self.label_names) == len(col_labels):
                col_labels = self.label_names
            probs = [[d["score"] for d in preds] for preds in raw]
            result = pd.DataFrame(probs, index=self.texts.index, columns=col_labels)
        else:
            labels = [p["label"] for p in raw]
            result = pd.DataFrame(
                {"probability": [p["score"] for p in raw]},
                index=self.texts.index,
            )
        result.insert(0, "prediction", pd.Series(labels, index=self.texts.index))
        return result
