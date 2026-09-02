"""``NlpExplanation`` — the immutable artifact returned by ``NlpExplainer.explain()``.

The one result object for a batch: token-level contributions together with predictions,
ground truth and provenance, plus the word-importance aggregation behavior that acts on
them. There is deliberately no separate "contributions" sub-object nested inside it —
see the class docstring's Design note.

Persistence is a zip container (stdlib ``zipfile``) holding a plain-text ``meta.json``
(readable without any Python/shapash install) plus long/tidy parquet tables for the
numeric payload — no pickle anywhere. ``pyarrow`` is already a core shapash dependency,
so this costs nothing new. See :meth:`NlpExplanation.save`/:meth:`NlpExplanation.load`.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import pandas as pd

from shapash.__version__ import __version__ as _shapash_version
from shapash.backend.nlp_backend import is_punctuation

if TYPE_CHECKING:
    from shapash.explainer.nlp_plotter import NlpPlotter

# Stamped into meta.json on save and checked on load, so an incompatible file fails loudly instead of
# parsing into wrong numbers — a layout change (wide instead of tidy contributions, a different
# class_idx encoding) often still reads cleanly and yields a plausible, wrong explanation. The risk is
# live rather than hypothetical: explain() writes these files into cache_dir and its key covers
# texts/model/backend but *not* the shapash version, so an upgrade leaves old files the new reader
# will hit. It also cannot be added retroactively — a file written without a version is forever
# ambiguous — which is why it sits here at v1 with nothing yet to reject. Limit: it versions the
# *layout* only; numbers whose meaning changes under an identical schema are invisible to it (the
# shapash_version recorded alongside it is the breadcrumb for that, and nothing checks it). The tuple
# is what lets a future reader accept v1 as well as its own — the migration code does not exist yet.
_FORMAT_VERSION = 1
_SUPPORTED_FORMAT_VERSIONS = (1,)

_PROB_PREFIX = "y_prob__"


def select_label_column(values: np.ndarray, label_idx: int) -> np.ndarray:
    """Pick one class's column out of a per-token contribution array.

    Multiclass backends return ``values`` shaped ``(n_tokens, n_classes)``; binary and
    regression backends already return the 1-D case. Every render path — per-instance
    plots, corpus-level word importance, the webapp's live datapoint — needs this same
    rule applied consistently, so it lives here once rather than as a repeated inline
    ``if values.ndim == 2`` check.
    """
    return values[:, label_idx] if values.ndim == 2 else values


@dataclass(frozen=True, eq=False, slots=True)
class NlpExplanation:
    """Immutable result of :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.explain`.

    A pure function of *(texts, model, backend)* — model-free and backend-free by
    construction, so it can be saved, shared and reloaded without the model that
    produced it (:meth:`save`/:meth:`load`).

    Immutability is enforced, not merely intended, because rendering code and the webapp are
    written against the promise that what they draw is what was computed:

    - ``frozen=True`` — rebinding a field raises ``FrozenInstanceError``. This matters beyond
      tidiness: :attr:`is_additive` gates
      :meth:`~shapash.explainer.nlp_plotter.NlpPlotter.waterfall`, so a writable flag would be a
      correctness guard a caller could switch off by assignment.
    - The contribution arrays are sealed read-only in :meth:`__post_init__`, so the artifact is
      immutable in depth and not only at its surface.
    - ``slots=True`` — the instance has no ``__dict__``, so no attribute that is not a declared
      field can be attached to it at all. ``frozen`` already refuses ordinary assignment, but it is
      enforced through ``__setattr__`` and so is bypassable with ``object.__setattr__``; without a
      ``__dict__`` that bypass has nowhere to write an undeclared name. This is what makes "no
      display state on the artifact" a property of the class rather than a review rule: a webapp
      callback cannot stash ``_selected_row`` here even by accident. It is *not* here for memory —
      one instance exists per batch and it points at megabytes of arrays. The cost to know about:
      ``functools.cached_property`` needs a ``__dict__`` and so cannot be used on this class.
    - ``eq=False`` — value equality is meaningless here (comparing the ``pd.Series`` fields raises
      "truth value is ambiguous"), so the object compares and hashes by identity rather than
      offering an ``__eq__`` that only ever raises. Use :meth:`save`/:meth:`load` to compare runs.

    The boundary worth knowing: the pandas fields (``texts``, ``y_pred``, ``y_prob``, ``y_true``)
    are held by reference and *not* sealed — pandas has no equivalent write lock — so in-place
    edits through them are still possible. Derive a modified artifact with
    ``dataclasses.replace(explanation, ...)``, which is the supported way to vary any field.

    Design note: token-level data (``token_strings``/``values``/``base_values``/``folds_case``)
    lives here as flat fields rather than behind a separate ``NlpContributions`` sub-object.
    An earlier version nested one, which produced two mutable copies of ``label_names`` (one
    here, one on the sub-object) that could disagree, plus a redundant ``index`` field that was
    always exactly ``texts.index``. There is no batch-level reuse that justifies that split — the
    only other place per-sample contributions travel is
    :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.explain_text`'s single-text return
    (:class:`~shapash.backend.nlp_backend.NlpContributions`, the backend's own raw 3-field
    output), which every caller immediately unpacks to bare arrays and never wraps in this class.

    Attributes
    ----------
    texts : pd.Series
        The explained text samples.
    token_strings : list[list[str]]
        Tokenized (word-level) representation of each sample, variable length.
    values : list[np.ndarray]
        Per-token contribution values, one array per sample, shape ``(n_tokens_i, n_classes)``
        for multi-class models or ``(n_tokens_i,)`` for binary/regression.
    base_values : np.ndarray or None
        Baseline prediction for each sample, shape ``(n_samples, n_classes)`` or
        ``(n_samples,)``. ``None`` when the backend has no reference at all (see
        :attr:`~shapash.backend.nlp_backend.NlpBackend.reference_kind`).
    y_pred : pd.Series
        Argmax label per sample.
    y_prob : pd.DataFrame or None
        Per-class probabilities, one column per class (or a single confidence
        column), aligned to ``texts``.
    y_true : pd.Series or None
        Ground-truth labels, when supplied to ``explain()``.
    label_names : list[str] or None
        Human-readable class names in model-output order.
    folds_case : bool or None
        Whether the model's tokenizer normalises case away, from
        :meth:`~shapash.model.base.SupportsTokenization.folds_case`. Decides the default unit
        grouping in :meth:`word_importance` — see :meth:`resolve_lowercase`. ``None`` when the
        model exposes no tokenizer to ask.
    backend_name : str
        ``type(backend).name`` — which explanation method produced this artifact.
    is_additive : bool
        Whether the contributions sum to a well-defined total — read off the backend
        at ``explain()`` time. Licenses feature grouping / waterfall-style charts;
        ``False`` means summing the numbers means nothing (e.g. LIME).
    reference_kind : {"distribution", "statistics", "point", "none"}
        What kind of reference, if any, the contributions are measured against — read
        off the backend at ``explain()`` time. See ``NlpBackend.reference_kind``.
    """

    texts: pd.Series
    token_strings: list[list[str]]
    values: list[np.ndarray]
    base_values: np.ndarray | None
    y_pred: pd.Series
    y_prob: pd.DataFrame | None
    y_true: pd.Series | None
    label_names: list[str] | None
    folds_case: bool | None
    backend_name: str
    is_additive: bool
    reference_kind: Literal["distribution", "statistics", "point", "none"]

    # ClassVar, not a field: it is a shared constant, not per-explanation data, so it stays out
    # of ``fields()`` — and therefore out of ``__init__``, ``replace()`` and ``save()``.
    _SPECIAL_RE: ClassVar[re.Pattern] = re.compile(r"^\[.*\]$|^##|^\s*$")

    def __post_init__(self) -> None:
        """Seal the numeric payload, and check the fields that must agree actually do.

        ``frozen=True`` stops attributes being rebound, but a frozen dataclass still hands out
        live numpy arrays: ``explanation.values[0][0, 0] = 999`` would sail through. That is not
        hypothetical here — :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.explain`
        memoizes its result and returns a ``dataclasses.replace`` of it, which is a *shallow*
        copy, so one in-place edit would silently corrupt the cache and every artifact later
        served from it. Clearing the write flag closes that at zero copy cost: the mutation
        raises where it happens instead of surfacing as wrong numbers somewhere else.

        To derive a modified artifact, build new arrays and pass them through
        ``dataclasses.replace(explanation, values=...)`` rather than writing in place.

        The shared-index check exists because ``replace`` is now the mandated way to vary a frozen
        artifact, and ``replace`` is *silent about the fields it does not touch*. Updating
        ``texts`` without also updating ``y_pred``/``y_prob``/``y_true`` leaves an artifact whose
        halves are indexed differently — which does not raise anywhere: pandas alignment quietly
        yields all-NaN, and :meth:`confusion_matrix` zips positionally and reports plausible,
        wrong counts. Validating here rather than at each call site protects every ``replace``
        call, including ones not yet written.
        """
        for array in self.values:
            array.setflags(write=False)
        if self.base_values is not None:
            self.base_values.setflags(write=False)
        self._check_shared_index()

    def _check_shared_index(self) -> None:
        """Every index-bearing field must be indexed like :attr:`texts`.

        Raises
        ------
        ValueError
            If ``y_pred``, ``y_prob`` or ``y_true`` carries a different index from ``texts``.
        """
        index = self.texts.index
        for name, labelled in (("y_pred", self.y_pred), ("y_prob", self.y_prob), ("y_true", self.y_true)):
            if labelled is None or labelled.index.equals(index):
                continue
            raise ValueError(
                f"{name} is indexed differently from texts, so this explanation cannot be aligned "
                f"with itself: texts.index starts {list(index[:3])!r} while {name}.index starts "
                f"{list(labelled.index[:3])!r} (lengths {len(index)} and {len(labelled.index)}). "
                f"Pass {name} aligned to texts.index, or as a plain list to be indexed positionally."
            )

    # Which half of the artifact each field belongs to. Everything outside ``_CALLER_FIELDS`` is a
    # pure function of *(texts, model, backend)* — i.e. exactly what a content-addressed cache key
    # can cover — which is what makes memoizing on text content alone sound. The two sets are
    # checked against ``fields()`` at import time (bottom of this module), so a field added later
    # has to be classified rather than silently carried over from a cached run.
    _CALLER_FIELDS: ClassVar[frozenset[str]] = frozenset({"texts", "y_pred", "y_prob", "y_true"})
    _COMPUTED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "token_strings",
            "values",
            "base_values",
            "label_names",
            "folds_case",
            "backend_name",
            "is_additive",
            "reference_kind",
        }
    )

    def relabelled(self, texts: pd.Series, y_true: pd.Series | None = None) -> NlpExplanation:
        """This artifact's computed payload, carried onto another caller's index and ground truth.

        Contributions are positional, so an artifact computed for one caller is valid for any other
        caller passing the same texts in the same order, whatever index they carry. That is what
        lets :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.explain` memoize on text content
        alone: on a cache hit it re-labels the stored result through here instead of re-running the
        backend, and attaches *this* call's ground truth rather than the one the cached run happened
        to be given.

        The field list lives here rather than at that call site because this class owns it — adding
        a field means classifying it a few lines above, not remembering to thread it through another
        module.

        Parameters
        ----------
        texts : pd.Series
            The caller's texts. Expected to hold the same values in the same order as
            :attr:`texts`; only the index may differ.
        y_true : pd.Series or None
            Ground truth for this call, aligned to ``texts.index``.

        Returns
        -------
        NlpExplanation
            A new artifact sharing this one's contribution arrays, which are read-only (see
            :meth:`__post_init__`), so the sharing cannot corrupt either side.

        Raises
        ------
        ValueError
            If :attr:`_CALLER_FIELDS` names a field this method does not produce a value for, or
            vice versa — either way the two have drifted apart and the result would be wrong.
        """
        caller_values = {
            "texts": texts,
            "y_pred": self.y_pred.set_axis(texts.index),
            "y_prob": None if self.y_prob is None else self.y_prob.set_axis(texts.index),
            "y_true": y_true,
        }
        # ``replace`` is silent about every field it is not handed, so what it is handed is checked
        # against the declared partition instead of trusted. The import-time guard below only
        # establishes that each field has been *classified*; without this, classifying a new field
        # as caller-owned and then forgetting to produce it here would hand every later caller the
        # value belonging to whichever call happened to populate the cache — which is precisely the
        # failure this method exists to prevent. Together the two checks make the split total.
        if caller_values.keys() != self._CALLER_FIELDS:
            missing = sorted(self._CALLER_FIELDS - caller_values.keys())
            unexpected = sorted(caller_values.keys() - self._CALLER_FIELDS)
            raise ValueError(
                f"relabelled() and _CALLER_FIELDS disagree about which fields belong to the caller: "
                f"declared but not produced here {missing!r} (they would silently inherit the cached "
                f"run's values), produced here but not declared {unexpected!r}."
            )
        return replace(self, **caller_values)

    def __len__(self) -> int:
        return len(self.token_strings)

    @property
    def n_samples(self) -> int:
        """Number of samples in the batch."""
        return len(self.texts)

    @property
    def n_classes(self) -> int:
        """Number of model output columns the contributions carry."""
        return _n_classes(self.values, self.base_values, self.label_names)

    @property
    def has_ground_truth(self) -> bool:
        """Whether ground-truth labels were supplied to ``explain()``."""
        return self.y_true is not None

    @property
    def label_to_idx(self) -> dict[str, int]:
        """Class name to output-column index, in :attr:`label_names` order.

        Falls back to stringified column indices when the model exposes no class names, so
        callers always get a usable mapping.
        """
        names = self.label_names or [str(i) for i in range(self.n_classes)]
        return {name: i for i, name in enumerate(names)}

    @property
    def plot(self) -> NlpPlotter:
        """Rendering helpers over this explanation — ``explanation.plot.waterfall(row=0)``.

        A plain accessor holding nothing but a reference back here: it reads the artifact and
        returns figures, and never writes to it, so display choices stay arguments rather than
        state. Built fresh on each access, which keeps it out of :meth:`save`'s payload and out
        of ``dataclasses.fields``.

        Returns
        -------
        NlpPlotter

        Examples
        --------
        >>> explanation, _ = NlpExplanation.load("run.zip")  # no model, no backend
        >>> explanation.plot.waterfall(row=0, label_idx=1).show()
        """
        # Imported here, not at module scope: this module is the persistence boundary and stays
        # free of plotly/dash so that loading a saved artifact costs no rendering imports.
        from shapash.explainer.nlp_plotter import NlpPlotter  # noqa: PLC0415

        return NlpPlotter(self)

    def resolve_lowercase(self, lowercase: bool | None = None) -> bool:
        """Decide whether word units should be case-folded, deferring to the model by default.

        One source of truth for the question, so a caller that has to normalise units the same
        way (the webapp's exclusion vocabulary) cannot drift from :meth:`word_importance`.

        Parameters
        ----------
        lowercase : bool, optional
            Explicit caller override. ``None`` (default) derives the answer from ``folds_case``.

        Returns
        -------
        bool
            ``True`` when ``AWFUL`` and ``awful`` should aggregate into one unit.

        Notes
        -----
        With no override, an *uncased* model folds and a *cased* model does not: on an uncased
        tokenizer both spellings are the same input ids, so keeping them apart splits one word
        into variants the model provably cannot distinguish; on a cased one they are different
        inputs whose attributions differ for real reasons, and merging would hide that.

        When ``folds_case`` is ``None`` — no tokenizer to probe, e.g. a bare LIME
        ``classifier_fn`` — this falls back to folding, which keeps a corpus-level ranking from
        fragmenting on a question that cannot be answered here.
        """
        if lowercase is not None:
            return lowercase
        return True if self.folds_case is None else self.folds_case

    def word_importance(
        self,
        label_idx: int,
        n_top: int = 20,
        filter_special: bool = True,
        filter_punctuation: bool = True,
        lowercase: bool | None = None,
        filter_sign: str = "all",
        exclude_words: set[str] | None = None,
        sample_indices: list[int] | None = None,
    ) -> pd.Series:
        """Aggregate token contributions by word across all samples for one class.

        For each unique (stripped, by default lowercased) token string that appears in
        the batch, computes the mean contribution across all its occurrences.

        Parameters
        ----------
        label_idx : int
            Index of the class to compute importance for.
        n_top : int
            Maximum number of words to return, sorted by ``|mean contribution|``.
        filter_special : bool
            If ``True``, skip empty strings, ``[CLS]``/``[SEP]``-style bracket
            tokens, and ``##subword`` wordpiece prefixes.
        filter_punctuation : bool
            If ``True`` (default), skip units made entirely of punctuation (``.``, ``,``, ``!!!``).
            Word segmentation emits punctuation as its own unit so it stays visible and additive in
            a *local* explanation; in a corpus-level ranking it is almost always noise, so it is
            hidden here unless explicitly asked for.
        lowercase : bool, optional
            Case-fold each unit before aggregating, so ``AWFUL``, ``Awful`` and ``awful``
            form one row. ``None`` (default) derives this from the model's own tokenizer
            via :meth:`resolve_lowercase` — folding on an uncased model, keeping case on a
            cased one — so pass a bool only to override that.
            Folding also makes the ranking comparable across backends: SHAP labels units by
            slicing the *source text* (keeping its casing) while Captum/LIG labels them from
            tokenizer output (already normalised), so on an uncased model the two would
            otherwise report different vocabularies for one corpus.
            The per-instance sentence highlight always keeps the original casing, whatever
            this is set to.
        filter_sign : {"all", "positive", "negative"}
            If ``"positive"``, keep only words with positive mean contribution.
            If ``"negative"``, keep only words with negative mean contribution.
        exclude_words : set[str], optional
            Additional set of word strings to exclude (e.g. user-selected
            stopwords or corpus words to hide). Matched against the same
            normalised key, so under ``lowercase`` the exclusion is case-insensitive.
        sample_indices : list[int], optional
            Positional indices of the samples to include in the aggregation.
            When ``None`` (default) all samples are used.

        Returns
        -------
        pd.Series
            Word → mean contribution, sorted by absolute value descending.
        """

        fold = self.resolve_lowercase(lowercase)

        def _key(token: str) -> str:
            return token.lower() if fold else token

        _exclude: set[str] = {_key(w) for w in exclude_words} if exclude_words else set()
        word_contribs: dict[str, list[float]] = {}
        iter_data = (
            ((self.token_strings[i], self.values[i]) for i in sample_indices)
            if sample_indices is not None
            else zip(self.token_strings, self.values, strict=True)
        )
        for tokens, vals in iter_data:
            vals_1d: np.ndarray = select_label_column(vals, label_idx)
            for tok, val in zip(tokens, vals_1d, strict=True):
                tok_clean = tok.strip()
                if filter_special and self._SPECIAL_RE.match(tok_clean):
                    continue
                if filter_punctuation and is_punctuation(tok_clean):
                    continue
                key = _key(tok_clean)
                if key in _exclude:
                    continue
                if key not in word_contribs:
                    word_contribs[key] = []
                word_contribs[key].append(float(val))

        importance = pd.Series({w: float(np.mean(vs)) for w, vs in word_contribs.items()})
        importance = importance.reindex(importance.abs().sort_values(ascending=False).index)

        if filter_sign == "positive":
            importance = importance[importance > 0]
        elif filter_sign == "negative":
            importance = importance[importance < 0]

        return importance.head(n_top)

    def confusion_matrix(self) -> np.ndarray:
        """Counts of ground truth against predictions, in :attr:`label_to_idx` order.

        Returns
        -------
        numpy.ndarray
            Square integer array of shape ``(n_classes, n_classes)`` where ``cm[true, pred]``
            counts the samples whose true class is ``true`` and predicted class is ``pred`` —
            the scikit-learn orientation, and the layout
            :func:`~shapash.plots.plot_confusion_matrix.plot_confusion_matrix` expects.

        Raises
        ------
        ValueError
            If no ground truth was supplied to ``explain()``.

        Notes
        -----
        Labels are matched by their string form through :attr:`label_to_idx`; a sample whose
        true or predicted label is not among the known classes is skipped rather than raising,
        so an unexpected label costs one row of the matrix and not the whole plot.
        """
        if self.y_true is None:
            raise ValueError(
                "A confusion matrix needs ground truth, but this explanation has no y_true. "
                "Pass y_true to explain() to enable it."
            )

        idx_of = self.label_to_idx
        k = len(idx_of)
        cm = np.zeros((k, k), dtype=int)
        for true_label, pred_label in zip(self.y_true.tolist(), self.y_pred.tolist(), strict=True):
            t, p = idx_of.get(str(true_label), -1), idx_of.get(str(pred_label), -1)
            if t >= 0 and p >= 0:
                cm[t, p] += 1
        return cm

    def save(self, path: str | Path, scatter_xy: np.ndarray | None = None) -> None:
        """Persist this explanation (no model or backend required to reload).

        Writes a single zip file containing a plain-text ``meta.json`` (readable
        without shapash) plus long/tidy parquet tables for contributions, base
        values, samples (texts/predictions/ground truth) and, optionally, a 2-D
        scatter projection.

        Parameters
        ----------
        path : str or Path
            Destination file. Any extension is accepted; ``.shxpl`` is a reasonable
            convention but not enforced.
        scatter_xy : np.ndarray, optional
            Pre-computed 2-D projection to bundle alongside the explanation, shape
            ``(n_samples, 2)`` (the same array :meth:`~NlpExplainer.compute_projection`
            returns).
        """
        contrib_df, base_df, values_ndim = _contributions_to_frames(self.token_strings, self.values, self.base_values)
        samples_df = _samples_to_frame(self.texts, self.y_pred, self.y_prob, self.y_true)

        meta = {
            "format_version": _FORMAT_VERSION,
            "shapash_version": _shapash_version,
            "created_at": datetime.now(UTC).isoformat(),
            "backend_name": self.backend_name,
            "is_additive": self.is_additive,
            "reference_kind": self.reference_kind,
            "label_names": self.label_names,
            "folds_case": self.folds_case,
            # Authoritative counts, not re-derived from the tidy tables on load: a sample with
            # zero tokens has no row in contrib_df at all, so inferring n_samples from the table
            # would silently drop it (and n_classes has nothing to read from an all-empty batch).
            "n_samples": len(self.texts),
            "n_classes": _n_classes(self.values, self.base_values, self.label_names),
            "values_ndim": values_ndim,
            "has_base_values": base_df is not None,
            "has_ground_truth": self.y_true is not None,
            "has_scatter": scatter_xy is not None,
        }

        with zipfile.ZipFile(Path(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
            _write_parquet(zf, "contributions.parquet", contrib_df)
            if base_df is not None:
                _write_parquet(zf, "base_values.parquet", base_df)
            _write_parquet(zf, "samples.parquet", samples_df)
            if scatter_xy is not None:
                scatter_arr = np.asarray(scatter_xy)
                scatter_df = pd.DataFrame(
                    {"sample_idx": np.arange(len(scatter_arr)), "x": scatter_arr[:, 0], "y": scatter_arr[:, 1]}
                )
                _write_parquet(zf, "scatter.parquet", scatter_df)

    @classmethod
    def load(cls, path: str | Path) -> tuple[NlpExplanation, np.ndarray | None]:
        """Restore an explanation saved by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            File written by :meth:`save`.

        Returns
        -------
        explanation : NlpExplanation
            The restored explanation.
        scatter_xy : np.ndarray or None
            The projection array bundled at save time, or ``None``.

        Raises
        ------
        ValueError
            If the file's ``format_version`` is not one this shapash version reads.
        """
        with zipfile.ZipFile(Path(path), "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            version = meta.get("format_version")
            if version not in _SUPPORTED_FORMAT_VERSIONS:
                raise ValueError(
                    f"Unsupported NlpExplanation format_version={version!r}; this shapash version "
                    f"({_shapash_version}) reads version(s) {_SUPPORTED_FORMAT_VERSIONS}. "
                    "Re-save this explanation with a compatible shapash version."
                )
            contrib_df = _read_parquet(zf, "contributions.parquet")
            base_df = _read_parquet(zf, "base_values.parquet") if meta.get("has_base_values", True) else None
            samples_df = _read_parquet(zf, "samples.parquet")
            scatter_xy = None
            if meta.get("has_scatter"):
                scatter_df = _read_parquet(zf, "scatter.parquet")
                scatter_xy = scatter_df[["x", "y"]].to_numpy()

        token_strings, values, base_values = _frames_to_contributions(
            contrib_df, base_df, meta["values_ndim"], meta["n_samples"], meta["n_classes"]
        )
        texts, y_pred, y_true, y_prob = _frame_to_samples(samples_df)

        explanation = cls(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=base_values,
            y_pred=y_pred,
            y_prob=y_prob,
            y_true=y_true,
            label_names=meta.get("label_names"),
            folds_case=meta.get("folds_case"),
            backend_name=meta["backend_name"],
            is_additive=meta["is_additive"],
            reference_kind=meta["reference_kind"],
        )
        return explanation, scatter_xy


def _write_parquet(zf: zipfile.ZipFile, name: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    zf.writestr(name, buf.getvalue())


def _read_parquet(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(zf.read(name)), engine="pyarrow")


def _contributions_to_frames(
    token_strings: list[list[str]], values: list[np.ndarray], base_values: np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame | None, int]:
    """Ragged per-sample arrays -> long/tidy ``(contributions_df, base_values_df, values_ndim)``.

    ``values_ndim`` records whether the original per-sample arrays were 1-D
    (binary/regression) or 2-D (multi-class), so :func:`_frames_to_contributions` can
    restore the exact original shape rather than always returning 2-D arrays.
    """
    values_ndim = 2 if values and values[0].ndim == 2 else 1

    contrib_rows: list[tuple[int, int, str, int, float]] = []
    for sample_idx, (tokens, vals) in enumerate(zip(token_strings, values, strict=True)):
        vals_2d = vals if vals.ndim == 2 else vals[:, None]
        for token_idx, token in enumerate(tokens):
            for class_idx in range(vals_2d.shape[1]):
                contrib_rows.append((sample_idx, token_idx, token, class_idx, float(vals_2d[token_idx, class_idx])))
    contrib_df = pd.DataFrame(contrib_rows, columns=["sample_idx", "token_idx", "token", "class_idx", "value"])

    if base_values is None:
        return contrib_df, None, values_ndim

    base_2d = base_values if base_values.ndim == 2 else base_values[:, None]
    base_rows = [
        (sample_idx, class_idx, float(base_2d[sample_idx, class_idx]))
        for sample_idx in range(base_2d.shape[0])
        for class_idx in range(base_2d.shape[1])
    ]
    base_df = pd.DataFrame(base_rows, columns=["sample_idx", "class_idx", "base_value"])
    return contrib_df, base_df, values_ndim


def _n_classes(values: list[np.ndarray], base_values: np.ndarray | None, label_names: list[str] | None) -> int:
    """Number of classes for a batch, robust to an all-empty-tokens / no-baseline batch.

    Needed as ``meta.json`` state rather than re-derived from the tidy tables on load: a
    sample with zero tokens leaves no row in ``contributions.parquet``, so if *every*
    sample in the batch happened to be empty *and* there is no baseline either, the
    tables alone carry no evidence of ``n_classes`` at all.
    """
    if base_values is not None:
        return base_values.shape[1] if base_values.ndim == 2 else 1
    for vals in values:
        if vals.shape[0] > 0:
            return vals.shape[1] if vals.ndim == 2 else 1
    return len(label_names) if label_names else 1


def _frames_to_contributions(
    contrib_df: pd.DataFrame,
    base_df: pd.DataFrame | None,
    values_ndim: int,
    n_samples: int,
    n_classes: int,
) -> tuple[list[list[str]], list[np.ndarray], np.ndarray | None]:
    """Inverse of :func:`_contributions_to_frames`.

    ``n_samples``/``n_classes`` come from ``meta.json`` (see :func:`_n_classes`) rather
    than being re-derived from the tables, which cannot see a trailing all-empty sample.
    """
    groups = dict(tuple(contrib_df.groupby("sample_idx"))) if len(contrib_df) else {}
    token_strings: list[list[str]] = []
    values: list[np.ndarray] = []
    for sample_idx in range(n_samples):
        group = groups.get(sample_idx)
        if group is None or group.empty:
            token_strings.append([])
            values.append(np.zeros((0, n_classes) if values_ndim == 2 else (0,), dtype=float))
            continue
        tokens = group.drop_duplicates("token_idx").sort_values("token_idx")["token"].tolist()
        arr = np.zeros((len(tokens), n_classes), dtype=float)
        arr[group["token_idx"].to_numpy(), group["class_idx"].to_numpy()] = group["value"].to_numpy()
        token_strings.append(tokens)
        values.append(arr if values_ndim == 2 else arr[:, 0])

    if base_df is None:
        return token_strings, values, None

    base_values = np.zeros((n_samples, n_classes), dtype=float)
    if len(base_df):
        base_values[base_df["sample_idx"].to_numpy(), base_df["class_idx"].to_numpy()] = base_df[
            "base_value"
        ].to_numpy()
    if values_ndim == 1:
        base_values = base_values[:, 0]
    return token_strings, values, base_values


def _samples_to_frame(
    texts: pd.Series, y_pred: pd.Series, y_prob: pd.DataFrame | None, y_true: pd.Series | None
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(texts)),
            "orig_index": list(texts.index),
            "text": texts.to_numpy(),
            "y_pred": y_pred.to_numpy(),
        }
    )
    if y_true is not None:
        df["y_true"] = y_true.to_numpy()
    if y_prob is not None:
        for col in y_prob.columns:
            df[f"{_PROB_PREFIX}{col}"] = y_prob[col].to_numpy()
    return df


def _frame_to_samples(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.DataFrame | None]:
    index = pd.Index(df["orig_index"].tolist())
    texts = pd.Series(df["text"].tolist(), index=index, name="text")
    y_pred = pd.Series(df["y_pred"].tolist(), index=index, name="prediction")
    y_true = pd.Series(df["y_true"].tolist(), index=index, name="ground_truth") if "y_true" in df.columns else None

    prob_cols = [c for c in df.columns if c.startswith(_PROB_PREFIX)]
    y_prob = None
    if prob_cols:
        y_prob = pd.DataFrame(
            {c[len(_PROB_PREFIX) :]: df[c].tolist() for c in prob_cols},
            index=index,
        )
    return texts, y_pred, y_true, y_prob


# Checked once, at import, rather than on every ``relabelled`` call: an unclassified field is a
# coding error, not a runtime condition, and this way it surfaces as an ImportError on ``import
# shapash`` instead of waiting for a cache hit to expose it. This is what makes ``relabelled``'s
# ``replace`` safe — ``replace`` is silent about fields it is not handed, so the guarantee that
# every field has been *considered* has to come from somewhere, and it comes from here.
_unclassified = {f.name for f in fields(NlpExplanation)} - (
    NlpExplanation._CALLER_FIELDS | NlpExplanation._COMPUTED_FIELDS
)
if _unclassified:
    raise ValueError(
        f"NlpExplanation field(s) {sorted(_unclassified)!r} are classified as neither caller-owned "
        f"nor computed. Add each one to NlpExplanation._CALLER_FIELDS if its value comes from the "
        f"caller of explain(), or to _COMPUTED_FIELDS if it is determined by (texts, model, "
        f"backend) and may therefore be served from cache."
    )
