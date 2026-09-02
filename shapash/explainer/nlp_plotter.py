"""``NlpPlotter`` — rendering helpers bound to one :class:`NlpExplanation`.

Reached as ``explanation.plot``. It holds a reference to the artifact and nothing else: every
call slices that artifact, hands the slice to a pure function in :mod:`shapash.plots`, and
returns the figure. No display state is stored anywhere — pass different arguments and you get
a different figure from the *same* untouched explanation.

Why a separate object rather than methods on :class:`NlpExplanation` itself: the explanation is
the persistence boundary (``save``/``load`` with a ``format_version``), so keeping rendering off
it keeps the dataclass a dataclass. It also mirrors the tabular side, where ``SmartExplainer``
exposes :class:`~shapash.explainer.smart_plotter.SmartPlotter` as ``.plot``.

The webapp deliberately does *not* go through here: its panels render ``dcc.Store`` datapoints
via :func:`~shapash.webapp.nlp_components.datapoint.unpack_datapoint`, because a what-if
datapoint has no row in the artifact at all. This accessor serves notebook use, including a
snapshot reloaded with :meth:`NlpExplanation.load`, which has no model and no backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dash import html
from plotly import graph_objs as go

from shapash.explainer.nlp_explanation import select_label_column
from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.plots.plot_waterfall import plot_waterfall
from shapash.plots.plot_word_importance import plot_word_importance

if TYPE_CHECKING:
    from shapash.explainer.nlp_explanation import NlpExplanation


class NlpPlotter:
    """Plots over one :class:`~shapash.explainer.nlp_explanation.NlpExplanation`.

    Not instantiated directly — reach it as ``explanation.plot``.

    Parameters
    ----------
    explanation : NlpExplanation
        The artifact to render. Held by reference and never written to.

    Examples
    --------
    >>> explanation, _ = NlpExplanation.load("run.zip")  # no model needed
    >>> explanation.plot.waterfall(row=0, label_idx=1).show()
    """

    def __init__(self, explanation: NlpExplanation) -> None:
        self._exp = explanation

    def __repr__(self) -> str:
        exp = self._exp
        return f"<NlpPlotter over {len(exp)} sample(s), backend={exp.backend_name!r}>"

    # ── internals ───────────────────────────────────────────────────────────────────────
    def _check_label_idx(self, label_idx: int) -> int:
        """Fail with the class list rather than a bare numpy ``IndexError``."""
        n_classes = self._exp.n_classes
        if not -n_classes <= label_idx < n_classes:
            names = self._exp.label_names
            known = f" Classes: {names}." if names else ""
            raise IndexError(f"label_idx={label_idx} is out of range for {n_classes} output column(s).{known}")
        return label_idx % n_classes

    def _label_name(self, label_idx: int) -> str | None:
        names = self._exp.label_names
        return names[label_idx] if names is not None and label_idx < len(names) else None

    def _slice(self, row: int, label_idx: int) -> tuple[list[str], np.ndarray, float | None]:
        """One sample, one class — the shape every per-instance plot function takes.

        Returns ``(tokens, 1-D values, base_value)``. ``base_value`` is ``None`` when the
        backend had no reference at all (``reference_kind == "none"``).
        """
        exp = self._exp
        if not -len(exp) <= row < len(exp):
            raise IndexError(f"row={row} is out of range for {len(exp)} sample(s).")

        tokens = exp.token_strings[row]
        values = select_label_column(exp.values[row], label_idx)

        base = exp.base_values
        base_value: float | None = None
        if base is not None:
            base_value = float(base[row, label_idx]) if base.ndim == 2 else float(base[row])

        return list(tokens), np.asarray(values, dtype=float), base_value

    # ── per-instance plots ──────────────────────────────────────────────────────────────
    def tokens(
        self,
        row: int = 0,
        label_idx: int = 0,
        max_tokens: int | None = None,
        title: str | None = None,
        width: int = 900,
        height: int | None = None,
    ) -> go.Figure:
        """Bar chart of one sample's token contributions for one class.

        Parameters
        ----------
        row : int
            Positional index of the sample (``0`` is the first row, negatives count from the
            end). This is a position, not a label from ``texts.index``.
        label_idx : int
            Index of the class to display, in ``label_names`` order.
        max_tokens : int, optional
            Keep only the ``max_tokens`` highest-magnitude tokens, in sentence order.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width, height : int, optional
            Figure size in pixels.

        Returns
        -------
        plotly.graph_objs.Figure
        """
        label_idx = self._check_label_idx(label_idx)
        toks, values, _ = self._slice(row, label_idx)
        name = self._label_name(label_idx)
        if title is None:
            title = f"Token contributions — {name}" if name else "Token contributions"
        return plot_token_highlight(
            tokens=toks, values=values, title=title, max_tokens=max_tokens, width=width, height=height
        )

    def waterfall(
        self,
        row: int = 0,
        label_idx: int = 0,
        min_pct: float = 0.10,
        filter_special: bool = True,
        title: str | None = None,
        width: int | None = None,
    ) -> go.Figure:
        """Waterfall decomposing one prediction from the baseline into token contributions.

        Parameters
        ----------
        row : int
            Positional index of the sample (see :meth:`tokens`).
        label_idx : int
            Index of the class to display, in ``label_names`` order.
        min_pct : float
            Fraction of the largest absolute contribution below which tokens collapse into a
            single "other" bar. Range [0, 1].
        filter_special : bool
            Drop ``[CLS]``/``[SEP]``-style and ``##subword`` tokens.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width : int, optional
            Figure width in pixels.

        Returns
        -------
        plotly.graph_objs.Figure

        Raises
        ------
        ValueError
            If the producing backend is not additive. A waterfall's running total only means
            something when the contributions sum to the prediction, which is exactly what
            :attr:`~shapash.explainer.nlp_explanation.NlpExplanation.is_additive` records — so
            this refuses rather than drawing a chart whose arithmetic is meaningless.
        """
        exp = self._exp
        if not exp.is_additive:
            raise ValueError(
                f"A waterfall decomposes a prediction into contributions that sum to it, but the "
                f"{exp.backend_name!r} backend is not additive, so the running total would be "
                f"meaningless. Use .plot.tokens() or .plot.sentence() instead."
            )
        label_idx = self._check_label_idx(label_idx)
        toks, values, base_value = self._slice(row, label_idx)
        name = self._label_name(label_idx)
        if title is None:
            title = f"Token contributions — {name}" if name else "Token contributions"
        return plot_waterfall(
            tokens=toks,
            values=values,
            base_value=base_value,
            min_pct=min_pct,
            filter_special=filter_special,
            title=title,
            width=width,
        )

    def sentence(self, row: int = 0, label_idx: int = 0) -> html.Div:
        """One sample rendered inline, each token background-shaded by its contribution.

        Parameters
        ----------
        row : int
            Positional index of the sample (see :meth:`tokens`).
        label_idx : int
            Index of the class to display, in ``label_names`` order.

        Returns
        -------
        dash.html.Div
            A Dash component. In a notebook, display it inside a
            ``jupyter_dash``/``dash`` app, or use :meth:`tokens` for a standalone figure.
        """
        label_idx = self._check_label_idx(label_idx)
        toks, values, base_value = self._slice(row, label_idx)
        return plot_sentence_highlight(tokens=toks, values=values, base_value=base_value)

    # ── batch-level plots ───────────────────────────────────────────────────────────────
    def words(
        self,
        label_idx: int = 0,
        n_top: int = 20,
        title: str | None = None,
        width: int = 900,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Corpus-level word importance for one class.

        Parameters
        ----------
        label_idx : int
            Index of the class to aggregate, in ``label_names`` order.
        n_top : int
            Number of words to show, ranked by ``|mean contribution|``.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width, height : int, optional
            Figure size in pixels.
        **kwargs
            Forwarded to
            :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.word_importance` —
            ``filter_special``, ``filter_punctuation``, ``lowercase``, ``filter_sign``,
            ``exclude_words``, ``sample_indices``.

        Returns
        -------
        plotly.graph_objs.Figure
        """
        label_idx = self._check_label_idx(label_idx)
        word_imp = self._exp.word_importance(label_idx=label_idx, n_top=n_top, **kwargs)
        name = self._label_name(label_idx)
        if title is None:
            title = f"Word importance — {name}" if name else "Word importance"
        return plot_word_importance(word_imp, title=title, width=width, height=height)

    def confusion(
        self,
        normalize: str | None = None,
        title: str = "Confusion matrix",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """Confusion matrix of predictions against ground truth.

        Parameters
        ----------
        normalize : {None, "true"}, optional
            ``"true"`` divides each row by its sum so cells show recall. ``None`` shows counts.
        title : str
            Figure title.
        width, height : int, optional
            Figure size in pixels.

        Returns
        -------
        plotly.graph_objs.Figure

        Raises
        ------
        ValueError
            If no ground truth was supplied to ``explain()``.
        """
        exp = self._exp
        if not exp.has_ground_truth:
            raise ValueError(
                "A confusion matrix needs ground truth, but this explanation has no y_true. "
                "Pass y_true to explain() to enable it."
            )
        cm = exp.confusion_matrix()
        labels = list(exp.label_to_idx)
        return plot_confusion_matrix(cm, labels, normalize=normalize, title=title, width=width, height=height)
