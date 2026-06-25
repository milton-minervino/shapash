"""Word-level importance plot for NLP explanations."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objs as go

_COLOR_POSITIVE = "#1f77b4"
_COLOR_NEGATIVE = "#d62728"


def plot_word_importance(
    word_imp: pd.Series,
    title: str = "Word importance",
    width: int = 900,
    height: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of mean token-level SHAP contributions per word.

    Words are displayed sorted by absolute mean contribution (highest at top).
    Positive mean contributions are shown in blue, negative in red.

    Parameters
    ----------
    word_imp : pd.Series
        Word → mean SHAP contribution, already sorted by ``|value|`` descending
        (as returned by ``NlpContributions.word_importance``).
    title : str
        Figure title.
    width : int
        Figure width in pixels.
    height : int, optional
        Figure height in pixels. Defaults to ``max(400, 30 * n_words + 120)``.

    Returns
    -------
    go.Figure

    Examples
    --------
    >>> imp = contributions.word_importance(label_idx=1)
    >>> fig = plot_word_importance(imp, title="Word importance — joy")
    >>> fig.show()
    """
    words = list(word_imp.index)
    values = list(word_imp.values)
    colors = [_COLOR_POSITIVE if v >= 0 else _COLOR_NEGATIVE for v in values]

    # Plotly renders y-axis bottom-to-top; reverse so highest-importance word appears at top.
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=words[::-1],
            orientation="h",
            marker_color=colors[::-1],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Mean SHAP contribution",
        width=width,
        height=height or max(400, 30 * len(words) + 120),
        plot_bgcolor="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(automargin=True),
        margin=dict(l=20, r=20, t=60, b=40),
    )

    return fig
