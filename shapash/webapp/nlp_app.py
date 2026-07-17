"""Minimal Dash webapp for NLP text classification explanations.

Prototype bridge toward Phase 5b (composable WebappComponents). Panels are being extracted into
``WebappComponent``s one at a time (see ``shapash/webapp/nlp_components/``); global word importance,
the dataset table, scatter, and error analysis are still built inline here pending their own
extraction. All tabular-only SmartApp panels (violin, cluster, scatter prediction picking) beyond
what NLP needs are absent rather than disabled.
"""

from __future__ import annotations

import re

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_word_importance import plot_word_importance
from shapash.webapp.nlp_components import (
    CounterfactualComponent,
    DataEditorComponent,
    SentenceHighlightComponent,
    SimilarExamplesComponent,
    WaterfallComponent,
    pack_datapoint,
)
from shapash.webapp.nlp_view import NlpView

_APPLY_STORE = "whatif-apply-store"
_CURRENT_STORE = "current-datapoint"

_SPECIAL_RE = re.compile(r"^\[.*\]$|^##|^\s*$")
_HIDDEN = {"display": "none"}
# The visible tab body is a flex-column item that fills the bodies container (which itself fills the
# card). This unbroken flex chain is what lets a body's inner content use height:100% / flex:1 (e.g. the
# dataset grid) to fill the panel; a plain `display:block` here would collapse to content height.
_VISIBLE = {"display": "flex", "flexDirection": "column", "flex": "1 1 auto", "minHeight": "0", "overflowY": "auto"}
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _compose_selection(
    selected_indices: list[int] | None,
    cell_indices: list[int] | None,
    error_positions: set[int] | None,
) -> list[int] | None:
    """Intersect the app's active sample filters into one index list.

    Each argument is an independent filter that may be inactive (``None``): the scatter box/lasso
    selection, the confusion-matrix cell, and — when the errors-only switch is on — the set of
    misclassified positions. Active filters intersect; returns ``None`` when none are active.
    """
    combined = selected_indices
    if cell_indices is not None:
        cell_set = set(cell_indices)
        combined = list(cell_indices) if combined is None else [i for i in combined if i in cell_set]
    if error_positions is not None:
        combined = sorted(error_positions) if combined is None else [i for i in combined if i in error_positions]
    return combined


def _cell_from_click(click_data: dict | None, name_to_idx: dict[str, int]) -> tuple[int, int] | None:
    """Resolve a confusion-matrix cell click to ``(pred_idx, true_idx)``.

    Heatmap ``clickData`` does not reliably include ``customdata`` the way scatter traces do, so
    prefer it when present but fall back to mapping the predicted (``x``) and true (``y``) axis label
    names back to their class indices. Returns ``None`` for an empty or unrecognised click.
    """
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    custom = point.get("customdata")
    if custom is not None and len(custom) == 2:
        return int(custom[0]), int(custom[1])
    if point.get("x") in name_to_idx and point.get("y") in name_to_idx:
        return name_to_idx[point["x"]], name_to_idx[point["y"]]
    return None


def _empty_word_fig(message: str) -> go.Figure:
    """Placeholder figure with a centred hint, for the per-cell word charts before a cell is picked."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color="#888888"), xref="paper", yref="paper")
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


class NlpWebApp:
    """Minimal Dash webapp driven by an ``NlpExplainer``.

    The layout follows an "overview → filter → detail" funnel:

    Top row — global "where to look" panels that filter the table below:
    - **Global word importance** — mean SHAP contribution per unique word for
      the selected class (updates on any control change or scatter selection).
      Its own controls (top-K slider, positive/negative sign filter, corpus
      word multi-select for manual stopword exclusion) live inside this panel.
    - **Sample scatter** *(optional)* — 2-D projection of the text embeddings,
      coloured by prediction or ground-truth label.  Draw a box or lasso to
      filter the table and word importance to the selected subset.

    Hub — full-width:
    - **Dataset table** — text samples with predicted (and optional ground-truth)
      labels; click a row to populate the local contribution panel.  Filtered
      to the scatter selection and/or a clicked word importance bar.

    Detail-on-demand — full-width:
    - **Local contributions** — inline sentence highlight as the primary view.
      An optional waterfall chart (toggle with a radio button) groups tokens
      below a configurable contribution threshold into a single "other" bar.

    Parameters
    ----------
    explainer : NlpExplainer
        A compiled ``NlpExplainer`` instance (``compile()`` must have been called).
    scatter_xy : np.ndarray, optional
        Pre-computed 2-D projection, shape ``(n_samples, 2)``.  When provided,
        a scatter panel is added to the layout.  Compute with PaCMAP, UMAP,
        t-SNE, PCA, etc. and pass the result here — Shapash does not perform
        the projection itself to avoid heavy optional dependencies.
    """

    def __init__(self, explainer, scatter_xy: np.ndarray | None = None) -> None:
        if explainer.contributions is None:
            raise RuntimeError("NlpExplainer.compile() must be called before launching the webapp.")
        if scatter_xy is not None:
            scatter_xy = np.asarray(scatter_xy)
            n = len(explainer.texts)
            if scatter_xy.shape != (n, 2):
                raise ValueError(f"scatter_xy must have shape ({n}, 2), got {scatter_xy.shape}")
        self._scatter_xy: np.ndarray | None = scatter_xy

        # What-if Lab: read-only view + live engine (the explainer itself when it is live).
        # Components self-disable via their `requires` when the engine lacks a capability
        # (e.g. an explainer restored from a snapshot holds no model). `self._components` itself is
        # assembled later in `_build_layout` (it needs `_full_table_records`, not ready yet here).
        self._view = NlpView(explainer)
        self._engine = explainer

        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Shapash — NLP Explainer"
        # Pin the document to the viewport so only the inner panels scroll, never the page. The
        # 100vh shell needs html/body at full height with no default margin; overflow:hidden is the
        # guarantee against any residual sub-pixel overflow producing a page scrollbar.
        self.app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { height: 100%; margin: 0; overflow: hidden; }
            #react-entry-point { height: 100%; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""
        self._build_layout()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        contrib = self._view.contributions
        label_names = contrib.label_names or [str(i) for i in range(self._view.n_classes)]
        # Predicted-label name → class index, shared by the confusion matrix and (via
        # NlpView.label_to_idx) by SentenceHighlightComponent's predicted-class sync callback.
        self._label_to_idx = self._view.label_to_idx
        n = self._view.n_samples

        # ── Table records — always include _orig_idx for scatter filtering ──
        records: dict[str, list] = {
            "_orig_idx": list(range(n)),
            "text": self._view.texts.tolist(),
            "prediction": (self._view.y_pred.tolist() if self._view.y_pred is not None else [""] * n),
        }
        if self._view.y_true is not None:
            records["ground_truth"] = self._view.y_true.tolist()

        # Single "Probability" column — confidence of the predicted class.
        # max(axis=1) works for both binary and multiclass since the predicted
        # label is always the argmax, regardless of how columns are named.
        y_prob: pd.DataFrame | None = self._view.y_prob
        if y_prob is not None:
            records["probability"] = y_prob.max(axis=1).tolist()

        table_df = pd.DataFrame(records)

        # Full records cached for scatter-driven re-filtering in callbacks
        self._full_table_records: list[dict] = table_df.to_dict("records")

        has_gt = "ground_truth" in table_df.columns
        self._has_gt = has_gt
        has_prob = y_prob is not None
        column_defs: list[dict] = [
            {"field": "text", "headerName": "Text", "flex": 3, "tooltipField": "text", "filter": "agTextColumnFilter"},
            {"field": "prediction", "headerName": "Prediction", "flex": 1, "filter": "agTextColumnFilter"},
        ]
        if has_gt:
            column_defs.append(
                {"field": "ground_truth", "headerName": "Ground Truth", "flex": 1, "filter": "agTextColumnFilter"}
            )
        if has_prob:
            column_defs.append(
                {
                    "field": "probability",
                    "headerName": "Probability",
                    "flex": 1,
                    "filter": "agNumberColumnFilter",
                    "valueFormatter": {"function": "params.value != null ? params.value.toFixed(3) : ''"},
                }
            )

        # ── Corpus word list for the exclusion multi-select ────────────
        all_words = sorted(
            {
                tok.strip()
                for sample_tokens in contrib.token_strings
                for tok in sample_tokens
                if tok.strip() and not _SPECIAL_RE.match(tok.strip())
            }
        )

        # ── Scatter panel content (only when scatter_xy is provided) ──
        scatter_col_content = None
        if self._scatter_xy is not None:
            color_options = [{"label": "Prediction", "value": "prediction"}]
            if self._view.y_true is not None:
                color_options.append({"label": "Ground Truth", "value": "ground_truth"})
            color_options.append({"label": "Word contribution", "value": "word_contribution"})

            scatter_col_content = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H6("Sample Space", className="fw-bold mb-0"),
                                width="auto",
                                className="align-self-center",
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="color-by",
                                    options=color_options,
                                    value="prediction",
                                    clearable=False,
                                    style={"width": "160px", "fontSize": "0.9em"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="scatter-word-select",
                                    options=[{"label": w, "value": w} for w in all_words],
                                    value=[],
                                    multi=True,
                                    clearable=True,
                                    placeholder="Select words…",
                                    style={"display": "none", "minWidth": "180px", "fontSize": "0.9em"},
                                ),
                                width="auto",
                                className="align-self-center",
                            ),
                        ],
                        className="align-items-center mb-2",
                    ),
                    html.Small(
                        "Box/lasso or click to filter. Click a word bar to color by its SHAP contribution.",
                        className="text-muted d-block mb-2",
                    ),
                    dcc.Graph(
                        id="scatter-plot",
                        figure=self._build_scatter_fig("prediction"),
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["autoScale2d", "resetScale2d"],
                            "responsive": True,
                        },
                        # Grow to fill the card's remaining height so the scatter
                        # matches the taller word-importance panel beside it.
                        style={"flex": "1 1 auto", "minHeight": "340px"},
                    ),
                ],
                # Flex column so the graph above can stretch to the panel height.
                style={"display": "flex", "flexDirection": "column", "height": "100%"},
            )

        # ── Error Analysis panel (confusion matrix + per-cell word importance) ──
        # Only meaningful with ground truth. The matrix is a global selection driver like the
        # scatter: clicking a cell writes ``error-cell`` (see callbacks), which composes with any
        # scatter selection to filter the table and the main Word Importance tab, and drives the two
        # per-cell word charts below (words toward the predicted vs. the true class).
        error_analysis_body = None
        if has_gt:
            idx_of = self._label_to_idx
            true_arr = np.array([idx_of.get(str(v), -1) for v in self._view.y_true.tolist()])
            pred_arr = np.array([idx_of.get(str(v), -1) for v in self._view.y_pred.tolist()])
            k = len(label_names)
            cm = np.zeros((k, k), dtype=int)
            for t, p in zip(true_arr, pred_arr, strict=True):
                if t >= 0 and p >= 0:
                    cm[t, p] += 1
            self._cm = cm
            self._cm_true_idx = true_arr
            self._cm_pred_idx = pred_arr

            error_analysis_body = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H6(
                                    "Confusion Matrix — click a cell to inspect that error group",
                                    className="fw-bold mb-0",
                                ),
                                width="auto",
                                className="align-self-center",
                            ),
                            dbc.Col(
                                dcc.RadioItems(
                                    id="cm-normalize",
                                    options=[
                                        {"label": " Counts", "value": "count"},
                                        {"label": " Recall", "value": "recall"},
                                    ],
                                    value="count",
                                    inline=True,
                                    inputStyle={"marginRight": "4px"},
                                    labelStyle={"marginRight": "12px"},
                                ),
                                width="auto",
                                className="align-self-center",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "× clear cell",
                                    id="error-cell-clear-btn",
                                    n_clicks=0,
                                    color="link",
                                    size="sm",
                                    className="text-muted p-0",
                                    style={"fontSize": "0.8em"},
                                ),
                                width="auto",
                                className="align-self-center ms-auto",
                            ),
                        ],
                        className="align-items-center mb-2",
                        style={"flex": "0 0 auto"},
                    ),
                    dcc.Graph(
                        id="confusion-matrix-graph",
                        figure=plot_confusion_matrix(cm, label_names, title="", width=None, height=None),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "360px", "flex": "0 0 auto"},
                    ),
                    html.Small(
                        id="error-cell-caption",
                        children="Click a cell to see the words behind those errors.",
                        className="text-muted d-block my-2",
                        style={"flex": "0 0 auto"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    id="error-pred-importance",
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "320px"},
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    id="error-true-importance",
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "320px"},
                                ),
                                width=6,
                            ),
                        ],
                        className="g-2",
                        style={"flex": "0 0 auto"},
                    ),
                ],
                style={"height": "100%", "display": "flex", "flexDirection": "column", "overflowY": "auto"},
            )

        # ── Global Word Importance panel — controls live inside it ──
        # (Top-K / Sign / Exclude / Class only affect this panel, so they are co-located here —
        # independent from the local class picker in the Sentence Highlight panel below.)
        # The tab label already names the panel, so no H6; a tight controls row + a graph that
        # flex-grows to fill the panel means the whole chart is visible without scrolling.
        word_importance_panel = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Class", className="fw-bold small mb-0"),
                                dcc.Dropdown(
                                    id="global-class-selector",
                                    options=[{"label": name, "value": i} for i, name in enumerate(label_names)],
                                    value=0,
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Top-K words", className="fw-bold small mb-0"),
                                dcc.Slider(
                                    id="topk-slider",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=20,
                                    marks={1: "1", 10: "10", 20: "20", 30: "30", 50: "50"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                            width=5,
                        ),
                        dbc.Col(
                            [
                                html.Label("Contributions", className="fw-bold small mb-0"),
                                dcc.RadioItems(
                                    id="sign-filter",
                                    options=[
                                        {"label": " All", "value": "all"},
                                        {"label": " Positive", "value": "positive"},
                                        {"label": " Negative", "value": "negative"},
                                    ],
                                    value="all",
                                    inline=True,
                                    inputStyle={"marginRight": "4px"},
                                    labelStyle={"marginRight": "12px"},
                                ),
                            ],
                            width="auto",
                            className="align-self-start",
                        ),
                        dbc.Col(
                            [
                                html.Label("Exclude words", className="fw-bold small mb-0"),
                                dcc.Dropdown(
                                    id="word-filter",
                                    options=[{"label": w, "value": w} for w in all_words],
                                    value=[],
                                    multi=True,
                                    placeholder="Exclude…",
                                    style={"fontSize": "0.85em"},
                                ),
                            ],
                            width=True,
                        ),
                    ],
                    className="align-items-start g-2 mb-1",
                    style={"flex": "0 0 auto"},
                ),
                dcc.Graph(
                    id="global-importance-graph",
                    config={"displayModeBar": False, "responsive": True},
                    style={"flex": "1 1 auto", "minHeight": "0"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

        # ── Dataset table body (left-panel default tab) ───────────────
        text_samples_body = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H6(
                                "Text Samples — click a row to inspect",
                                className="fw-bold mb-0",
                                id="table-title",
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                    ],
                    className="align-items-center mb-2",
                    style={"flex": "0 0 auto"},
                ),
                # Grid grows to fill the tab body (now that the editor lives on its own tab).
                dag.AgGrid(
                    id="dataset-table",
                    rowData=self._full_table_records,
                    columnDefs=column_defs,
                    defaultColDef={"resizable": True, "sortable": True},
                    dashGridOptions={
                        # AG Grid v32.2+ (dash-ag-grid ≥31) replaced the string form
                        # ("single"/"multiple") with an object. In the new API a row click no
                        # longer selects by default — you must opt in via enableClickSelection,
                        # otherwise `selectedRows` never populates and every click 204s.
                        "rowSelection": {
                            "mode": "singleRow",
                            "checkboxes": False,
                            "enableClickSelection": True,
                        },
                        "tooltipShowDelay": 300,
                        "rowHeight": 38,
                    },
                    selectedRows=[self._full_table_records[0]],
                    style={
                        "flex": "1 1 auto",
                        "minHeight": "0",
                        "--ag-font-size": "15px",
                        "--ag-header-font-size": "14px",
                    },
                    className="ag-theme-alpine",
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

        # ── Selection bar: persistent, always-visible filter state + clears ──
        # Lives above the left tabs so a scatter/word selection stays visible whichever
        # left tab (table or embeddings) is active. Clear buttons are consolidated here
        # from the individual panels they used to live in.
        selection_children: list = [
            html.Span("Showing:", className="text-muted small me-2"),
            html.Span("all samples", id="selection-summary", className="small fw-bold me-3"),
        ]
        if scatter_col_content is not None:
            selection_children.append(
                dbc.Button(
                    "× clear selection",
                    id="scatter-clear-btn",
                    n_clicks=0,
                    color="link",
                    size="sm",
                    className="text-muted p-0 me-3",
                    style={"display": "none", "fontSize": "0.8em"},
                )
            )
        selection_children.append(
            dbc.Button(
                "× clear word filter",
                id="word-filter-clear-btn",
                n_clicks=0,
                color="link",
                size="sm",
                className="text-muted p-0",
                style={"display": "none", "fontSize": "0.8em"},
            )
        )
        # Right-aligned (ms-auto) and here — rather than inside the Dataset tab body — so it stays
        # usable from the Embeddings tab too, where it now also drives point highlighting.
        selection_children.append(
            dbc.Switch(
                id="errors-only-switch",
                label="Model Errors",
                value=False,
                className="small mb-0 ms-auto",
                style={} if has_gt else {"display": "none"},
            )
        )
        selection_bar = html.Div(
            selection_children,
            style={
                "border": "1px solid #dee2e6",
                "borderRadius": "4px",
                "padding": "6px 12px",
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "flex": "0 0 auto",
            },
        )

        # ── Assemble the three panels as tab groups (all bodies stay mounted) ──
        self._tab_groups = {}

        # Local class picker default: the predicted class of the initially selected row. It is reset
        # to the newly-selected text's prediction by a sync callback owned by
        # SentenceHighlightComponent — see sync_local_class_to_prediction — so switching sentences
        # always starts on "why did the model predict this", while still letting the user override it
        # for the current sentence.
        default_local_class = self._label_to_idx.get(self._full_table_records[0].get("prediction"), 0)
        self._components = [
            comp
            for comp in (
                SentenceHighlightComponent(default_local_class),
                WaterfallComponent(),
                DataEditorComponent(),
                CounterfactualComponent(),
                SimilarExamplesComponent(),
            )
            if type(comp).is_available(self._view, self._engine)
        ]
        highlight_comp = next(c for c in self._components if isinstance(c, SentenceHighlightComponent))
        waterfall_comp = next(c for c in self._components if isinstance(c, WaterfallComponent))
        editor_comp = next((c for c in self._components if isinstance(c, DataEditorComponent)), None)
        cf_comp = next((c for c in self._components if isinstance(c, CounterfactualComponent)), None)
        similar_comp = next((c for c in self._components if isinstance(c, SimilarExamplesComponent)), None)

        left_tabs: list = [("table", "Dataset", text_samples_body)]
        if scatter_col_content is not None:
            left_tabs.append(("scatter", "Embeddings", scatter_col_content))
        if editor_comp is not None:
            left_tabs.append(("editor", "Data Editor", editor_comp.layout(self._view, self._engine)))

        # Error Analysis sits beside Word Importance: it *is* an aggregated word-importance view
        # (per confusion-matrix cell), so it belongs with the other global "why" panels on the right.
        upper_right_tabs: list = [("importance", "Word Importance", word_importance_panel)]
        if error_analysis_body is not None:
            upper_right_tabs.append(("errors", "Error Analysis", error_analysis_body))
        if cf_comp is not None:
            upper_right_tabs.append(("counterfactual", "Counterfactuals", cf_comp.layout(self._view, self._engine)))
        if similar_comp is not None:
            upper_right_tabs.append(("similar", "Similar Examples", similar_comp.layout(self._view, self._engine)))

        lower_right_tabs: list = [
            ("highlight", "Sentence", highlight_comp.layout(self._view, self._engine)),
            ("waterfall", "Waterfall", waterfall_comp.layout(self._view, self._engine)),
        ]

        left_column = html.Div(
            [selection_bar, self._tabbed_card("left-tabs", left_tabs)],
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "height": "100%"},
        )
        right_column = html.Div(
            [
                self._tabbed_card("upper-right-tabs", upper_right_tabs),
                self._tabbed_card("lower-right-tabs", lower_right_tabs),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "height": "100%"},
        )

        stores: list = [
            # current-datapoint is the app's primary selection: the one text every
            # per-instance panel (highlight, waterfall, counterfactuals) reads from.
            dcc.Store(id=_CURRENT_STORE, data=None),
            dcc.Store(id="scatter-selected-indices", data=None),
            dcc.Store(id="word-click-filter", data=None),
            # Selected confusion-matrix cell: {"pred": idx, "true": idx, "indices": [...]} or None.
            dcc.Store(id="error-cell", data=None),
        ]
        # Only the What-if Lab (editor + counterfactual) reads/writes the apply store; the always-on
        # core panels (highlight, waterfall) never do, so their presence alone shouldn't create it.
        if editor_comp is not None or cf_comp is not None:
            stores.append(dcc.Store(id=_APPLY_STORE, data=None))

        self.app.layout = dbc.Container(
            [
                # ── Header — the Class selector now lives with the panel it drives: Word
                # Importance (global) and Sentence Highlight (local, see highlight_body) each
                # get their own, so switching one no longer silently reinterprets the other. ──
                dbc.Row(
                    [
                        dbc.Col(html.H4("Shapash — NLP Explainer", className="mb-0"), width=12),
                    ],
                    className="py-2 align-items-center",
                    style={"flex": "0 0 auto"},
                ),
                # ── Three-panel body: left (data) | right (global over local) ──
                dbc.Row(
                    [
                        dbc.Col(left_column, width=5, style={"height": "100%"}),
                        dbc.Col(right_column, width=7, style={"height": "100%"}),
                    ],
                    # gx-3 = horizontal gutter only. A vertical gutter (g-3) adds a 1rem net excess
                    # that pushes the 100vh shell past the viewport and makes the whole page scroll.
                    className="gx-3",
                    style={"flex": "1 1 auto", "minHeight": "0"},
                ),
                *stores,
            ],
            fluid=True,
            # Full-viewport shell: panels scroll internally, the page itself never scrolls.
            style={"height": "100vh", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
        )

    def _tabbed_card(self, tabs_id: str, tabs: list) -> html.Div:
        """Wrap ``(tab_id, label, body)`` tuples into a card with a tab header.

        Every body stays mounted in the DOM; a registered callback toggles ``display`` so
        cross-panel callbacks that target a component on an inactive tab keep working (Dash
        errors if an Output/Input component is absent). The tab ids are recorded on
        ``self._tab_groups`` for that toggle callback.
        """
        active = tabs[0][0]
        self._tab_groups[tabs_id] = [tid for tid, _, _ in tabs]
        headers = [dbc.Tab(label=label, tab_id=tid) for tid, label, _ in tabs]
        bodies = [
            html.Div(body, id=f"{tabs_id}-body-{tid}", style=(_VISIBLE if tid == active else _HIDDEN))
            for tid, _, body in tabs
        ]
        return html.Div(
            [
                dbc.Tabs(headers, id=tabs_id, active_tab=active),
                # Flex column so the single visible body (the others are display:none) fills the height.
                html.Div(
                    bodies,
                    style={
                        "flex": "1 1 auto",
                        "minHeight": "0",
                        "paddingTop": "8px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
            ],
            style={
                "border": "1px solid #dee2e6",
                "borderRadius": "4px",
                "padding": "12px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
                "flex": "1 1 0",
                "minHeight": "0",
            },
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        contrib = self._view.contributions
        full_records = self._full_table_records
        has_gt = self._has_gt

        def _compose_indices(selected_indices, error_cell, errors_only=False):
            """Combine the scatter selection, the confusion-cell selection, and the errors toggle.

            All three are optional filters that intersect: the embedding selection *within* the
            chosen confusion cell, further restricted to misclassified samples when ``errors_only``
            is set. Returns a list of original sample indices, or ``None`` when nothing is active.
            """
            cell_indices = error_cell.get("indices") if error_cell else None
            error_positions: set[int] | None = None
            if errors_only and has_gt:
                mask = self._error_mask()
                if mask is not None:
                    error_positions = set(np.where(mask)[0].tolist())
            return _compose_selection(selected_indices, cell_indices, error_positions)

        # ── Global word importance ───────────────────────────────────────
        @self.app.callback(
            Output("global-importance-graph", "figure"),
            [
                Input("global-class-selector", "value"),
                Input("topk-slider", "value"),
                Input("sign-filter", "value"),
                Input("word-filter", "value"),
                Input("scatter-selected-indices", "data"),
                Input("error-cell", "data"),
                Input("errors-only-switch", "value"),
            ],
        )
        def update_global_importance(
            label_idx, topk, sign_filter, exclude_words_list, selected_indices, error_cell, errors_only
        ):
            if label_idx is None:
                raise PreventUpdate
            effective_indices = _compose_indices(selected_indices, error_cell, bool(errors_only))
            word_imp = contrib.word_importance(
                label_idx=int(label_idx),
                n_top=int(topk or 20),
                filter_sign=sign_filter or "all",
                exclude_words=set(exclude_words_list or []) or None,
                sample_indices=effective_indices,
            )
            # No class name here — the Class dropdown right above the chart already shows it.
            suffix = f" ({len(effective_indices)} samples)" if effective_indices is not None else ""
            fig = plot_word_importance(
                word_imp,
                title=f"Word importance{suffix}",
                width=None,
                height=None,
            )
            fig.layout.height = None  # let the CSS container height take over
            return fig

        # ── Primary selection: selected table row → current-datapoint ────
        # The editor's Predict callback also writes this store (see DataEditorComponent),
        # so every per-instance panel reads one source regardless of where it came from.
        @self.app.callback(
            Output(_CURRENT_STORE, "data"),
            Input("dataset-table", "selectedRows"),
        )
        def set_current_from_row(selected_rows):
            if not selected_rows:
                raise PreventUpdate
            pos = int(selected_rows[0]["_orig_idx"])
            base_values = contrib.base_values
            base = base_values[pos] if base_values is not None else None
            return pack_datapoint(
                text=selected_rows[0].get("text", ""),
                orig_idx=pos,
                tokens=contrib.token_strings[pos],
                values=contrib.values[pos],
                base_values=base,
                label=selected_rows[0].get("prediction"),
            )

        # Local class picker sync + sentence highlight + waterfall are registered by
        # SentenceHighlightComponent / WaterfallComponent (see the component loop at the end of this
        # method) — they only need the current-datapoint store, already shared via `stores["current"]`.

        # ── Word-bar click / clear → table word filter ───────────────────
        # Resetting the graph's clickData to None after each event lets the SAME bar be clicked
        # again (Plotly does not re-fire clickData when the value is unchanged). Without a scatter
        # the bar click drives word-click-filter directly; WITH a scatter the filter is derived
        # from scatter-word-select instead (see the scatter block), so editing/removing the word
        # there also clears this filter.
        if self._scatter_xy is None:

            @self.app.callback(
                Output("word-click-filter", "data"),
                Output("global-importance-graph", "clickData"),
                Input("global-importance-graph", "clickData"),
                Input("word-filter-clear-btn", "n_clicks"),
                prevent_initial_call=True,
            )
            def update_word_click_filter(click_data, _clear_clicks):
                trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
                if "word-filter-clear-btn" in trigger:
                    return None, None
                if not click_data or not click_data.get("points"):
                    raise PreventUpdate
                return [click_data["points"][0]["y"]], None

        @self.app.callback(
            Output("word-filter-clear-btn", "style"),
            Input("word-click-filter", "data"),
        )
        def toggle_word_clear_button(word_filter):
            if word_filter:
                return {"display": "inline", "fontSize": "0.8em"}
            return {"display": "none", "fontSize": "0.8em"}

        # ── Unified table filter (scatter + word-bar click + errors-only) ──
        # Also writes the selection-summary readout: it already knows the resulting row count, so the
        # count and the filter description stay consistent (no separate, drift-prone callback).
        @self.app.callback(
            [
                Output("dataset-table", "rowData"),
                Output("dataset-table", "selectedRows"),
                Output("table-title", "children"),
                Output("selection-summary", "children"),
            ],
            [
                Input("scatter-selected-indices", "data"),
                Input("word-click-filter", "data"),
                Input("errors-only-switch", "value"),
                Input("error-cell", "data"),
            ],
        )
        def filter_table(selected_indices, word_filter, errors_only, error_cell):
            # word_filter may be a list of words (multi-select in the scatter) or None.
            words = word_filter if isinstance(word_filter, list) else ([word_filter] if word_filter else [])
            effective_indices = _compose_indices(selected_indices, error_cell)
            if effective_indices is None:
                recs = full_records
            else:
                idx_set = set(effective_indices)
                recs = [r for r in full_records if r["_orig_idx"] in idx_set] or full_records
            if errors_only and has_gt:
                misclassified = [r for r in recs if str(r.get("prediction", "")) != str(r.get("ground_truth", ""))]
                recs = misclassified or recs
            if words:
                lowers = [w.lower() for w in words]
                # Rows containing ANY of the selected words (matches the scatter's word colouring).
                filtered = [r for r in recs if any(w in r["text"].lower() for w in lowers)]
                recs = filtered or recs

            total = len(full_records)
            parts = []
            if error_cell:
                names = contrib.label_names or []
                pred_name = names[error_cell["pred"]] if error_cell["pred"] < len(names) else str(error_cell["pred"])
                true_name = names[error_cell["true"]] if error_cell["true"] < len(names) else str(error_cell["true"])
                parts.append(f"predicted {pred_name} · true {true_name}")
            if selected_indices is not None:
                parts.append(f"{len(selected_indices)} selected in embeddings")
            if words:
                parts.append("containing " + ", ".join(f'"{w}"' for w in words))
            if errors_only and has_gt:
                parts.append("model errors only")
            if parts:
                summary = " · ".join(parts) + f" ({len(recs)} of {total})"
                title = "Text Samples — filtered"
            else:
                summary = f"all {total} samples"
                title = "Text Samples — click a row to inspect"
            return recs, [recs[0]], title, summary

        # ── Scatter-specific callbacks (registered only when xy provided) ──
        if self._scatter_xy is not None:

            @self.app.callback(
                Output("scatter-plot", "figure"),
                [
                    Input("color-by", "value"),
                    Input("scatter-word-select", "value"),
                    Input("global-class-selector", "value"),
                    Input("errors-only-switch", "value"),
                ],
            )
            def update_scatter_color(color_by, words, label_idx, errors_only):
                return self._build_scatter_fig(
                    color_by or "prediction",
                    words=words or [],
                    label_idx=int(label_idx or 0),
                    errors_only=bool(errors_only),
                )

            @self.app.callback(
                Output("color-by", "value"),
                Input("scatter-word-select", "value"),
            )
            def sync_color_by(words):
                if not words:
                    raise PreventUpdate
                return "word_contribution"

            # Bar click / clear → scatter word selection (the single source of truth for the word
            # filter when a scatter exists). clickData is reset so the same bar can be re-clicked.
            @self.app.callback(
                Output("scatter-word-select", "value"),
                Output("global-importance-graph", "clickData"),
                Input("global-importance-graph", "clickData"),
                Input("word-filter-clear-btn", "n_clicks"),
                prevent_initial_call=True,
            )
            def set_scatter_words(click_data, _clear_clicks):
                trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
                if "word-filter-clear-btn" in trigger:
                    return [], None
                if not click_data or not click_data.get("points"):
                    raise PreventUpdate
                return [click_data["points"][0]["y"]], None

            # Table word filter follows the scatter selection: removing the word in the dropdown (or
            # hitting clear) unfilters the table and hides the clear button.
            @self.app.callback(
                Output("word-click-filter", "data"),
                Input("scatter-word-select", "value"),
            )
            def word_filter_from_scatter(words):
                return words or None

            @self.app.callback(
                Output("scatter-word-select", "style"),
                Input("color-by", "value"),
            )
            def toggle_word_select(color_by):
                base = {"minWidth": "180px", "fontSize": "0.9em"}
                return base if color_by == "word_contribution" else {**base, "display": "none"}

            @self.app.callback(
                Output("scatter-selected-indices", "data"),
                Input("scatter-plot", "selectedData"),
                Input("scatter-plot", "clickData"),
                Input("scatter-clear-btn", "n_clicks"),
            )
            def update_scatter_selection(selected_data, click_data, _clear_clicks):
                trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
                if "scatter-clear-btn" in trigger:
                    return None
                if "clickData" in trigger:
                    if not click_data or not click_data.get("points"):
                        return None
                    return [int(click_data["points"][0]["customdata"][0])]
                # An empty selectedData here is almost always plotly re-emitting on a figure recolor
                # (color-by / word / errors-only toggle), NOT a user deselect — ignore it so the box
                # survives. Genuine clears go through the clear button or a point click above.
                if not selected_data or not selected_data.get("points"):
                    raise PreventUpdate
                return [int(pt["customdata"][0]) for pt in selected_data["points"]]

            @self.app.callback(
                Output("scatter-clear-btn", "style"),
                Input("scatter-selected-indices", "data"),
            )
            def toggle_clear_button(selected_indices):
                visible = {"display": "inline", "fontSize": "0.8em"}
                hidden = {"display": "none", "fontSize": "0.8em"}
                return visible if selected_indices else hidden

        # ── Error Analysis callbacks (registered only with ground truth) ──
        if has_gt:
            names = contrib.label_names or [str(i) for i in range(self._cm.shape[0])]
            name_to_idx = {name: i for i, name in enumerate(names)}
            cm = self._cm
            pred_idx_arr = self._cm_pred_idx
            true_idx_arr = self._cm_true_idx

            @self.app.callback(
                Output("confusion-matrix-graph", "figure"),
                Input("cm-normalize", "value"),
            )
            def update_confusion_matrix(normalize):
                fig = plot_confusion_matrix(
                    cm,
                    names,
                    normalize="true" if normalize == "recall" else None,
                    title="",  # the tab header already labels this panel
                )
                # Let the container height drive size (this panel is only half-column tall).
                fig.layout.width = None
                fig.layout.height = None
                return fig

            # Cell click / clear → the shared error-cell selection. A single owner (dispatching on the
            # trigger) keeps this the only writer, so no allow_duplicate coordination is needed.
            @self.app.callback(
                Output("error-cell", "data"),
                Output("confusion-matrix-graph", "clickData"),
                Input("confusion-matrix-graph", "clickData"),
                Input("error-cell-clear-btn", "n_clicks"),
                prevent_initial_call=True,
            )
            def set_error_cell(click_data, _clear_clicks):
                trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
                if "error-cell-clear-btn" in trigger:
                    return None, None
                cell = _cell_from_click(click_data, name_to_idx)
                if cell is None:
                    raise PreventUpdate
                pred_i, true_i = cell
                indices = np.where((pred_idx_arr == pred_i) & (true_idx_arr == true_i))[0].tolist()
                # Reset clickData so re-clicking the *same* cell registers as a change and re-fires.
                return {"pred": pred_i, "true": true_i, "indices": indices}, None

            # Per-cell word importance: words driving the (wrong) predicted class vs. the true class.
            @self.app.callback(
                Output("error-pred-importance", "figure"),
                Output("error-true-importance", "figure"),
                Output("error-cell-caption", "children"),
                Input("error-cell", "data"),
            )
            def update_error_word_charts(error_cell):
                if not error_cell:
                    empty = _empty_word_fig("Click a confusion-matrix cell to see the words behind those errors.")
                    return empty, empty, "Click a cell to see the words behind those errors."
                pred_i, true_i, indices = error_cell["pred"], error_cell["true"], error_cell["indices"]
                pred_name, true_name = names[pred_i], names[true_i]
                if not indices:
                    empty = _empty_word_fig("No samples for this (predicted, true) pair.")
                    return empty, empty, f"predicted {pred_name} · true {true_name}: 0 samples"
                wi_pred = contrib.word_importance(label_idx=pred_i, n_top=15, sample_indices=indices)
                wi_true = contrib.word_importance(label_idx=true_i, n_top=15, sample_indices=indices)
                fig_pred = plot_word_importance(
                    wi_pred, title=f"Words toward predicted: {pred_name}", width=None, height=None
                )
                fig_true = plot_word_importance(
                    wi_true, title=f"Words toward true: {true_name}", width=None, height=None
                )
                fig_pred.layout.height = None
                fig_true.layout.height = None
                caption = f"predicted {pred_name} · true {true_name}: {len(indices)} sample(s)"
                if pred_i == true_i:
                    caption += " — correct predictions (diagonal)"
                elif len(indices) < 5:
                    caption += " — few samples; interpret with caution"
                return fig_pred, fig_true, caption

        # ── Tab visibility: toggle display of always-mounted bodies ──────
        # Bodies stay in the DOM (see _tabbed_card); only their `display` flips so the
        # cross-panel callbacks above keep firing for panels on inactive tabs.
        for tabs_id, tab_ids in self._tab_groups.items():

            @self.app.callback(
                [Output(f"{tabs_id}-body-{tid}", "style") for tid in tab_ids],
                Input(tabs_id, "active_tab"),
            )
            def _toggle_tab_bodies(active, _tab_ids=tab_ids):
                return [(_VISIBLE if tid == active else _HIDDEN) for tid in _tab_ids]

        # (The selection-summary readout is written by filter_table, which knows the row count.)

        # ── Registered components (always-on core panels + capability-gated What-if Lab) ──
        stores = {"apply": _APPLY_STORE, "current": _CURRENT_STORE}
        for comp in self._components:
            comp.register_callbacks(self.app, self._view, self._engine, stores)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, port: int = 8050, debug: bool = False, host: str = "127.0.0.1") -> None:
        """Launch the Dash development server."""
        self.app.run(port=port, debug=debug, host=host)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _word_contributions(self, word: str, label_idx: int) -> np.ndarray:
        """Per-sample sum of SHAP contributions for all tokens matching *word*."""
        contrib = self._view.contributions
        n = self._view.n_samples
        result = np.zeros(n)
        word_lower = word.lower()
        for i in range(n):
            tokens = contrib.token_strings[i]
            vals = contrib.values[i]
            if vals.ndim == 2:
                vals = vals[:, label_idx]
            for j, tok in enumerate(tokens):
                if tok.strip().lower() == word_lower:
                    result[i] += vals[j]
        return result

    def _error_mask(self) -> np.ndarray | None:
        """Boolean array, ``True`` where the prediction disagrees with the ground truth.

        ``None`` when either is unavailable. Matches the string comparison used by the
        "Model Errors" table filter so both stay consistent.
        """
        y_true = self._view.y_true
        y_pred = self._view.y_pred
        if y_true is None or y_pred is None:
            return None
        return np.asarray(y_true).astype(str) != np.asarray(y_pred).astype(str)

    @staticmethod
    def _emphasize_errors(
        idx_arr: np.ndarray,
        base_opacity: float,
        base_size: float,
        error_mask: np.ndarray | None,
    ) -> tuple[list[float] | float, list[float] | float]:
        """Per-point opacity/size that pops model errors and shadows everything else.

        Keeps the caller's coloring untouched — only opacity and marker size change —
        so points stay grouped/colored however the "Color by" dropdown already draws them.
        Returns scalars (the unmodified base values) when ``error_mask`` is unavailable.
        """
        if error_mask is None or len(idx_arr) == 0:
            return base_opacity, base_size
        is_error = error_mask[idx_arr]
        opacity = np.where(is_error, max(base_opacity, 0.9), 0.12).tolist()
        size = np.where(is_error, base_size + 3, max(base_size - 2, 3)).tolist()
        return opacity, size

    def _build_scatter_fig(
        self,
        color_by: str,
        words: list[str] | None = None,
        label_idx: int = 0,
        errors_only: bool = False,
    ) -> go.Figure:
        """2-D scatter coloured by prediction, ground-truth label, or word SHAP contribution.

        One trace per class for label-based coloring so the legend works correctly
        and Plotly's box/lasso select dims unselected points across all traces.
        Word-contribution mode uses a single diverging-colorscale trace.
        ``customdata`` always stores the original sample index. When ``errors_only`` is
        set, misclassified points are emphasized (larger, opaque) and the rest are
        shadowed (small, faint) without altering their color.
        """
        n = self._view.n_samples
        contrib = self._view.contributions
        texts_short = [(t[:120] + "…") if len(t) > 120 else t for t in self._view.texts]
        xy = self._scatter_xy
        error_mask = self._error_mask() if errors_only else None

        if color_by == "word_contribution" and words:
            contributions = sum(self._word_contributions(w, label_idx) for w in words)
            max_abs = float(np.abs(contributions).max()) or 1.0
            present_mask = np.where(contributions != 0.0)[0]
            absent_mask = np.where(contributions == 0.0)[0]
            colorbar_title = " + ".join(f'"{w}"' for w in words) if len(words) <= 3 else f"{len(words)} words"

            fig = go.Figure()
            # Both layers are ALWAYS added (even when a mask is empty) so the trace structure stays
            # constant across word additions/removals. With a stable `uirevision`, a changing WebGL
            # (Scattergl) trace count leaves ghost/"shadow" points from the previous render — keeping
            # exactly two traces avoids that.
            absent_opacity, absent_size = self._emphasize_errors(absent_mask, 0.35, 5, error_mask)
            present_opacity, present_size = self._emphasize_errors(present_mask, 0.9, 9, error_mask)

            # Gray context layer — absent points (no selected word contributes to them).
            fig.add_trace(
                go.Scattergl(
                    x=xy[absent_mask, 0].tolist(),
                    y=xy[absent_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(color="#b0b0b0", size=absent_size, opacity=absent_opacity),
                    customdata=absent_mask.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in absent_mask],
                    hovertemplate="%{text}<extra>absent</extra>",
                    showlegend=False,
                )
            )
            # Colored overlay — samples where at least one selected word contributes.
            fig.add_trace(
                go.Scattergl(
                    x=xy[present_mask, 0].tolist(),
                    y=xy[present_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(
                        color=contributions[present_mask].tolist(),
                        colorscale="RdBu",
                        cmin=-max_abs,
                        cmax=max_abs,
                        size=present_size,
                        opacity=present_opacity,
                        colorbar=dict(
                            title=dict(text=colorbar_title, side="right"),
                            thickness=12,
                            tickformat=".2f",
                        ),
                    ),
                    customdata=present_mask.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in present_mask],
                    hovertemplate="<b>SHAP: %{marker.color:.3f}</b><br>%{text}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.update_layout(
                dragmode="select",
                uirevision="scatter",
                xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
                yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
                plot_bgcolor="#f9f9f9",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                autosize=True,
                showlegend=False,
            )
            return fig

        if color_by == "ground_truth" and self._view.y_true is not None:
            labels = [str(label) for label in self._view.y_true.tolist()]
        elif self._view.y_pred is not None:
            labels = [str(label) for label in self._view.y_pred.tolist()]
        else:
            labels = [""] * n

        label_names = contrib.label_names or sorted(set(labels))

        fig = go.Figure()
        for i, name in enumerate(label_names):
            mask = [j for j, lbl in enumerate(labels) if lbl == name]
            if not mask:
                continue
            mask_arr = np.array(mask)
            opacity, size = self._emphasize_errors(mask_arr, 0.75, 7, error_mask)
            fig.add_trace(
                go.Scattergl(
                    x=xy[mask_arr, 0].tolist(),
                    y=xy[mask_arr, 1].tolist(),
                    mode="markers",
                    marker=dict(color=_PALETTE[i % len(_PALETTE)], size=size, opacity=opacity),
                    customdata=mask_arr.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in mask],
                    hovertemplate=f"<b>{name}</b><br>%{{text}}<extra></extra>",
                    name=name,
                )
            )

        fig.update_layout(
            dragmode="select",
            uirevision="scatter",
            xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
            yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            # No fixed height — the responsive Graph stretches it to fill the card.
            autosize=True,
            legend=dict(itemsizing="constant", orientation="v", title_text=""),
            showlegend=True,
        )
        return fig
