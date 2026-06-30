"""Minimal Dash webapp for NLP text classification explanations.

Prototype bridge toward Phase 5b (composable WebappComponents). The three
panels here (global word importance, dataset table, local token contributions)
map directly onto the three ``WebappComponent`` instances that Phase 5b would
register for a text modality. All tabular-only SmartApp panels (violin, cluster,
scatter prediction picking, confusion matrix) are absent rather than disabled.
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

from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.plots.plot_waterfall import plot_waterfall
from shapash.plots.plot_word_importance import plot_word_importance

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}
# min() only shrinks on small screens; returns the natural size on large ones.
# 800 px CSS (HiDPI laptop): 224 px chart / 240 px table — enough to show token contributions
# 1080 px CSS (external): 302 px chart / 324 px table — close to original feel
_WORD_IMPORTANCE_HEIGHT_CSS = "min(390px, 28vh)"
_TABLE_HEIGHT_CSS = "min(460px, 30vh)"
_SPECIAL_RE = re.compile(r"^\[.*\]$|^##|^\s*$")
_HIDDEN = {"display": "none"}
_VISIBLE = {"display": "block"}
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
        self.explainer = explainer
        if scatter_xy is not None:
            scatter_xy = np.asarray(scatter_xy)
            n = len(explainer.texts)
            if scatter_xy.shape != (n, 2):
                raise ValueError(f"scatter_xy must have shape ({n}, 2), got {scatter_xy.shape}")
        self._scatter_xy: np.ndarray | None = scatter_xy
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Shapash — NLP Explainer"
        self._build_layout()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        contrib = self.explainer.contributions
        label_names = contrib.label_names or [str(i) for i in range(self._n_classes())]
        n = len(self.explainer.texts)

        # ── Table records — always include _orig_idx for scatter filtering ──
        records: dict[str, list] = {
            "_orig_idx": list(range(n)),
            "text": self.explainer.texts.tolist(),
            "prediction": (self.explainer.y_pred.tolist() if self.explainer.y_pred is not None else [""] * n),
        }
        if getattr(self.explainer, "y_true", None) is not None:
            records["ground_truth"] = self.explainer.y_true.tolist()

        # Single "Probability" column — confidence of the predicted class.
        # max(axis=1) works for both binary and multiclass since the predicted
        # label is always the argmax, regardless of how columns are named.
        y_prob: pd.DataFrame | None = getattr(self.explainer, "y_prob", None)
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
            if getattr(self.explainer, "y_true", None) is not None:
                color_options.append({"label": "Ground Truth", "value": "ground_truth"})

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
                                dbc.Button(
                                    "× clear",
                                    id="scatter-clear-btn",
                                    n_clicks=0,
                                    color="link",
                                    size="sm",
                                    className="text-muted p-0",
                                    style={"display": "none", "fontSize": "0.8em"},
                                ),
                                width="auto",
                                className="align-self-center",
                            ),
                        ],
                        className="align-items-center mb-2",
                    ),
                    html.Small(
                        "Click a point or box/lasso to filter — click again or × clear to reset.",
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
                # Flex column so the graph above can stretch to the card height.
                style={**_CARD_STYLE, "display": "flex", "flexDirection": "column"},
            )

        # ── Global Word Importance panel — controls now live inside it ──
        # (Top-K / Sign / Exclude only affect this panel, so they are
        # co-located here rather than in a misleading full-width top bar.)
        word_importance_panel = html.Div(
            [
                html.H6("Global Word Importance", className="fw-bold mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Top-K words", className="fw-bold small mb-1"),
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
                            width=7,
                        ),
                        dbc.Col(
                            [
                                html.Label("Contributions", className="fw-bold small mb-1"),
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
                            width=5,
                        ),
                    ],
                    className="align-items-center mb-2",
                ),
                html.Div(
                    [
                        html.Label("Exclude words", className="fw-bold small mb-1"),
                        dcc.Dropdown(
                            id="word-filter",
                            options=[{"label": w, "value": w} for w in all_words],
                            value=[],
                            multi=True,
                            placeholder="Select words to exclude…",
                            style={"fontSize": "0.9em"},
                        ),
                    ],
                    className="mb-2",
                ),
                dcc.Graph(
                    id="global-importance-graph",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": _WORD_IMPORTANCE_HEIGHT_CSS},
                ),
            ],
            style=_CARD_STYLE,
        )

        # ── Text Samples panel (full-width hub) ───────────────────────
        text_samples_panel = html.Div(
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
                        dbc.Col(
                            dbc.Button(
                                "× clear word filter",
                                id="word-filter-clear-btn",
                                n_clicks=0,
                                color="link",
                                size="sm",
                                className="text-muted p-0",
                                style={"display": "none", "fontSize": "0.8em"},
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                        dbc.Col(
                            dbc.Switch(
                                id="errors-only-switch",
                                label="Errors only",
                                value=False,
                                className="small mb-0",
                                style={} if has_gt else {"display": "none"},
                            ),
                            width="auto",
                            className="align-self-center ms-2",
                        ),
                    ],
                    className="align-items-center mb-2",
                ),
                dag.AgGrid(
                    id="dataset-table",
                    rowData=self._full_table_records,
                    columnDefs=column_defs,
                    defaultColDef={"resizable": True, "sortable": True},
                    dashGridOptions={
                        "rowSelection": "single",
                        "tooltipShowDelay": 300,
                        "rowHeight": 38,
                    },
                    selectedRows=[self._full_table_records[0]],
                    style={
                        "height": _TABLE_HEIGHT_CSS,
                        "--ag-font-size": "15px",
                        "--ag-header-font-size": "14px",
                    },
                    className="ag-theme-alpine",
                ),
            ],
            style=_CARD_STYLE,
        )

        # ── Token Contributions panel (full-width detail-on-demand) ───
        token_contributions_panel = html.Div(
            [
                html.H6(
                    id="sentence-highlight-title",
                    children="Token Contributions",
                    className="fw-bold",
                ),
                # Min-height + vertical centring so short samples still give the
                # panel presence instead of collapsing to a thin strip.
                html.Div(
                    id="sentence-highlight",
                    style={
                        "minHeight": "150px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center",
                    },
                ),
                html.Hr(style={"margin": "14px 0 10px"}),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.RadioItems(
                                id="show-waterfall",
                                options=[
                                    {"label": " Show waterfall", "value": "show"},
                                    {"label": " Hide", "value": "hide"},
                                ],
                                value="hide",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px"},
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                        dbc.Col(
                            html.Div(
                                id="waterfall-threshold-wrapper",
                                children=[
                                    html.Label(
                                        "Group tokens below (% of max contribution)",
                                        className="small fw-bold mb-0 me-2",
                                    ),
                                    dcc.Slider(
                                        id="waterfall-threshold",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=10,
                                        marks={0: "0%", 10: "10%", 25: "25%", 50: "50%"},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                    ),
                                ],
                                style=_HIDDEN,
                            ),
                            width=True,
                        ),
                    ],
                    className="align-items-center mb-2",
                ),
                html.Div(
                    id="waterfall-container",
                    children=[
                        dcc.Graph(
                            id="waterfall-graph",
                            config={"displayModeBar": False},
                        ),
                    ],
                    style=_HIDDEN,
                ),
            ],
            style=_CARD_STYLE,
        )

        # ── Top row: two global "where to look" panels side by side ───
        # Word importance shares the row with the sample-space scatter when a
        # projection is provided; otherwise it spans the full width.
        if scatter_col_content is not None:
            top_row_cols = [
                dbc.Col(word_importance_panel, width=6),
                dbc.Col(scatter_col_content, width=6),
            ]
        else:
            top_row_cols = [dbc.Col(word_importance_panel, width=12)]

        self.app.layout = dbc.Container(
            [
                # ── Header (Class selector is the only truly global control) ──
                dbc.Row(
                    [
                        dbc.Col(html.H3("Shapash — NLP Explainer", className="mb-0"), width=8),
                        dbc.Col(
                            [
                                html.Label("Class", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="class-selector",
                                    options=[{"label": name, "value": i} for i, name in enumerate(label_names)],
                                    value=0,
                                    clearable=False,
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3 mt-3 align-items-center",
                ),
                # ── Overview / controls row (filters the funnel below) ────────
                dbc.Row(top_row_cols, className="mb-3"),
                # ── Hub: full-width text samples table ───────────────────────
                dbc.Row(dbc.Col(text_samples_panel, width=12), className="mb-3"),
                # ── Detail-on-demand: full-width token contributions ─────────
                dbc.Row(dbc.Col(token_contributions_panel, width=12), className="mb-3"),
                # ── Hidden stores (always present) ───────────────────────────
                dcc.Store(id="scatter-selected-indices", data=None),
                dcc.Store(id="word-click-filter", data=None),
            ],
            fluid=True,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        contrib = self.explainer.contributions
        full_records = self._full_table_records
        has_gt = self._has_gt

        # ── Global word importance ───────────────────────────────────────
        @self.app.callback(
            Output("global-importance-graph", "figure"),
            [
                Input("class-selector", "value"),
                Input("topk-slider", "value"),
                Input("sign-filter", "value"),
                Input("word-filter", "value"),
                Input("scatter-selected-indices", "data"),
            ],
        )
        def update_global_importance(label_idx, topk, sign_filter, exclude_words_list, selected_indices):
            if label_idx is None:
                raise PreventUpdate
            word_imp = contrib.word_importance(
                label_idx=int(label_idx),
                n_top=int(topk or 20),
                filter_sign=sign_filter or "all",
                exclude_words=set(exclude_words_list or []) or None,
                sample_indices=selected_indices,
            )
            label_name = (contrib.label_names or [])[int(label_idx)] if contrib.label_names else str(label_idx)
            suffix = f" ({len(selected_indices)} samples)" if selected_indices is not None else ""
            fig = plot_word_importance(
                word_imp,
                title=f"Word importance — {label_name}{suffix}",
                width=None,
                height=None,
            )
            fig.layout.height = None  # let the CSS container height take over
            return fig

        # ── Sentence highlight ───────────────────────────────────────────
        @self.app.callback(
            [
                Output("sentence-highlight", "children"),
                Output("sentence-highlight-title", "children"),
            ],
            [
                Input("dataset-table", "selectedRows"),
                Input("class-selector", "value"),
            ],
        )
        def update_sentence_highlight(selected_rows, label_idx):
            if not selected_rows or label_idx is None:
                raise PreventUpdate
            pos = int(selected_rows[0]["_orig_idx"])
            label_idx = int(label_idx)
            tokens, vals, base_value, label_name = self._sample_data(pos, label_idx)
            highlight = plot_sentence_highlight(tokens=tokens, values=vals, base_value=base_value)
            title = f"Token Contributions — {label_name}"
            return highlight, title

        # ── Waterfall show/hide ──────────────────────────────────────────
        @self.app.callback(
            [
                Output("waterfall-container", "style"),
                Output("waterfall-threshold-wrapper", "style"),
            ],
            Input("show-waterfall", "value"),
        )
        def toggle_waterfall(show):
            return (_VISIBLE, _VISIBLE) if show == "show" else (_HIDDEN, _HIDDEN)

        # ── Waterfall chart ──────────────────────────────────────────────
        @self.app.callback(
            Output("waterfall-graph", "figure"),
            [
                Input("dataset-table", "selectedRows"),
                Input("class-selector", "value"),
                Input("waterfall-threshold", "value"),
            ],
        )
        def update_waterfall(selected_rows, label_idx, threshold_pct):
            if not selected_rows or label_idx is None:
                raise PreventUpdate
            pos = int(selected_rows[0]["_orig_idx"])
            label_idx = int(label_idx)
            tokens, vals, base_value, label_name = self._sample_data(pos, label_idx)
            min_pct = (threshold_pct if threshold_pct is not None else 10) / 100.0
            return plot_waterfall(
                tokens=tokens,
                values=vals,
                base_value=base_value,
                min_pct=min_pct,
                title=f"Token contributions — {label_name}",
            )

        # ── Word-bar click → filter table ────────────────────────────────
        @self.app.callback(
            Output("word-click-filter", "data"),
            Input("global-importance-graph", "clickData"),
            Input("word-filter-clear-btn", "n_clicks"),
        )
        def update_word_click_filter(click_data, _clear_clicks):
            trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
            if "word-filter-clear-btn" in trigger:
                return None
            if not click_data or not click_data.get("points"):
                raise PreventUpdate
            return click_data["points"][0]["y"]

        @self.app.callback(
            Output("word-filter-clear-btn", "style"),
            Input("word-click-filter", "data"),
        )
        def toggle_word_clear_button(word_filter):
            if word_filter:
                return {"display": "inline", "fontSize": "0.8em"}
            return {"display": "none", "fontSize": "0.8em"}

        # ── Unified table filter (scatter + word-bar click + errors-only) ──
        @self.app.callback(
            [
                Output("dataset-table", "rowData"),
                Output("dataset-table", "selectedRows"),
                Output("table-title", "children"),
            ],
            [
                Input("scatter-selected-indices", "data"),
                Input("word-click-filter", "data"),
                Input("errors-only-switch", "value"),
            ],
        )
        def filter_table(selected_indices, word_filter, errors_only):
            if selected_indices is None:
                recs = full_records
            else:
                idx_set = set(selected_indices)
                recs = [r for r in full_records if r["_orig_idx"] in idx_set] or full_records
            if errors_only and has_gt:
                misclassified = [r for r in recs if str(r.get("prediction", "")) != str(r.get("ground_truth", ""))]
                recs = misclassified or recs
            if word_filter:
                word_lower = word_filter.lower()
                filtered = [r for r in recs if word_lower in r["text"].lower()]
                recs = filtered or recs
            title = (
                f'Text Samples — filtered by "{word_filter}"'
                if word_filter
                else "Text Samples — click a row to inspect"
            )
            return recs, [recs[0]], title

        # ── Scatter-specific callbacks (registered only when xy provided) ──
        if self._scatter_xy is not None:

            @self.app.callback(
                Output("scatter-plot", "figure"),
                Input("color-by", "value"),
            )
            def update_scatter_color(color_by):
                return self._build_scatter_fig(color_by or "prediction")

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
                if not selected_data or not selected_data.get("points"):
                    return None
                return [int(pt["customdata"][0]) for pt in selected_data["points"]]

            @self.app.callback(
                Output("scatter-clear-btn", "style"),
                Input("scatter-selected-indices", "data"),
            )
            def toggle_clear_button(selected_indices):
                visible = {"display": "inline", "fontSize": "0.8em"}
                hidden = {"display": "none", "fontSize": "0.8em"}
                return visible if selected_indices else hidden

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, port: int = 8050, debug: bool = False, host: str = "127.0.0.1") -> None:
        """Launch the Dash development server."""
        self.app.run(port=port, debug=debug, host=host)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_classes(self) -> int:
        sample = self.explainer.contributions.values[0]
        return sample.shape[1] if sample.ndim == 2 else 1

    def _sample_data(self, pos: int, label_idx: int):
        """Return tokens, 1-D values, base_value, and label_name for one sample."""
        contrib = self.explainer.contributions
        tokens = contrib.token_strings[pos]
        vals = contrib.values[pos]
        if vals.ndim == 2:
            vals = vals[:, label_idx]

        bv = contrib.base_values
        base_value: float | None = None
        if bv is not None and bv.ndim == 2 and pos < bv.shape[0] and label_idx < bv.shape[1]:
            base_value = float(bv[pos, label_idx])
        elif bv is not None and bv.ndim == 1 and pos < bv.shape[0]:
            base_value = float(bv[pos])

        label_name = (contrib.label_names or [])[label_idx] if contrib.label_names else str(label_idx)
        return tokens, vals, base_value, label_name

    def _build_scatter_fig(self, color_by: str) -> go.Figure:
        """2-D scatter coloured by prediction or ground-truth label.

        One trace per class so the legend works correctly and Plotly's
        box/lasso select dims unselected points across all traces.
        ``customdata`` stores the original sample index so ``selectedData``
        callbacks can recover it regardless of which trace a point is in.
        """
        n = len(self.explainer.texts)
        contrib = self.explainer.contributions

        if color_by == "ground_truth" and getattr(self.explainer, "y_true", None) is not None:
            labels = [str(label) for label in self.explainer.y_true.tolist()]
        elif self.explainer.y_pred is not None:
            labels = [str(label) for label in self.explainer.y_pred.tolist()]
        else:
            labels = [""] * n

        label_names = contrib.label_names or sorted(set(labels))
        texts_short = [(t[:120] + "…") if len(t) > 120 else t for t in self.explainer.texts]
        xy = self._scatter_xy

        fig = go.Figure()
        for i, name in enumerate(label_names):
            mask = [j for j, lbl in enumerate(labels) if lbl == name]
            if not mask:
                continue
            mask_arr = np.array(mask)
            fig.add_trace(
                go.Scattergl(
                    x=xy[mask_arr, 0].tolist(),
                    y=xy[mask_arr, 1].tolist(),
                    mode="markers",
                    marker=dict(color=_PALETTE[i % len(_PALETTE)], size=7, opacity=0.75),
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
