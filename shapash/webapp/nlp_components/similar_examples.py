"""Similar-examples component: reference examples most like the selected/edited text.

Reads the shared ``current-datapoint`` store — so it serves a clicked dataset row, an edited sentence,
*and* an applied counterfactual through the same code path — and renders the top-k most similar
reference-corpus examples (cosine similarity in the model's similarity layer). Each neighbour shows its
similarity, its label, whether that label matches the current prediction, its text, and an *Inspect*
button that (mirroring the counterfactual panel) re-explains that example live and makes it the current
datapoint, so its token contributions render immediately in the Sentence/Waterfall panels.

The lookup is one forward pass against a precomputed activation bank, so it runs live on every
selection change. Gated by ``CAP_SIMILAR`` + ``CAP_PREDICT``: mounts only when the explainer holds a
reference corpus and a live, activation-capable model (self-disables on a snapshot or a
prediction-only pipeline).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.webapp.nlp_components.base import CAP_PREDICT, CAP_SIMILAR, WebappComponent
from shapash.webapp.nlp_components.datapoint import datapoint_from_contributions

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}
_INLINE = {"display": "flex", "alignItems": "center", "gap": "0.4rem"}
_DEFAULT_TOP_K = 5


class SimilarExamplesComponent(WebappComponent):
    """Nearest reference examples for the current datapoint (representation similarity)."""

    id = "similar"
    name = "Similar Examples"
    scope = "local"
    # Needs predict too: the Inspect button re-explains a chosen example via engine.explain_text.
    requires = frozenset({CAP_SIMILAR, CAP_PREDICT})

    def _layer_name(self, engine) -> str | None:
        """The similarity layer the retriever compares in, for a transparency caption (or ``None``)."""
        retriever = getattr(engine, "_retriever", None)
        return getattr(retriever, "layer", None) if retriever is not None else None

    def layout(self, view, engine=None) -> html.Div:
        """Return the panel: a top-k control plus a results area fed by the current datapoint."""
        layer = self._layer_name(engine)
        caption = "Reference examples most similar to the selected text in the model's decision space"
        if layer:
            caption += f" (layer: {layer})"
        caption += ". Click Inspect to explain one here."
        return html.Div(
            [
                html.H6("Similar Examples", className="fw-bold mb-2"),
                html.Small(caption, className="text-muted d-block mb-2"),
                html.Div(
                    [
                        html.Label("Top-K", className="small fw-bold mb-0"),
                        dcc.Input(
                            id=f"{self.id}-topk",
                            type="number",
                            min=1,
                            max=25,
                            step=1,
                            value=_DEFAULT_TOP_K,
                            style={"width": "80px"},
                        ),
                    ],
                    style=_INLINE,
                    className="mb-2",
                ),
                dcc.Loading(html.Div(id=f"{self.id}-results")),
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, view, engine, stores) -> None:
        """Recompute neighbours on selection/top-k change; wire per-row Inspect into the current store."""
        current_store = stores["current"]

        @app.callback(
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(current_store, "data"),
            Input(f"{self.id}-topk", "value"),
        )
        def update_similar(datapoint, top_k):
            text = (datapoint or {}).get("text", "")
            if not text or not text.strip():
                raise PreventUpdate
            k = int(top_k) if top_k else _DEFAULT_TOP_K
            neighbors = engine.find_similar(text, top_k=k)
            if not neighbors:
                return html.Div("No reference examples available.", className="text-muted"), []
            predicted = (datapoint or {}).get("label")
            return _neighbors_table(neighbors, predicted, self.id), [n.text for n in neighbors]

        # "Inspect" re-explains the chosen reference example and makes it the current datapoint, so its
        # token contributions render in the shared Sentence/Waterfall panels — the same flow the
        # counterfactual panel's Inspect uses (current-datapoint is a shared, allow_duplicate output).
        @app.callback(
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-apply", "index": ALL}, "n_clicks"),
            State(f"{self.id}-store", "data"),
            prevent_initial_call=True,
        )
        def inspect(n_clicks_list, neighbor_texts):
            if not any(n_clicks_list or []):
                raise PreventUpdate
            triggered = callback_context.triggered_id
            if not triggered:
                raise PreventUpdate
            index = triggered["index"]
            if not neighbor_texts or not (0 <= index < len(neighbor_texts)):
                raise PreventUpdate
            text = neighbor_texts[index]
            contributions, label, _probs = engine.explain_text(text)
            return datapoint_from_contributions(text, contributions, label)


def _neighbors_table(neighbors, predicted_label, component_id):
    """Render neighbours as a table with a match flag and a per-row Inspect button."""
    show_label = any(n.label is not None for n in neighbors)
    head_cells = [html.Th("Cosine")]
    if show_label:
        head_cells.append(html.Th("Label"))
    head_cells.extend([html.Th("Text"), html.Th("")])
    header = html.Thead(html.Tr(head_cells))

    rows = []
    for i, n in enumerate(neighbors):
        cells = [html.Td(f"{n.score:.3f}")]
        if show_label:
            matches = predicted_label is not None and n.label == predicted_label
            cells.append(
                html.Td(
                    [
                        html.B(n.label if n.label is not None else "—"),
                        html.Span(" ✓", title="matches the current prediction") if matches else "",
                    ]
                )
            )
        cells.append(html.Td(n.text, style={"fontSize": "0.85em"}))
        cells.append(
            html.Td(
                dbc.Button(
                    "Inspect",
                    id={"type": f"{component_id}-apply", "index": i},
                    color="link",
                    size="sm",
                    className="p-0",
                    title="Explain this example here and show its token contributions on the right",
                )
            )
        )
        rows.append(html.Tr(cells))
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0")
