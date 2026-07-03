"""Counterfactual component: generate what-if flips and apply them into the editor.

Controls are rendered automatically from the bound generator's ``config_spec()`` (so a new knob or a
new generator needs no UI change). Generated counterfactuals are shown as a table; an *Apply* button
publishes the chosen text to a shared store the data editor subscribes to.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.compute.generators.base import Field, IntField, TokenListField
from shapash.webapp.nlp_components.base import CAP_COUNTERFACTUAL, CAP_PREDICT, WebappComponent
from shapash.webapp.nlp_components.datapoint import datapoint_from_contributions

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}


class CounterfactualComponent(WebappComponent):
    """Spec-driven counterfactual generation panel."""

    id = "counterfactual"
    name = "Counterfactual Suggestions"
    scope = "local"
    # Requires the editor too (it reads the editor's text and applies flips back into it).
    requires = frozenset({CAP_COUNTERFACTUAL, CAP_PREDICT})

    def _config_fields(self, engine) -> list[tuple[str, Field]]:
        """Ordered ``(name, field)`` pairs from the engine's config spec."""
        return list(engine.cf_config_spec().items())

    def _config_controls(self, engine) -> list:
        """Render one control per config field (IntField → number, TokenListField → text)."""
        rows = []
        for name, fld in self._config_fields(engine):
            cid = f"{self.id}-cfg-{name}"
            if isinstance(fld, IntField):
                control = dcc.Input(
                    id=cid,
                    type="number",
                    min=fld.minimum,
                    max=fld.maximum,
                    step=1,
                    value=fld.default,
                    style={"width": "90px"},
                )
            elif isinstance(fld, TokenListField):
                control = dbc.Input(
                    id=cid,
                    type="text",
                    value=",".join(fld.default),
                    placeholder="comma,separated",
                    style={"width": "180px"},
                )
            else:  # pragma: no cover - future field types
                control = dbc.Input(id=cid, type="text", value=str(fld.default))
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(html.Label(fld.label, className="small fw-bold mb-0"), width="auto"),
                        dbc.Col(control, width="auto"),
                    ],
                    className="align-items-center mb-2 g-2",
                )
            )
        return rows

    def layout(self, view, engine=None) -> html.Div:
        """Return the counterfactual card with config controls rendered from the generator spec.

        The controls are built here (not injected by a callback) so their ids exist in the initial
        layout — otherwise the ``generate`` callback's ``State`` references a not-yet-created object.
        """
        controls = self._config_controls(engine) if engine is not None else []
        return html.Div(
            [
                html.H6("Counterfactual Suggestions", className="fw-bold mb-2"),
                html.Small(
                    "Find minimal token substitutions that flip the prediction of the edited text.",
                    className="text-muted d-block mb-2",
                ),
                html.Div(controls, id=f"{self.id}-controls"),
                dbc.Button("Generate", id=f"{self.id}-generate-btn", color="secondary", size="sm", className="mb-2"),
                dcc.Loading(html.Div(id=f"{self.id}-results")),
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, view, engine, stores) -> None:
        """Wire control rendering, Generate, and per-row Apply (→ shared editor store)."""
        apply_store = stores["apply"]
        current_store = stores["current"]
        fields = self._config_fields(engine)

        # Config controls are rendered in `layout()` (they must be in the initial layout), so no
        # separate render callback is needed here.
        config_states = [State(f"{self.id}-cfg-{name}", "value") for name, _ in fields]

        # Reads the current datapoint (a selected dataset row *or* an edited/predicted text) so
        # counterfactuals can be generated for both, not only hand-typed editor text.
        @app.callback(
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(f"{self.id}-generate-btn", "n_clicks"),
            State(current_store, "data"),
            *config_states,
        )
        def generate(n_clicks, datapoint, *config_values):
            text = (datapoint or {}).get("text", "")
            if not n_clicks or not text or not text.strip():
                raise PreventUpdate
            config = {}
            for (name, fld), value in zip(fields, config_values, strict=True):
                if isinstance(fld, IntField):
                    config[name] = int(value) if value is not None else fld.default
                elif isinstance(fld, TokenListField):
                    config[name] = [t.strip() for t in (value or "").split(",") if t.strip()]
                else:  # pragma: no cover
                    config[name] = value
            cfs = engine.generate_counterfactuals(text, config=config)
            if not cfs:
                return html.Div("No counterfactual found within these limits.", className="text-muted"), []
            return _results_table(cfs, self.id), [cf.new_text for cf in cfs]

        # "Inspect" loads the chosen counterfactual into the editor (apply_store → textarea) *and*
        # makes it the current datapoint, so its token contributions render immediately in the shared
        # Sentence/Waterfall panels without a separate Predict click.
        @app.callback(
            Output(apply_store, "data"),
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-apply", "index": ALL}, "n_clicks"),
            State(f"{self.id}-store", "data"),
            prevent_initial_call=True,
        )
        def apply(n_clicks_list, cf_texts):
            if not any(n_clicks_list or []):
                raise PreventUpdate
            triggered = callback_context.triggered_id
            if not triggered:
                raise PreventUpdate
            index = triggered["index"]
            if not cf_texts or not (0 <= index < len(cf_texts)):
                raise PreventUpdate
            text = cf_texts[index]
            contributions, label, _probs = engine.explain_text(text)
            return text, datapoint_from_contributions(text, contributions, label)


def _results_table(cfs, component_id: str):
    """Render counterfactuals as a table with per-row Apply buttons."""
    header = html.Thead(
        html.Tr([html.Th("→ Label"), html.Th("Δ prob"), html.Th("Substitutions"), html.Th("Text"), html.Th("")])
    )
    rows = []
    for i, cf in enumerate(cfs):
        subs = ", ".join(f"{old}→{new}" for _, old, new in cf.substitutions)
        rows.append(
            html.Tr(
                [
                    html.Td(html.B(cf.new_label)),
                    html.Td(f"{cf.prob_delta:.2f}"),
                    html.Td(subs, style={"fontSize": "0.85em"}),
                    html.Td(cf.new_text, style={"fontSize": "0.85em"}),
                    html.Td(
                        dbc.Button(
                            "Inspect",
                            id={"type": f"{component_id}-apply", "index": i},
                            color="link",
                            size="sm",
                            className="p-0",
                            title="Load into the editor and show its token contributions on the right",
                        )
                    ),
                ]
            )
        )
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0")
