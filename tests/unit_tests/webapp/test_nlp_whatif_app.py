"""Unit tests for the What-if Lab wiring in ``NlpWebApp`` (no real model, no server).

A fake explainer/engine with synthetic contributions drives layout construction and capability
gating, verifying that the data-editor and counterfactual components mount only when the engine
reports the matching capabilities.
"""

import unittest

import numpy as np
import pandas as pd

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.generators.base import Counterfactual, IntField, TokenListField
from shapash.webapp.nlp_app import NlpWebApp
from shapash.webapp.nlp_view import NlpView

LABEL_NAMES = ["neg", "pos"]


def _contributions() -> NlpContributions:
    token_strings = [["i", "am", "happy"], ["this", "is", "bad"]]
    values = [np.random.randn(3, 2), np.random.randn(3, 2)]
    base_values = np.zeros((2, 2))
    c = NlpContributions(token_strings=token_strings, values=values, base_values=base_values)
    c.label_names = LABEL_NAMES
    c.index = pd.RangeIndex(2)
    return c


class FakeEngine:
    """Minimal explainer/engine stand-in exposing the InteractiveEngine surface + compiled data."""

    def __init__(self, can_edit: bool, can_cf: bool):
        self._can_edit = can_edit
        self._can_cf = can_cf
        self.label_names = LABEL_NAMES
        self.texts = pd.Series(["i am happy", "this is bad"], index=pd.RangeIndex(2))
        self.contributions = _contributions()
        self.y_pred = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="prediction")
        self.y_prob = pd.DataFrame({"neg": [0.2, 0.8], "pos": [0.8, 0.2]}, index=pd.RangeIndex(2))
        self.y_true = None

    def can_edit(self):
        return self._can_edit

    def can_counterfactual(self):
        return self._can_cf

    def available_cf_generators(self):
        return [("hotflip", "HotFlip"), ("ablation_flip", "Ablation")]

    def cf_config_spec(self, generator=None):
        max_field = "max_ablations" if generator == "ablation_flip" else "max_flips"
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            max_field: IntField(label="Max token edits", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    def predict(self, text):
        return "pos", {"neg": 0.3, "pos": 0.7}

    def explain_text(self, text):
        c = _contributions()
        return c, "pos", {"neg": 0.3, "pos": 0.7}

    def generate_counterfactuals(self, text, config=None, generator=None):
        return [
            Counterfactual(
                original_text=text,
                new_text=text.replace("happy", "bad"),
                tokens=["i", "am", "happy"],
                flipped_positions=[2],
                substitutions=[(2, "happy", "bad")],
                orig_label="pos",
                new_label="neg",
                orig_prob=0.8,
                new_prob=0.7,
                prob_delta=0.5,
            )
        ]


def _collect_ids(node, found):
    """Recursively collect all string component ids in a Dash layout tree."""
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for ch in children:
            _collect_ids(ch, found)
    else:
        _collect_ids(children, found)


class TestWhatIfMounting(unittest.TestCase):
    def _ids(self, engine):
        app = NlpWebApp(engine)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_full_capabilities_mount_both(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("data-editor-input", ids)
        self.assertIn("data-editor-predict-btn", ids)
        self.assertIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(app._components), 2)

    def test_counterfactual_config_controls_in_initial_layout(self):
        # The generate callback's State references these ids, so they must exist in the initial
        # layout (rendered per generator from cf_config_spec), not be injected by a later callback.
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        for name in ("num_examples", "max_flips", "tokens_to_ignore"):
            self.assertIn(f"counterfactual-cfg-hotflip-{name}", ids)
        for name in ("num_examples", "max_ablations", "tokens_to_ignore"):
            self.assertIn(f"counterfactual-cfg-ablation_flip-{name}", ids)

    def test_counterfactual_method_selector_present_with_multiple_generators(self):
        # A method selector and one visibility-toggled control group per generator are in the layout.
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("counterfactual-generator", ids)
        self.assertIn("counterfactual-cfg-group-hotflip", ids)
        self.assertIn("counterfactual-cfg-group-ablation_flip", ids)

    def test_selector_toggle_callback_registered_with_multiple_generators(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True))
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("counterfactual-cfg-group-hotflip.style", outputs)

    def test_edit_only_mounts_editor_not_counterfactual(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False))
        self.assertIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(app._components), 1)

    def test_no_capabilities_hides_whatif_lab(self):
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertNotIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertNotIn("whatif-apply-store", ids)
        self.assertEqual(app._components, [])

    def test_callbacks_registered_when_mounted(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True))
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("data-editor-prob.figure", outputs)
        self.assertIn("counterfactual-results.children", outputs)


def _callback_binding_ids(app, output_substr):
    """Return the (id, property) pairs bound as inputs+state for the callback with this output."""
    for key, spec in app.app.callback_map.items():
        if output_substr in key:
            pairs = [(i["id"], i["property"]) for i in spec["inputs"]]
            pairs += [(s["id"], s["property"]) for s in spec.get("state", [])]
            return pairs
    raise KeyError(output_substr)


class TestThreePanelLayout(unittest.TestCase):
    """The LIT-style three-panel shell: tab groups, mounted bodies, and the current-datapoint store."""

    def _ids(self, engine, **kwargs):
        app = NlpWebApp(engine, **kwargs)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_full_tab_groups_when_all_panels_available(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), scatter_xy=np.zeros((2, 2)))
        self.assertEqual(
            app._tab_groups,
            {
                "left-tabs": ["table", "scatter", "editor"],
                "upper-right-tabs": ["importance", "counterfactual"],
                "lower-right-tabs": ["highlight", "waterfall"],
            },
        )
        # Every tab body is mounted in the DOM (visibility is toggled, not the mount).
        for body_id in (
            "left-tabs-body-table",
            "left-tabs-body-scatter",
            "left-tabs-body-editor",
            "upper-right-tabs-body-importance",
            "upper-right-tabs-body-counterfactual",
            "lower-right-tabs-body-highlight",
            "lower-right-tabs-body-waterfall",
        ):
            self.assertIn(body_id, ids)

    def test_tabs_degrade_without_scatter_or_whatif(self):
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertEqual(app._tab_groups["left-tabs"], ["table"])
        self.assertEqual(app._tab_groups["upper-right-tabs"], ["importance"])
        self.assertEqual(app._tab_groups["lower-right-tabs"], ["highlight", "waterfall"])
        self.assertNotIn("left-tabs-body-scatter", ids)
        self.assertNotIn("left-tabs-body-editor", ids)
        self.assertNotIn("upper-right-tabs-body-counterfactual", ids)

    def test_counterfactual_tab_only_with_editor(self):
        # Editor without CF: editor tab present on the left, no counterfactual tab on the right.
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False))
        self.assertIn("editor", app._tab_groups["left-tabs"])
        self.assertNotIn("counterfactual", app._tab_groups["upper-right-tabs"])

    def test_selection_bar_present(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), scatter_xy=np.zeros((2, 2)))
        self.assertIn("selection-summary", ids)
        self.assertIn("word-filter-clear-btn", ids)
        self.assertIn("scatter-clear-btn", ids)

    def test_scatter_clear_absent_without_scatter(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("word-filter-clear-btn", ids)
        self.assertNotIn("scatter-clear-btn", ids)

    def test_current_datapoint_store_present(self):
        _, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertIn("current-datapoint", ids)

    def test_detail_panels_read_current_datapoint(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True))
        # Highlight and waterfall render off the shared primary-selection store...
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "sentence-highlight.children"))
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "waterfall-graph.figure"))
        # ...and counterfactuals generate from it too (so a selected row works, not only editor text).
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "counterfactual-results.children"))

    def test_current_datapoint_written_by_row_and_editor(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True))
        # The table-selection writer keys off the selected row.
        self.assertIn(("dataset-table", "selectedRows"), _callback_binding_ids(app, "current-datapoint.data"))
        # The editor's Predict also writes it (allow_duplicate) — its combined key carries the prob figure.
        editor_key = next(k for k in app.app.callback_map if "data-editor-prob.figure" in k)
        self.assertIn("current-datapoint.data", editor_key)

    def test_tab_toggle_callbacks_registered(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True), scatter_xy=np.zeros((2, 2)))
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("left-tabs-body-table.style", outputs)
        self.assertIn("lower-right-tabs-body-waterfall.style", outputs)


class TestReadContractSeam(unittest.TestCase):
    """The app shell reads compiled data only through ``NlpView``; it keeps no raw explainer handle.

    ``FakeEngine`` exposes the view/engine surface but has no ``explainer`` attribute, so a mounted
    app proves the read path is served by the contract alone. These asserts lock the seam so a future
    edit cannot silently reintroduce a ``self.explainer`` bypass (see nlp_app engine/view split).
    """

    def test_app_holds_no_raw_explainer_handle(self):
        app = NlpWebApp(FakeEngine(can_edit=True, can_cf=True))
        self.assertFalse(hasattr(app, "explainer"))

    def test_app_exposes_view_and_engine_roles(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine)
        self.assertIsInstance(app._view, NlpView)  # read contract
        self.assertIs(app._engine, engine)  # live-action contract


if __name__ == "__main__":
    unittest.main()
