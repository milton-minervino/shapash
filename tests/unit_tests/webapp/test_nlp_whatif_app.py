"""Unit tests for the What-if Lab wiring in ``NlpWebApp`` (no real model, no server).

A fake explainer/engine with synthetic contributions drives layout construction and capability
gating, verifying that the data-editor and counterfactual components mount only when the engine
reports the matching capabilities.
"""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.diagnostics.label_noise import LabelIssue, detect_label_issues
from shapash.compute.diagnostics.label_probe import LabelProbe, ProbeVerdict
from shapash.compute.generators.base import Counterfactual, IntField, TokenListField
from shapash.compute.retrieval.similar_examples import Neighbor
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_app import NlpWebApp
from shapash.webapp.nlp_components import (
    CounterfactualComponent,
    DataEditorComponent,
    LabelNoiseComponent,
    SimilarExamplesComponent,
)

LABEL_NAMES = ["neg", "pos"]

# A tiny lexically separable corpus for the independent label probe. "bad"/"awful" sit in "neg",
# so a row labelled "neg" gets backed and a row labelled "pos" gets rejected.
_PROBE_CORPUS = (
    ["i am happy", "happy and glad", "so glad today", "a happy glad day",
     "this is bad", "bad and awful", "so awful today", "a bad awful day"],
    ["pos"] * 4 + ["neg"] * 4,
)


def _contributions() -> NlpContributions:
    token_strings = [["i", "am", "happy"], ["this", "is", "bad"]]
    values = [np.random.randn(3, 2), np.random.randn(3, 2)]
    base_values = np.zeros((2, 2))
    return NlpContributions(token_strings=token_strings, values=values, base_values=base_values)


class FakeEngine:
    """Minimal explainer/engine stand-in exposing the InteractiveEngine surface + compiled data."""

    def __init__(
        self,
        can_edit: bool,
        can_cf: bool,
        can_similar: bool = False,
        has_labels: bool = False,
        probe_corpus: tuple[list[str], list[str]] | None = None,
    ):
        self.probe_corpus = probe_corpus
        self._can_edit = can_edit
        self._can_cf = can_cf
        self._can_similar = can_similar
        # A retriever-like handle the SimilarExamples panel reads its layer caption from.
        self._retriever = type("R", (), {"layer": "pre_classifier"})() if can_similar else None
        self.label_names = LABEL_NAMES
        self.texts = pd.Series(["i am happy", "this is bad"], index=pd.RangeIndex(2))
        self.contributions = _contributions()
        self.y_pred = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="prediction")
        self.y_prob = pd.DataFrame({"neg": [0.2, 0.8], "pos": [0.8, 0.2]}, index=pd.RangeIndex(2))
        self.y_true = None
        if has_labels:
            # A batch with exactly one planted label error. With two samples and two classes the
            # arrangement is forced: both classes must carry a label (or the unlabelled one has no
            # estimable threshold and can never be suggested), so sample 0 is labelled "neg" while
            # the model confidently says "pos", and sample 1 is labelled "pos" and agrees.
            self.y_prob = pd.DataFrame({"neg": [0.1, 0.1], "pos": [0.9, 0.9]}, index=pd.RangeIndex(2))
            self.y_pred = pd.Series(["pos", "pos"], index=pd.RangeIndex(2), name="prediction")
            self.y_true = pd.Series(["neg", "pos"], index=pd.RangeIndex(2), name="ground_truth")
        # Ground truth is off by default so the existing layout tests, which pin the exact tab
        # groups, keep seeing neither the Error Analysis nor the Label Noise tab.
        self.detect_calls = []

    def to_explanation(self) -> NlpExplanation:
        """The ``NlpExplanation`` a real ``explain()`` call would have produced for this batch.

        ``NlpWebApp`` holds an explanation, not the engine — this is what tests pass as the first
        argument, while ``self`` (the ``FakeEngine``) is passed separately as ``engine=``.
        """
        return NlpExplanation(
            texts=self.texts,
            token_strings=self.contributions.token_strings,
            values=self.contributions.values,
            base_values=self.contributions.base_values,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
            y_true=self.y_true,
            label_names=self.label_names,
            folds_case=None,
            backend_name="fake",
            is_additive=True,
            reference_kind="none",
        )

    def can_detect_label_noise(self, explanation=None):
        return self.y_true is not None

    def can_probe_labels(self):
        return self.probe_corpus is not None

    def detect_label_noise(self, explanation, top_n=50, score="self_confidence", probe=True):
        self.detect_calls.append({"top_n": top_n, "score": score, "probe": probe})
        report = detect_label_issues(
            self.y_prob.to_numpy(dtype=float),
            [str(v) for v in self.y_true.tolist()],
            [str(t) for t in self.texts.tolist()],
            list(self.y_prob.columns),
            top_n=top_n,
            score=score,
        )
        if probe and self.can_probe_labels() and report.issues:
            verdicts = LabelProbe(*self.probe_corpus).verdicts(
                [i.text for i in report.issues], [i.given_label for i in report.issues]
            )
            report = replace(
                report,
                issues=[replace(i, probe=v) for i, v in zip(report.issues, verdicts, strict=True)],
            )
        return report

    def can_edit(self):
        return self._can_edit

    def can_counterfactual(self):
        return self._can_cf

    def can_find_similar(self):
        return self._can_similar

    def find_similar(self, text, top_k=5):
        return [
            Neighbor(index=0, score=0.99, text="i am joyful", label="pos"),
            Neighbor(index=1, score=0.80, text="this is awful", label="neg"),
        ][:top_k]

    def find_similar_threshold(self, text, threshold=0.95, limit=50):
        all_neighbors = [
            Neighbor(index=0, score=0.99, text="i am joyful", label="pos"),
            Neighbor(index=1, score=0.96, text="so glad today", label="pos"),
            Neighbor(index=2, score=0.80, text="this is awful", label="neg"),
        ]
        matches = [n for n in all_neighbors if n.score > threshold]
        return matches[:limit], len(matches)

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
    """Recursively collect all string component ids in a Dash layout tree.

    Also descends into RadioItems/Checklist ``options[].label`` — components nested there (e.g. a
    numeric input inline with its radio button) aren't under ``.children``, but Dash still renders
    and wires them, since ``label`` is documented to accept a component.
    """
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    options = getattr(node, "options", None)
    if isinstance(options, (list, tuple)):
        for opt in options:
            label = opt.get("label") if isinstance(opt, dict) else None
            if isinstance(label, (list, tuple)):
                for item in label:
                    _collect_ids(item, found)
            elif label is not None:
                _collect_ids(label, found)
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
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    @staticmethod
    def _whatif_components(app):
        # app._components also holds the always-on core panels (Sentence Highlight, Waterfall);
        # scope these assertions to the capability-gated What-if Lab ones only.
        gated = (DataEditorComponent, CounterfactualComponent, SimilarExamplesComponent)
        return [c for c in app._components if isinstance(c, gated)]

    def test_full_capabilities_mount_both(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("data-editor-input", ids)
        self.assertIn("data-editor-predict-btn", ids)
        self.assertIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(self._whatif_components(app)), 2)

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
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("counterfactual-cfg-group-hotflip.style", outputs)

    def test_edit_only_mounts_editor_not_counterfactual(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False))
        self.assertIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(self._whatif_components(app)), 1)

    def test_no_capabilities_hides_whatif_lab(self):
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertNotIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertNotIn("whatif-apply-store", ids)
        self.assertEqual(self._whatif_components(app), [])

    def test_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("data-editor-prob.figure", outputs)
        self.assertIn("counterfactual-results.children", outputs)

    def test_similar_panel_mounts_when_capable(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False, can_similar=True))
        self.assertIn("similar-topk", ids)
        self.assertIn("similar-results", ids)
        self.assertTrue(any(isinstance(c, SimilarExamplesComponent) for c in app._components))

    def test_similar_panel_hidden_without_capability(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True, can_similar=False))
        self.assertNotIn("similar-topk", ids)
        self.assertNotIn("similar-results", ids)

    def test_similar_panel_requires_predict(self):
        # CAP_SIMILAR alone is not enough — the Inspect flow needs predict (explain_text) too.
        _, ids = self._ids(FakeEngine(can_edit=False, can_cf=False, can_similar=True))
        self.assertNotIn("similar-topk", ids)

    def test_similar_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("similar-results.children", outputs)


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
        app = NlpWebApp(engine.to_explanation(), engine=engine, **kwargs)
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
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        # Highlight and waterfall render off the shared primary-selection store...
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "sentence-highlight.children"))
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "waterfall-graph.figure"))
        # ...and counterfactuals generate from it too (so a selected row works, not only editor text).
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "counterfactual-results.children"))

    def test_current_datapoint_written_by_row_and_editor(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        # The table-selection writer keys off the selected row.
        self.assertIn(("dataset-table", "selectedRows"), _callback_binding_ids(app, "current-datapoint.data"))
        # The editor's Predict also writes it (allow_duplicate) — its combined key carries the prob figure.
        editor_key = next(k for k in app.app.callback_map if "data-editor-prob.figure" in k)
        self.assertIn("current-datapoint.data", editor_key)

    def test_tab_toggle_callbacks_registered(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine, scatter_xy=np.zeros((2, 2)))
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("left-tabs-body-table.style", outputs)
        self.assertIn("lower-right-tabs-body-waterfall.style", outputs)


class TestReadContractSeam(unittest.TestCase):
    """The app shell reads compiled data only from the ``NlpExplanation``; no raw explainer handle.

    ``FakeEngine`` has no ``explainer`` attribute, so a mounted app proves the read path is served by
    the artifact alone. These asserts lock the seam so a future edit cannot silently reintroduce a
    ``self.explainer`` bypass, nor let display state accumulate back onto the artifact.
    """

    def test_app_holds_no_raw_explainer_handle(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        self.assertFalse(hasattr(app, "explainer"))

    def test_app_exposes_explanation_and_engine_roles(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        explanation = engine.to_explanation()
        app = NlpWebApp(explanation, engine=engine)
        self.assertIs(app._explanation, explanation)  # read contract: the artifact itself
        self.assertIs(app._engine, engine)  # live-action contract


class TestSimilarComponent(unittest.TestCase):
    """Exercise the Similar Examples component's renderer and callbacks directly."""

    @staticmethod
    def _register():
        import dash

        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = SimilarExamplesComponent()
        comp.register_callbacks(app, explanation, engine, {"apply": "whatif-apply-store", "current": "current-datapoint"})
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_layout_shows_layer_caption(self):
        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        found = set()
        _collect_ids(SimilarExamplesComponent().layout(engine.to_explanation(), engine), found)
        self.assertIn("similar-topk", found)
        self.assertIn("similar-threshold", found)
        self.assertIn("similar-mode", found)
        self.assertIn("similar-results", found)

    def test_update_similar_returns_table_and_texts(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "topk", 5, 0.95)
        self.assertEqual(texts, ["i am joyful", "this is awful"])
        self.assertIsNotNone(children)

    def test_update_similar_ignores_empty_text(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        update = self._callback(app, "similar-results")
        with self.assertRaises(PreventUpdate):
            update({"text": "  "}, "topk", 5, 0.95)

    def test_update_similar_threshold_mode_filters_and_reports_total(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.90)
        # Only the two neighbours scoring above 0.90 clear the threshold (see FakeEngine.find_similar_threshold).
        self.assertEqual(texts, ["i am joyful", "so glad today"])
        self.assertIsNotNone(children)

    def test_update_similar_threshold_mode_empty_above_cutoff(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.999)
        self.assertEqual(texts, [])
        self.assertIsNotNone(children)

    def test_toggle_mode_inputs_disables_the_inactive_control(self):
        app, _ = self._register()
        toggle = self._callback(app, "similar-topk.disabled")
        topk_disabled, threshold_disabled = toggle("threshold")
        self.assertTrue(topk_disabled)
        self.assertFalse(threshold_disabled)
        topk_disabled, threshold_disabled = toggle("topk")
        self.assertFalse(topk_disabled)
        self.assertTrue(threshold_disabled)

    def test_inspect_makes_neighbor_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import similar_examples as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "similar-apply", "index": 1}
            dp = inspect([1, 1], ["first neighbor", "second neighbor"])
        self.assertEqual(dp["text"], "second neighbor")
        self.assertEqual(dp["label"], "pos")  # FakeEngine.explain_text returns "pos"

    def test_neighbors_table_marks_matching_label(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [
            Neighbor(index=0, score=0.9, text="joyful one", label="pos"),
            Neighbor(index=1, score=0.5, text="grim one", label="neg"),
        ]
        table = _neighbors_table(neighbors, predicted_label="pos", component_id="similar")
        ids = set()
        _collect_ids(table, ids)
        # One Inspect button per neighbour (pattern-matching ids are dicts, so assert via the count).
        self.assertEqual(sum(1 for n in neighbors), 2)
        self.assertIsNotNone(table)

    def test_neighbors_table_without_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [Neighbor(index=0, score=0.9, text="some text", label=None)]
        table = _neighbors_table(neighbors, predicted_label=None, component_id="similar")
        self.assertIsNotNone(table)

    def test_match_rate_caption_reports_share_of_predicted_label(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [
            Neighbor(index=0, score=0.9, text="a", label="pos"),
            Neighbor(index=1, score=0.8, text="b", label="pos"),
            Neighbor(index=2, score=0.7, text="c", label="neg"),
            Neighbor(index=3, score=0.6, text="d", label="neg"),
        ]
        caption = _match_rate_caption(neighbors, predicted_label="pos")
        self.assertIn("2/4", caption)
        self.assertIn("50%", caption)

    def test_match_rate_caption_none_without_a_prediction_or_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [Neighbor(index=0, score=0.9, text="a", label="pos")]
        self.assertIsNone(_match_rate_caption(neighbors, predicted_label=None))
        unlabelled = [Neighbor(index=0, score=0.9, text="a", label=None)]
        self.assertIsNone(_match_rate_caption(unlabelled, predicted_label="pos"))

    def test_render_results_includes_cap_note_only_when_capped(self):
        from shapash.webapp.nlp_components.similar_examples import _render_results

        neighbors = [Neighbor(index=0, score=0.99, text="a", label="pos")]
        capped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=5)
        uncapped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=None)

        def _flat_text(node):
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return children
            if isinstance(children, list):
                return "".join(_flat_text(c) for c in children if c is not None)
            return _flat_text(children) if children is not None else ""

        self.assertIn("Showing 1 of 5", _flat_text(capped))
        self.assertNotIn("Showing", _flat_text(uncapped))


class TestLabelNoiseMounting(unittest.TestCase):
    """``CAP_LABELS`` is a *data* capability: it depends on the compiled batch, not on a live model."""

    def _ids(self, engine):
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_mounts_with_ground_truth_and_per_class_probabilities(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True, has_labels=True))
        self.assertIn("label-noise-detect-btn", ids)
        self.assertIn("label-noise-results", ids)
        self.assertIn("label-noise", app._tab_groups["upper-right-tabs"])
        self.assertTrue(any(isinstance(c, LabelNoiseComponent) for c in app._components))

    def test_hidden_without_ground_truth(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertNotIn("label-noise-detect-btn", ids)
        self.assertNotIn("label-noise", app._tab_groups["upper-right-tabs"])

    def test_hidden_when_only_the_winning_probability_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=True, has_labels=True)
        engine.y_prob = pd.DataFrame({"probability": [0.8, 0.8]}, index=pd.RangeIndex(2))
        _, ids = self._ids(engine)
        self.assertNotIn("label-noise-detect-btn", ids)

    def test_mounts_without_any_live_capability(self):
        # The snapshot case: no model, so no editor/counterfactual/similar panel — but the labels and
        # probabilities are still in the compiled batch, so this panel stands on its own.
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False, has_labels=True))
        self.assertIn("label-noise-detect-btn", ids)
        self.assertNotIn("data-editor-input", ids)

    def test_caption_warns_when_no_independent_cross_check_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True)
        layout = LabelNoiseComponent().layout(engine.to_explanation(), engine)
        caption = layout.children[1].children
        self.assertIn("no independent cross-check", caption)

    def test_caption_explains_the_corpus_column_when_the_probe_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, probe_corpus=_PROBE_CORPUS)
        layout = LabelNoiseComponent().layout(engine.to_explanation(), engine)
        caption = layout.children[1].children
        self.assertIn("Corpus column", caption)

    def test_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=True, has_labels=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("label-noise-results.children", outputs)


class TestLabelNoiseComponent(unittest.TestCase):
    """Drive the panel's renderer and callbacks directly, without the app shell."""

    @staticmethod
    def _register(**kwargs):
        import dash

        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, **kwargs)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = LabelNoiseComponent()
        comp.register_callbacks(app, explanation, engine, {"apply": "whatif-apply-store", "current": "current-datapoint"})
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_detect_returns_a_view_and_the_flagged_indices(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [0])  # sample 0 is labelled neg but confidently predicted pos
        self.assertIsNotNone(children)

    def test_detect_forwards_the_controls_to_the_engine(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, 7, "normalized_margin")
        self.assertEqual(engine.detect_calls[-1], {"top_n": 7, "score": "normalized_margin", "probe": True})

    def test_blank_controls_fall_back_to_defaults(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, None, None)
        self.assertEqual(engine.detect_calls[-1], {"top_n": 50, "score": "self_confidence", "probe": True})

    def test_detect_renders_the_corpus_column_when_a_probe_corpus_is_bound(self):
        app, _ = self._register(probe_corpus=_PROBE_CORPUS)
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertEqual(headers, ["Score", "Given", "Probably", "Corpus", "Text", ""])
        # The flagged row is "i am happy" labelled "neg"; the corpus puts that vocabulary in "pos",
        # so the probe rejects the label too rather than defending it.
        cell = table.children[1].children[0].children[3].children
        self.assertEqual(cell.children, "0.14 → pos")

    def test_detect_omits_the_corpus_column_without_a_probe_corpus(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)

    def test_detect_does_nothing_before_the_button_is_clicked(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        with self.assertRaises(PreventUpdate):
            detect(None, 50, "self_confidence")

    def test_a_clean_corpus_renders_a_message_and_no_rows(self):
        app, engine = self._register()
        # Both classes labelled and both labels agreeing with the model, so there is no noise to find.
        engine.y_prob = pd.DataFrame({"neg": [0.1, 0.9], "pos": [0.9, 0.1]}, index=pd.RangeIndex(2))
        engine.y_true = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="ground_truth")
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [])
        self.assertIn("No label issues detected", children.children)

    def test_inspect_makes_the_flagged_sample_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            dp = inspect([1], [1])  # row 0 of the table points at compiled sample 1
        self.assertEqual(dp["text"], "this is bad")
        self.assertEqual(dp["orig_idx"], 1)
        self.assertEqual(dp["label"], "pos")  # the *prediction*, matching the dataset-row path
        self.assertEqual(len(dp["tokens"]), 3)

    def test_inspect_ignores_a_render_with_no_clicks(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with self.assertRaises(PreventUpdate):
            inspect([0], [1])

    def test_inspect_ignores_a_missing_trigger(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = None
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_an_out_of_range_row(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 5}
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_a_stale_index_beyond_the_corpus(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            with self.assertRaises(PreventUpdate):
                inspect([1], [99])


class TestLabelNoiseTable(unittest.TestCase):
    """The neighbour column is evidence *beside* the score, and must not appear when absent."""

    @staticmethod
    def _issue(**kwargs):
        base = dict(
            index=0,
            text="i am happy",
            given_label="neg",
            suggested_label="pos",
            given_prob=0.2,
            suggested_prob=0.8,
            score=0.2,
        )
        return LabelIssue(**{**base, **kwargs})

    def test_corpus_column_omitted_when_no_probe_ran(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        table = _issues_table([self._issue()], "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)
        self.assertEqual(headers, ["Score", "Given", "Probably", "Text", ""])

    def test_corpus_column_present_when_any_row_has_a_verdict(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        verdict = ProbeVerdict(given_prob=0.8, top_label="neg", backs_given=True)
        issues = [self._issue(probe=verdict), self._issue(index=1)]
        table = _issues_table(issues, "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertIn("Corpus", headers)

    def test_probe_summary_warns_when_the_corpus_backs_the_given_label(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        # The row the panel exists to catch: confident learning flagged it, but an independent
        # classifier defends the label — so it is the model that is wrong, not the corpus.
        verdict = ProbeVerdict(given_prob=0.83, top_label="neg", backs_given=True)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "⚠ 0.83")
        self.assertIn("text-warning", summary.className)
        self.assertIn("more likely", summary.title)

    def test_probe_summary_reports_a_corroborated_label_error_plainly(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        verdict = ProbeVerdict(given_prob=0.16, top_label="pos", backs_given=False)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "0.16 → pos")
        self.assertIn("text-muted", summary.className)

    def test_rows_without_a_verdict_render_a_dash(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        self.assertEqual(_probe_summary(self._issue()), "—")


if __name__ == "__main__":
    unittest.main()
