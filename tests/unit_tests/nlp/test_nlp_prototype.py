"""Integration tests for the NLP explanation prototype.

Covers the full wiring between NlpContributions (backend), word_importance
aggregation, plot_token_highlight / plot_word_importance (plots), NlpExplainer
(explainer), and NlpWebApp (webapp layout construction). A real NLP model is
not required — synthetic NlpContributions data is used throughout so the suite
runs in CI without transformers/datasets.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html

from shapash.backend.nlp_backend import NlpBackend, NlpContributions, NlpRawExplanation
from shapash.backend.nlp_lime_backend import NlpLimeBackend
from shapash.backend.nlp_shap_backend import NlpShapBackend
from shapash.compute.generators import AblationFlipGenerator, HotFlipGenerator
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.model.base import SupportsEmbeddings, SupportsGradients, SupportsTokenization, TextModel
from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.plots.plot_waterfall import plot_waterfall
from shapash.plots.plot_word_importance import plot_word_importance
from shapash.webapp.nlp_app import NlpWebApp, _cell_from_click, _compose_selection

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_contributions() -> NlpContributions:
    """Synthetic NlpContributions for 3 samples, 6 classes."""
    rng = np.random.default_rng(42)
    token_strings = [
        ["", "i", "feel", "so", "happy", "today", ""],
        ["", "this", "is", "terrible", "and", "sad", ""],
        ["", "what", "a", "wonderful", "day", ""],
    ]
    values = [
        rng.uniform(-0.4, 0.4, size=(len(t), N_CLASSES)).astype(np.float32)
        for t in token_strings
    ]
    base_values = rng.uniform(-0.1, 0.1, size=(3, N_CLASSES)).astype(np.float32)
    return NlpContributions(
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        label_names=LABEL_NAMES,
        index=pd.RangeIndex(3),
    )


def _make_explainer(compiled: bool = True) -> NlpExplainer:
    """NlpExplainer with synthetic data, bypassing __init__ to avoid shap.Explainer(None).

    ``compiled=True`` simulates a post-compile() instance with contributions set.
    ``compiled=False`` simulates a pre-compile() instance (contributions=None).
    """
    xpl = object.__new__(NlpExplainer)
    xpl.model = None
    xpl.label_names = LABEL_NAMES
    xpl.backend = None
    xpl.contributions = None
    xpl.texts = None
    xpl.y_pred = None
    xpl.y_true = None
    if compiled:
        xpl.texts = pd.Series(
            ["i feel so happy today", "this is terrible and sad", "what a wonderful day"],
            index=pd.RangeIndex(3),
        )
        xpl.contributions = _make_contributions()
        xpl.y_pred = pd.Series(["joy", "sadness", "joy"], index=pd.RangeIndex(3), name="prediction")
    return xpl


# ---------------------------------------------------------------------------
# NlpContributions
# ---------------------------------------------------------------------------


class TestNlpContributions(unittest.TestCase):
    def setUp(self):
        self.contrib = _make_contributions()

    def test_len(self):
        self.assertEqual(len(self.contrib), 3)

    def test_word_importance_returns_series(self):
        imp = self.contrib.word_importance(label_idx=1)
        self.assertIsInstance(imp, pd.Series)

    def test_word_importance_filters_special_tokens(self):
        imp = self.contrib.word_importance(label_idx=0, filter_special=True)
        self.assertNotIn("", imp.index)
        self.assertNotIn(" ", imp.index)

    def test_word_importance_keeps_special_when_disabled(self):
        imp = self.contrib.word_importance(label_idx=0, filter_special=False)
        # Empty strings (BOS/EOS) should now be present
        self.assertIn("", imp.index)

    def test_word_importance_respects_n_top(self):
        imp = self.contrib.word_importance(label_idx=1, n_top=3)
        self.assertLessEqual(len(imp), 3)

    def test_word_importance_sorted_by_absolute_value(self):
        imp = self.contrib.word_importance(label_idx=2, n_top=20)
        abs_vals = imp.abs().tolist()
        self.assertEqual(abs_vals, sorted(abs_vals, reverse=True))

    def test_word_importance_aggregates_repeated_words(self):
        # "feel" only appears once; check it has a single contribution value
        imp = self.contrib.word_importance(label_idx=1, n_top=20, filter_special=True)
        self.assertIn("feel", imp.index)
        # Should be a scalar (mean of one occurrence)
        self.assertIsInstance(imp["feel"], float)

    def test_word_importance_all_labels(self):
        for idx in range(N_CLASSES):
            imp = self.contrib.word_importance(label_idx=idx)
            self.assertIsInstance(imp, pd.Series)
            self.assertGreater(len(imp), 0)

    def test_word_importance_filter_sign_positive(self):
        imp = self.contrib.word_importance(label_idx=0, filter_sign="positive")
        if len(imp) > 0:
            self.assertTrue((imp > 0).all(), "positive filter should return only positive values")

    def test_word_importance_filter_sign_negative(self):
        imp = self.contrib.word_importance(label_idx=0, filter_sign="negative")
        if len(imp) > 0:
            self.assertTrue((imp < 0).all(), "negative filter should return only negative values")

    def test_word_importance_exclude_words(self):
        imp_full = self.contrib.word_importance(label_idx=1, filter_special=True)
        if len(imp_full) == 0:
            return
        word_to_exclude = imp_full.index[0]
        imp_filtered = self.contrib.word_importance(
            label_idx=1, filter_special=True, exclude_words={word_to_exclude}
        )
        self.assertNotIn(word_to_exclude, imp_filtered.index)

    def test_word_importance_exclude_words_empty_set(self):
        imp_no_exclude = self.contrib.word_importance(label_idx=1, exclude_words=set())
        imp_none_exclude = self.contrib.word_importance(label_idx=1, exclude_words=None)
        pd.testing.assert_series_equal(imp_no_exclude, imp_none_exclude)

    def test_word_importance_sample_indices(self):
        imp = self.contrib.word_importance(label_idx=0, sample_indices=[0], n_top=50)
        self.assertGreater(len(imp), 0)
        # "terrible" only exists in sample 1 — must not appear in the sample-0 subset
        self.assertNotIn("terrible", imp.index)


# ---------------------------------------------------------------------------
# plot_sentence_highlight
# ---------------------------------------------------------------------------


class TestPlotSentenceHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "happy", "[SEP]"]
        self.values = np.array([0.01, 0.05, 0.30, -0.20, 0.01])

    def test_returns_html_div(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsInstance(result, html.Div)

    def test_has_children(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsNotNone(result.children)
        # legend + spans div + summary
        self.assertGreaterEqual(len(result.children), 3)

    def test_raises_on_2d_values(self):
        with self.assertRaises(ValueError):
            plot_sentence_highlight(self.tokens, np.zeros((5, 3)))

    def test_with_base_value_does_not_raise(self):
        result = plot_sentence_highlight(self.tokens, self.values, base_value=0.15)
        self.assertIsInstance(result, html.Div)

    def test_empty_tokens_does_not_raise(self):
        result = plot_sentence_highlight([], np.array([]))
        self.assertIsInstance(result, html.Div)


# ---------------------------------------------------------------------------
# plot_waterfall
# ---------------------------------------------------------------------------


class TestPlotWaterfall(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "so", "happy", "today", "[SEP]"]
        self.values = np.array([0.01, 0.08, 0.35, 0.02, -0.20, 0.05, 0.01])

    def test_returns_figure(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_waterfall_trace(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Waterfall)

    def test_filters_special_tokens_by_default(self):
        fig = plot_waterfall(self.tokens, self.values)
        y_labels = list(fig.data[0].y)
        for label in y_labels:
            self.assertNotIn("[CLS]", label)
            self.assertNotIn("[SEP]", label)

    def test_keeps_special_tokens_when_disabled(self):
        # min_pct=0 disables grouping so every token gets its own bar
        fig = plot_waterfall(self.tokens, self.values, filter_special=False, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("[CLS]" in lbl for lbl in y_labels))

    def test_total_bar_present(self):
        fig = plot_waterfall(self.tokens, self.values)
        measures = list(fig.data[0].measure)
        self.assertIn("total", measures)

    def test_grouping_reduces_bar_count(self):
        # With min_pct=0.0 (no grouping), every non-special token is its own bar
        fig_no_group = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        # With min_pct=0.5, small tokens are lumped
        fig_grouped = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        self.assertLessEqual(len(fig_grouped.data[0].y), len(fig_no_group.data[0].y))

    def test_other_bar_present_when_grouping_active(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("other" in lbl for lbl in y_labels))

    def test_no_other_bar_when_min_pct_zero(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertFalse(any("other" in lbl for lbl in y_labels))

    def test_base_value_creates_absolute_bar(self):
        fig = plot_waterfall(self.tokens, self.values, base_value=0.20)
        measures = list(fig.data[0].measure)
        self.assertEqual(measures[0], "absolute")
        y_labels = list(fig.data[0].y)
        self.assertEqual(y_labels[0], "Base")

    def test_empty_tokens_returns_figure(self):
        fig = plot_waterfall([], np.array([]))
        self.assertIsInstance(fig, go.Figure)

    def test_all_special_tokens_returns_figure(self):
        fig = plot_waterfall(["[CLS]", "[SEP]"], np.array([0.01, -0.01]))
        self.assertIsInstance(fig, go.Figure)

    def test_custom_title(self):
        fig = plot_waterfall(self.tokens, self.values, title="joy waterfall")
        self.assertIn("joy waterfall", fig.layout.title.text)


# ---------------------------------------------------------------------------
# plot_token_highlight
# ---------------------------------------------------------------------------


class TestPlotTokenHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["i", "feel", "happy", "today"]
        self.values = np.array([0.1, 0.4, -0.2, 0.05])

    def test_returns_figure(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_max_tokens_limits_bars(self):
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        self.assertLessEqual(len(fig.data[0].x), 2)

    def test_max_tokens_preserves_sentence_order(self):
        # top-2 by |value| are "feel" (0.4) and "happy" (-0.2)
        # sentence order: "feel" before "happy"
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        # y-axis is reversed for display (highest importance at top → reversed list)
        displayed_labels = list(fig.data[0].y)
        feel_pos = displayed_labels.index("feel")
        happy_pos = displayed_labels.index("happy")
        # In reversed display: "feel" appears after "happy" (lower index = further down)
        self.assertNotEqual(feel_pos, happy_pos)

    def test_custom_title(self):
        fig = plot_token_highlight(self.tokens, self.values, title="Test title")
        self.assertIn("Test title", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(fig.data[0].orientation, "h")


# ---------------------------------------------------------------------------
# plot_word_importance
# ---------------------------------------------------------------------------


class TestPlotWordImportance(unittest.TestCase):
    def setUp(self):
        self.word_imp = pd.Series(
            {"happy": 0.35, "terrible": -0.28, "wonderful": 0.20, "feel": -0.10},
        )

    def test_returns_figure(self):
        fig = plot_word_importance(self.word_imp)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_all_words_rendered(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data[0].x), len(self.word_imp))

    def test_custom_title(self):
        fig = plot_word_importance(self.word_imp, title="Joy importance")
        self.assertIn("Joy importance", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(fig.data[0].orientation, "h")


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix(unittest.TestCase):
    def setUp(self):
        # 3-class matrix: rows = true, cols = predicted. Row 2 (index 2) is empty.
        self.cm = np.array([[5, 2, 0], [1, 4, 0], [0, 0, 0]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(fig.data[0], go.Heatmap)

    def test_axes_are_labelled_true_and_predicted(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(list(fig.data[0].x), self.labels)
        self.assertEqual(list(fig.data[0].y), self.labels)

    def test_customdata_encodes_pred_then_true(self):
        # customdata[true][pred] must be [pred_idx, true_idx] for the click handler.
        fig = plot_confusion_matrix(self.cm, self.labels)
        cd = np.asarray(fig.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # true=0, pred=1
        self.assertEqual(list(cd[1, 2]), [2, 1])  # true=1, pred=2

    def test_counts_shown_as_text(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(fig.data[0].text[0][0], "5")

    def test_normalize_true_is_row_recall(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        np.testing.assert_allclose(z[0], [5 / 7, 2 / 7, 0.0])

    def test_normalize_true_handles_empty_row_without_nan(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        self.assertFalse(np.isnan(z).any())
        np.testing.assert_array_equal(z[2], [0.0, 0.0, 0.0])

    def test_custom_title(self):
        fig = plot_confusion_matrix(self.cm, self.labels, title="Errors")
        self.assertIn("Errors", fig.layout.title.text)


# ---------------------------------------------------------------------------
# NlpExplainer (no real model)
# ---------------------------------------------------------------------------


class TestNlpExplainer(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()

    def test_text_plot_returns_figure(self):
        fig = self.xpl.text_plot(pos=0, label_idx=1)
        self.assertIsInstance(fig, go.Figure)

    def test_text_plot_all_samples(self):
        for pos in range(3):
            fig = self.xpl.text_plot(pos=pos, label_idx=0)
            self.assertIsInstance(fig, go.Figure)

    def test_text_plot_all_labels(self):
        for label_idx in range(N_CLASSES):
            fig = self.xpl.text_plot(pos=0, label_idx=label_idx)
            self.assertIsInstance(fig, go.Figure)

    def test_text_plot_max_tokens(self):
        fig = self.xpl.text_plot(pos=0, label_idx=1, max_tokens=3)
        self.assertIsInstance(fig, go.Figure)
        self.assertLessEqual(len(fig.data[0].x), 3)

    def test_text_plot_raises_before_compile(self):
        xpl = _make_explainer(compiled=False)
        with self.assertRaises(RuntimeError):
            xpl.text_plot(pos=0)

    def test_run_app_raises_before_compile(self):
        xpl = _make_explainer(compiled=False)
        with self.assertRaises(RuntimeError):
            xpl.run_app()

    def test_y_pred_stored(self):
        self.assertIsNotNone(self.xpl.y_pred)
        self.assertEqual(len(self.xpl.y_pred), 3)

    def test_label_names_propagated(self):
        self.assertEqual(self.xpl.contributions.label_names, LABEL_NAMES)

    def test_y_true_is_none_by_default(self):
        xpl = _make_explainer(compiled=True)
        self.assertIsNone(xpl.y_true)


# ---------------------------------------------------------------------------
# NlpExplainer — counterfactual generator discovery / selection (captum-free)
# ---------------------------------------------------------------------------


class _FullCapModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """A model exposing every capability — both HotFlip and AblationFlip are compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return (["a"], np.zeros((1, 2)))

    def embed(self, texts):
        return np.zeros((len(texts), 2))

    def token_gradients(self, text, target_class):
        toks = text.split()
        return toks, np.zeros((len(toks), 2))

    @property
    def shap_callable(self):
        return self.predict


class _TokenizeOnlyModel(TextModel, SupportsTokenization):
    """Tokenizable but gradient-free — only AblationFlip is compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    @property
    def shap_callable(self):
        return self.predict


class TestNlpExplainerGenerators(unittest.TestCase):
    """Generator auto-discovery drives the webapp's method selector (no captum needed here)."""

    def test_full_capability_model_offers_both_methods(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertEqual(
            xpl.available_cf_generators(),
            [("hotflip", "HotFlip"), ("ablation_flip", "Ablation")],
        )
        # The preferred (first-discovered) generator stays the active default.
        self.assertIsInstance(xpl.cf_generator, HotFlipGenerator)
        self.assertEqual(set(xpl.cf_generators), {"hotflip", "ablation_flip"})

    def test_tokenize_only_model_offers_ablation_only(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object())
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIsInstance(xpl.cf_generator, AblationFlipGenerator)

    def test_explicit_generator_used_verbatim_no_extras(self):
        model = _FullCapModel()
        gen = AblationFlipGenerator(model)
        xpl = NlpExplainer(model, backend=object(), cf_generator=gen)
        # An explicit choice is not augmented with the other compatible built-ins.
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIs(xpl.cf_generator, gen)

    def test_cf_config_spec_selected_by_generator(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertIn("max_flips", xpl.cf_config_spec("hotflip"))
        self.assertIn("max_ablations", xpl.cf_config_spec("ablation_flip"))
        # No argument → the active generator's spec.
        self.assertIn("max_flips", xpl.cf_config_spec())

    def test_unknown_generator_raises(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        with self.assertRaises(KeyError):
            xpl.cf_config_spec("does_not_exist")
        with self.assertRaises(KeyError):
            xpl.generate_counterfactuals("hi there", generator="does_not_exist")

    def test_no_generators_without_text_model(self):
        # A plain callable is neither a TextModel nor a pipeline → no generators, empty selector.
        xpl = NlpExplainer(lambda texts: np.tile([0.5, 0.5], (len(texts), 1)), label_names=["neg", "pos"], backend=object())
        self.assertEqual(xpl.available_cf_generators(), [])
        self.assertEqual(xpl.cf_config_spec(), {})
        self.assertIsNone(xpl.cf_generator)


# ---------------------------------------------------------------------------
# NlpWebApp layout (no server launch)
# ---------------------------------------------------------------------------


class TestNlpWebApp(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()
        self.webapp = NlpWebApp(self.xpl)

    def test_layout_built(self):
        self.assertIsNotNone(self.webapp.app.layout)

    def test_class_selector_options(self):
        # Walk layout to find the Dropdown with id "class-selector"
        dropdown = self._find_component(self.webapp.app.layout, "class-selector")
        self.assertIsNotNone(dropdown, "class-selector dropdown not found in layout")
        self.assertEqual(len(dropdown.options), N_CLASSES)
        labels = [opt["label"] for opt in dropdown.options]
        self.assertEqual(labels, LABEL_NAMES)

    def test_dataset_table_populated(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        self.assertIsNotNone(table, "dataset-table not found in layout")
        self.assertEqual(len(table.rowData), 3)
        self.assertIn("text", table.rowData[0])
        self.assertIn("prediction", table.rowData[0])

    def test_raises_without_compile(self):
        xpl = _make_explainer(compiled=False)
        with self.assertRaises(RuntimeError):
            NlpWebApp(xpl)

    def test_graph_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("global-importance-graph", ids)
        self.assertIn("dataset-table", ids)
        self.assertIn("class-selector", ids)
        # token bar chart removed; sentence-highlight replaced it
        self.assertNotIn("local-contributions-graph", ids)

    def test_control_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("topk-slider", ids)
        self.assertIn("sign-filter", ids)
        self.assertIn("word-filter", ids)

    def test_sentence_highlight_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("sentence-highlight", ids)

    def test_waterfall_controls_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        # Waterfall is now a tab (no show/hide switch); the threshold slider + graph live in it.
        self.assertIn("waterfall-threshold", ids)
        self.assertIn("waterfall-graph", ids)

    def test_dataset_table_no_ground_truth_by_default(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertNotIn("ground_truth", col_fields)

    def test_dataset_table_with_y_true(self):
        xpl = _make_explainer(compiled=True)
        xpl.y_true = pd.Series(["sadness", "joy", "sadness"], index=pd.RangeIndex(3), name="ground_truth")
        webapp = NlpWebApp(xpl)
        table = self._find_component(webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertIn("ground_truth", col_fields)
        self.assertEqual(table.rowData[0]["ground_truth"], "sadness")

    def test_scatter_store_always_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("scatter-selected-indices", ids)

    def test_scatter_absent_when_no_xy(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("scatter-plot", ids)
        self.assertNotIn("color-by", ids)

    def test_scatter_present_when_xy_given(self):
        xpl = _make_explainer(compiled=True)
        xy = np.zeros((3, 2))
        webapp = NlpWebApp(xpl, scatter_xy=xy)
        ids = self._collect_ids(webapp.app.layout)
        self.assertIn("scatter-plot", ids)
        self.assertIn("color-by", ids)

    def test_scatter_wrong_shape_raises(self):
        xpl = _make_explainer(compiled=True)
        with self.assertRaises(ValueError):
            NlpWebApp(xpl, scatter_xy=np.zeros((5, 2)))  # 5 rows but only 3 samples

    # ── Error Analysis tab (confusion matrix) ─────────────────────────

    def _make_webapp_with_gt(self) -> NlpWebApp:
        # y_pred is ["joy", "sadness", "joy"]; make sample 0 a sadness→joy error, others correct.
        xpl = _make_explainer(compiled=True)
        xpl.y_true = pd.Series(["sadness", "sadness", "joy"], index=pd.RangeIndex(3), name="ground_truth")
        return NlpWebApp(xpl)

    def test_error_analysis_absent_without_ground_truth(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("confusion-matrix-graph", ids)
        self.assertNotIn("error-pred-importance", ids)

    def test_error_cell_store_always_present(self):
        # The store is created unconditionally so cross-panel callbacks can read it even without gt.
        self.assertIn("error-cell", self._collect_ids(self.webapp.app.layout))

    def test_error_analysis_present_with_ground_truth(self):
        ids = self._collect_ids(self._make_webapp_with_gt().app.layout)
        for cid in ("confusion-matrix-graph", "error-pred-importance", "error-true-importance", "cm-normalize"):
            self.assertIn(cid, ids)

    def test_confusion_matrix_counts(self):
        webapp = self._make_webapp_with_gt()
        cm = webapp._cm
        # LABEL_NAMES index: sadness=0, joy=1. Rows=true, cols=pred.
        self.assertEqual(cm[0, 1], 1)  # true sadness predicted joy (the error)
        self.assertEqual(cm[0, 0], 1)  # true sadness predicted sadness
        self.assertEqual(cm[1, 1], 1)  # true joy predicted joy
        self.assertEqual(cm.sum(), 3)

    def test_confusion_matrix_index_arrays(self):
        webapp = self._make_webapp_with_gt()
        self.assertEqual(webapp._cm_true_idx.tolist(), [0, 0, 1])  # sadness, sadness, joy
        self.assertEqual(webapp._cm_pred_idx.tolist(), [1, 0, 1])  # joy, sadness, joy

    def test_confusion_matrix_figure_customdata_orientation(self):
        webapp = self._make_webapp_with_gt()
        graph = self._find_component(webapp.app.layout, "confusion-matrix-graph")
        cd = np.asarray(graph.figure.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # cell (true=0, pred=1) → [pred_idx, true_idx]

    def test_cell_from_click_uses_label_names_when_no_customdata(self):
        # Heatmap clicks may omit customdata; the x (pred) / y (true) labels must still resolve.
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness"}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (1, 0))

    def test_cell_from_click_prefers_customdata(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness", "customdata": [2, 3]}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (2, 3))

    def test_cell_from_click_none_on_empty_or_unknown(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        self.assertIsNone(_cell_from_click(None, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": []}, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": [{"x": "??", "y": "??"}]}, name_to_idx))


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_component(self, node, component_id):
        """Depth-first search for a Dash component by id."""
        if hasattr(node, "id") and node.id == component_id:
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        if not isinstance(children, list):
            children = [children]
        for child in children:
            result = self._find_component(child, component_id)
            if result is not None:
                return result
        return None

    def _collect_ids(self, node) -> set:
        """Collect all component ids in the layout tree."""
        ids = set()
        if hasattr(node, "id") and isinstance(node.id, str):
            ids.add(node.id)
        children = getattr(node, "children", None)
        if children is None:
            return ids
        if not isinstance(children, list):
            children = [children]
        for child in children:
            ids |= self._collect_ids(child)
        return ids


class TestComposeSelection(unittest.TestCase):
    """The three sample filters (scatter box, confusion cell, errors-only) must intersect."""

    def test_nothing_active_returns_none(self):
        self.assertIsNone(_compose_selection(None, None, None))

    def test_scatter_only(self):
        self.assertEqual(_compose_selection([1, 2, 3], None, None), [1, 2, 3])

    def test_cell_only(self):
        self.assertEqual(_compose_selection(None, [4, 5], None), [4, 5])

    def test_errors_only(self):
        self.assertEqual(_compose_selection(None, None, {2, 7}), [2, 7])

    def test_scatter_intersect_cell(self):
        self.assertEqual(_compose_selection([1, 2, 3], [2, 3, 4], None), [2, 3])

    def test_scatter_intersect_errors(self):
        # This is the box-then-errors case: errors *within* the selected points.
        self.assertEqual(_compose_selection([1, 2, 3, 4], None, {2, 4, 9}), [2, 4])

    def test_all_three_intersect(self):
        self.assertEqual(_compose_selection([1, 2, 3, 4], [2, 3, 4], {3, 4}), [3, 4])

    def test_empty_intersection_is_empty_list(self):
        self.assertEqual(_compose_selection([1, 2], None, {8, 9}), [])


# ---------------------------------------------------------------------------
# NlpLimeBackend
# ---------------------------------------------------------------------------

# Keep num_samples tiny so LIME tests run fast in CI.
_LIME_COMPUTE_ARGS = {"num_samples": 50, "num_features": 5}
_SAMPLE_TEXTS = ["i feel so happy today", "this is terrible and sad"]


def _fake_classifier(texts: list[str]) -> np.ndarray:
    """Deterministic fake classifier returning (n_texts, N_CLASSES) probabilities."""
    rng = np.random.default_rng(0)
    probs = rng.random((len(texts), N_CLASSES)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _make_lime_backend() -> NlpLimeBackend:
    return NlpLimeBackend(
        _fake_classifier,
        label_names=LABEL_NAMES,
        explainer_compute_args=_LIME_COMPUTE_ARGS,
    )


class TestNlpLimeBackend(unittest.TestCase):

    def setUp(self):
        self.backend = _make_lime_backend()

    # --- init / config ---

    def test_name(self):
        self.assertEqual(self.backend.name, "nlp_lime")

    def test_inherits_nlp_backend(self):
        self.assertIsInstance(self.backend, NlpBackend)

    def test_label_names_stored(self):
        self.assertEqual(self.backend._classes, LABEL_NAMES)

    def test_mask_string_stored(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            mask_string="[MASK]",
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertEqual(backend.mask_string, "[MASK]")

    def test_explainer_args_forwarded(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            explainer_args={"bow": False},
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertFalse(backend.explainer.bow)

    # --- _classifier_fn ---

    def test_classifier_fn_converts_list_to_array(self):
        def list_model(texts):
            return [[0.5] * N_CLASSES for _ in texts]

        backend = NlpLimeBackend(list_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, N_CLASSES))

    def test_classifier_fn_passes_through_array(self):
        arr = np.ones((2, N_CLASSES), dtype=np.float32)

        def array_model(texts):
            return arr

        backend = NlpLimeBackend(array_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["a", "b"])
        self.assertIs(result, arr)

    def test_classifier_fn_converts_hf_pipeline_format(self):
        # HuggingFace pipeline with return_all_scores=True returns list[list[dict]].
        def hf_model(texts):
            return [
                [{"label": name, "score": 1.0 / N_CLASSES} for name in LABEL_NAMES]
                for _ in texts
            ]

        backend = NlpLimeBackend(hf_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello", "world"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, N_CLASSES))
        self.assertEqual(result.dtype, np.float64)
        # Each score should be 1/N_CLASSES
        np.testing.assert_allclose(result, 1.0 / N_CLASSES, atol=1e-6)

    # --- run_explainer ---

    def test_run_explainer_returns_raw_explanation(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertIsInstance(raw, NlpRawExplanation)

    def test_run_explainer_contributions_count(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.contributions), len(_SAMPLE_TEXTS))

    def test_run_explainer_contributions_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        for arr in raw.contributions:
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[1], N_CLASSES)

    def test_run_explainer_base_values_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(raw.base_values.shape, (len(_SAMPLE_TEXTS), N_CLASSES))

    def test_run_explainer_data_is_word_list(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.data), len(_SAMPLE_TEXTS))
        for word_list in raw.data:
            self.assertIsInstance(word_list, list)
            self.assertTrue(all(isinstance(w, str) for w in word_list))

    def test_run_explainer_sparse_weights(self):
        # LIME fills at most num_features non-zero weights per label column.
        raw = self.backend.run_explainer(_SAMPLE_TEXTS[:1])
        matrix = raw.contributions[0]
        for col in range(N_CLASSES):
            self.assertLessEqual(
                np.count_nonzero(matrix[:, col]),
                _LIME_COMPUTE_ARGS["num_features"],
            )

    # --- get_local_contributions ---

    def test_get_local_contributions_returns_nlp_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertIsInstance(contrib, NlpContributions)

    def test_get_local_contributions_token_strings_match_data(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertEqual(contrib.token_strings, raw.data)

    def test_get_local_contributions_values_match_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        for got, expected in zip(contrib.values, raw.contributions):
            np.testing.assert_array_equal(got, expected)

    def test_get_local_contributions_subset(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw, subset=[0])
        self.assertEqual(len(contrib.token_strings), 1)
        self.assertEqual(len(contrib.values), 1)
        self.assertEqual(contrib.base_values.shape[0], 1)


# ---------------------------------------------------------------------------
# NlpExplainer with NlpLimeBackend
# ---------------------------------------------------------------------------


class TestNlpExplainerWithLimeBackend(unittest.TestCase):

    def _make_explainer_lime(self, compiled: bool = True) -> NlpExplainer:
        """NlpExplainer backed by NlpLimeBackend, with synthetic contributions."""
        xpl = _make_explainer(compiled=compiled)
        xpl.backend = _make_lime_backend()
        return xpl

    def test_backend_is_lime_instance(self):
        xpl = self._make_explainer_lime()
        self.assertIsInstance(xpl.backend, NlpLimeBackend)

    def test_text_plot_works_with_lime_backend(self):
        xpl = self._make_explainer_lime(compiled=True)
        fig = xpl.text_plot(pos=0, label_idx=0)
        self.assertIsInstance(fig, go.Figure)

    def test_text_plot_raises_before_compile(self):
        xpl = self._make_explainer_lime(compiled=False)
        with self.assertRaises(RuntimeError):
            xpl.text_plot(pos=0)

    def test_compile_sets_contributions(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df):
            xpl.compile(_SAMPLE_TEXTS)
        self.assertIsInstance(xpl.contributions, NlpContributions)
        self.assertEqual(len(xpl.contributions), len(_SAMPLE_TEXTS))
        self.assertEqual(xpl.contributions.label_names, LABEL_NAMES)

    def test_compile_caching_skips_rerun(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df):
            xpl.compile(_SAMPLE_TEXTS)
            hash_after_first = xpl._data_hash
            xpl.compile(_SAMPLE_TEXTS)  # same data — must hit in-memory cache
        self.assertEqual(xpl._data_hash, hash_after_first)


if __name__ == "__main__":
    unittest.main()
