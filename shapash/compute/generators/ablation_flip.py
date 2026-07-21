"""AblationFlip counterfactual generator — forward-pass-only token removal.

The perturbation-based sibling of :class:`~shapash.compute.generators.hotflip.HotFlipGenerator`. Where
HotFlip *substitutes* tokens using embedding gradients, AblationFlip *removes* tokens it scores as most
supportive of the current prediction, until the prediction flips:

1. Score each content token with Captum ``FeatureAblation``: starting from the full text, ablate one
   token at a time (to an empty baseline) and measure the drop in the original class's probability.
   Positive score = the token supports the original class (removing it hurts that class most).
2. Greedily remove the highest-scoring tokens in combinations of increasing size (1..``max_ablations``),
   rebuild the text, re-predict, and keep those that flip the prediction — enforcing **minimality**
   (never return a removal set that is a superset of an already-successful one).

It needs only forward passes (``predict``) plus tokenization, so — unlike HotFlip — it also works on
prediction-only models such as ``HFPipelineModel``. The Captum scoring step (:meth:`_ablation_scores`)
is isolated from the pure greedy-flip loop so the flip/minimality logic is testable without Captum.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from shapash._optional import import_optional_module
from shapash.compute.generators.base import (
    Counterfactual,
    CounterfactualGenerator,
    Field,
    IntField,
    TokenListField,
)
from shapash.compute.generators.cf_utils import display_form, is_prediction_flip
from shapash.model.base import SupportsTokenization, TextModel, has_capabilities

if TYPE_CHECKING:
    from collections.abc import Sequence

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'
_MAX_ABLATABLE_TOKENS = 10


class AblationFlipGenerator(CounterfactualGenerator):
    """Perturbation-based token-removal counterfactuals (Captum ``FeatureAblation``)."""

    name = "ablation_flip"
    display_name = "Ablation"

    def config_spec(self) -> dict[str, Field]:
        """Expose ``num_examples``, ``max_ablations`` and ``tokens_to_ignore`` as tunable knobs."""
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            "max_ablations": IntField(label="Max token removals", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    @classmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Compatible with any tokenizable model — only forward passes and tokenization are needed."""
        return has_capabilities(model, SupportsTokenization)

    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate minimal token-removal counterfactuals for ``text`` (see module docstring)."""
        cfg = self.resolve_config(config)
        num_examples = int(cfg["num_examples"])
        max_ablations = int(cfg["max_ablations"])
        ignore = {t.strip().lower() for t in cfg["tokens_to_ignore"]}

        model = self.model
        if not isinstance(model, SupportsTokenization):  # is_compatible guarantees this; guard direct use
            raise TypeError("AblationFlipGenerator requires a model implementing SupportsTokenization.")
        label_names = model.label_names or []

        orig_probs = model.predict([text])[0]
        orig_class = int(np.argmax(orig_probs))
        orig_label = label_names[orig_class] if orig_class < len(label_names) else str(orig_class)
        target_class = label_names.index(target_label) if (target_label and target_label in label_names) else None

        tokens = model.tokenize(text)
        # Removable positions are decided by the *model's* tokenization scheme (``is_substitutable``),
        # not by a string rule here: under SentencePiece/byte-BPE every content token carries a
        # ``▁``/``Ġ`` word-start marker, which a bare ``isalpha`` check rejects wholesale. ``ignore``
        # holds user-typed words, so it is matched against each token's display form for the same
        # reason — ``"great"`` must match the token ``"▁great"``.
        content_positions = [
            i for i, t in enumerate(tokens) if model.is_substitutable(t) and display_form(model, t) not in ignore
        ]
        if not content_positions:
            return []

        scores = self._ablation_scores(tokens, content_positions, orig_class)
        # Keep the most supportive positions (largest drop when removed) as removal candidates.
        ranked = [content_positions[j] for j in np.argsort(-scores)][:_MAX_ABLATABLE_TOKENS]

        results: list[Counterfactual] = []
        successful: list[frozenset[int]] = []
        for size in range(1, max_ablations + 1):
            for combo in combinations(ranked, size):
                combo_set = frozenset(combo)
                if any(prev <= combo_set for prev in successful):
                    continue  # minimality: skip supersets of a smaller success
                new_tokens = [tok for i, tok in enumerate(tokens) if i not in combo_set]
                new_text = model.detokenize(new_tokens)
                cf_probs = model.predict([new_text])[0]
                if is_prediction_flip(orig_probs, cf_probs, target_class):
                    cf_class = int(np.argmax(cf_probs))
                    results.append(
                        Counterfactual(
                            original_text=text,
                            new_text=new_text,
                            tokens=list(tokens),
                            flipped_positions=list(combo),
                            substitutions=[(pos, tokens[pos], "") for pos in combo],
                            orig_label=orig_label,
                            new_label=label_names[cf_class] if cf_class < len(label_names) else str(cf_class),
                            orig_prob=float(orig_probs[orig_class]),
                            new_prob=float(cf_probs[cf_class]),
                            prob_delta=float(orig_probs[orig_class] - cf_probs[orig_class]),
                        )
                    )
                    successful.append(combo_set)
                    if len(results) >= num_examples:
                        return results
        return results

    def _ablation_scores(self, tokens: list[str], content_positions: Sequence[int], orig_class: int) -> np.ndarray:
        """Score each content token by the drop in ``orig_class`` probability when it is removed.

        Uses Captum ``FeatureAblation`` over a token-presence mask: the input is an all-ones mask (every
        content token present); ablating a feature to the zero baseline drops that token from the
        rebuilt text. Returns one score per entry of ``content_positions`` (aligned by order).
        """
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        captum_attr = import_optional_module("captum.attr", extra=_NLP_EXTRA)

        model = self.model
        if not isinstance(model, SupportsTokenization):  # is_compatible guarantees this; guard direct use
            raise TypeError("AblationFlipGenerator requires a model implementing SupportsTokenization.")
        pos_index = {p: j for j, p in enumerate(content_positions)}

        def forward(mask):  # mask: (batch, n_content) float tensor of 0/1
            active = mask.detach().cpu().numpy() > 0.5
            texts = [
                model.detokenize([tok for i, tok in enumerate(tokens) if i not in pos_index or row[pos_index[i]]])
                for row in active
            ]
            return torch.tensor(model.predict(texts), dtype=torch.float32)

        ablator = captum_attr.FeatureAblation(forward)
        inputs = torch.ones((1, len(content_positions)), dtype=torch.float32)
        attributions = ablator.attribute(inputs, baselines=0.0, target=orig_class)
        return attributions.squeeze(0).detach().cpu().numpy()
