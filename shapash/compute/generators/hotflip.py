"""HotFlip counterfactual generator — a port of Google PAIR LIT's ``hotflip.py``.

HotFlip finds a *minimal* set of token substitutions that changes the model's prediction, using a
single backward pass to estimate each token's impact:

1. Rank content tokens by the L2 norm of their input-embedding gradient (largest first).
2. For each position, shortlist the top-K **word** candidates by the first-order estimate
   (``embedding_matrix @ token_gradient``, most-negative first), then **re-score that shortlist
   against the model** and keep the one that most reduces the original-class probability. The linear
   estimate alone is an unreliable proxy — its single best pick often does not flip at all — so
   model verification is what surfaces genuine substitutions. Sub-word (``##``) and non-word tokens
   are excluded so rebuilt text stays well-formed.
3. Try flip combinations of increasing size (1..``max_flips``), rebuild the text, re-predict, and
   keep those that flip the prediction — enforcing **minimality** (never return a combination that
   is a superset of an already-successful one).

Requires a model exposing :class:`~shapash.model.base.SupportsGradients` and
:class:`~shapash.model.base.SupportsEmbeddings`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol, cast

import numpy as np

from shapash.compute.generators.base import (
    Counterfactual,
    CounterfactualGenerator,
    Field,
    IntField,
    TokenListField,
)
from shapash.compute.generators.cf_utils import is_prediction_flip, is_word_token
from shapash.model.base import SupportsEmbeddings, SupportsGradients, TextModel, has_capabilities

_MAX_FLIPPABLE_TOKENS = 10
_CANDIDATES_PER_POSITION = 50


class _GradientModel(Protocol):
    """The exact capability surface HotFlip uses (predict + gradients + embeddings + detokenize)."""

    label_names: list[str] | None

    def predict(self, texts: list[str]) -> np.ndarray: ...

    def token_gradients(self, text: str, target_class: int) -> tuple[list[str], np.ndarray]: ...

    def get_embedding_table(self) -> tuple[list[str], np.ndarray]: ...

    def detokenize(self, tokens: list[str]) -> str: ...


class HotFlipGenerator(CounterfactualGenerator):
    """Gradient-based token-substitution counterfactuals (LIT HotFlip)."""

    name = "hotflip"

    def config_spec(self) -> dict[str, Field]:
        """Expose ``num_examples``, ``max_flips`` and ``tokens_to_ignore`` as tunable knobs."""
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            "max_flips": IntField(label="Max token flips", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    @classmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Compatible only with models that expose gradients *and* an embedding table."""
        return has_capabilities(model, SupportsGradients, SupportsEmbeddings)

    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate minimal HotFlip counterfactuals for ``text`` (see module docstring)."""
        cfg = self.resolve_config(config)
        num_examples = int(cfg["num_examples"])
        max_flips = int(cfg["max_flips"])
        ignore = {t.strip().lower() for t in cfg["tokens_to_ignore"]}

        model = cast(_GradientModel, self.model)  # is_compatible guarantees these capabilities
        label_names = model.label_names or []

        orig_probs = model.predict([text])[0]
        orig_class = int(np.argmax(orig_probs))
        orig_label = label_names[orig_class] if orig_class < len(label_names) else str(orig_class)
        target_class = label_names.index(target_label) if (target_label and target_label in label_names) else None

        tokens, grads = model.token_gradients(text, orig_class)
        if not tokens:
            return []

        # Rank content tokens by gradient L2 norm; keep the top-K flippable positions.
        grad_l2 = np.sum(grads * grads, axis=-1)
        ranked = np.argsort(-grad_l2).tolist()
        flippable = [p for p in ranked if tokens[p].strip().lower() not in ignore][:_MAX_FLIPPABLE_TOKENS]

        vocab, matrix = model.get_embedding_table()
        replacement: dict[int, str] = {}
        for pos in flippable:
            # Shortlist the top-K word candidates by the first-order estimate (most-negative first)...
            scores = matrix @ grads[pos]
            shortlist: list[str] = []
            for cand in np.argsort(scores):
                cand_tok = vocab[int(cand)]
                if cand_tok != tokens[pos] and is_word_token(cand_tok):
                    shortlist.append(cand_tok)
                    if len(shortlist) >= _CANDIDATES_PER_POSITION:
                        break
            if not shortlist:
                continue
            # ...then re-score the shortlist against the model and keep the one that most reduces the
            # original-class probability (the linear estimate's top pick is an unreliable proxy).
            cand_texts = [model.detokenize([*tokens[:pos], tok, *tokens[pos + 1 :]]) for tok in shortlist]
            cand_orig_probs = model.predict(cand_texts)[:, orig_class]
            replacement[pos] = shortlist[int(np.argmin(cand_orig_probs))]

        flippable = [p for p in flippable if p in replacement]

        results: list[Counterfactual] = []
        successful: list[frozenset[int]] = []
        for size in range(1, max_flips + 1):
            for combo in combinations(flippable, size):
                combo_set = frozenset(combo)
                if any(prev <= combo_set for prev in successful):
                    continue  # minimality: skip supersets of a smaller success
                new_tokens = list(tokens)
                subs: list[tuple[int, str, str]] = []
                for pos in combo:
                    subs.append((pos, tokens[pos], replacement[pos]))
                    new_tokens[pos] = replacement[pos]
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
                            substitutions=subs,
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
