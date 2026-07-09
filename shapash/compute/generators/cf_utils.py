"""Counterfactual utility functions (ports of LIT ``cf_utils``).

Small, pure helpers shared by generators: detecting a prediction flip and measuring how far a
counterfactual moved the original class probability.
"""

from __future__ import annotations

import numpy as np


def is_prediction_flip(orig_probs: np.ndarray, cf_probs: np.ndarray, target_class: int | None = None) -> bool:
    """Return ``True`` when the counterfactual changes the predicted class.

    Parameters
    ----------
    orig_probs : np.ndarray, shape (n_classes,)
        Class probabilities for the original text.
    cf_probs : np.ndarray, shape (n_classes,)
        Class probabilities for the counterfactual text.
    target_class : int or None
        When given, success requires the argmax to become exactly ``target_class``; otherwise any
        change of argmax counts.

    Returns
    -------
    bool
        Whether the prediction flipped (towards ``target_class`` if specified).
    """
    orig_class = int(np.argmax(orig_probs))
    cf_class = int(np.argmax(cf_probs))
    if target_class is not None:
        return cf_class == target_class and cf_class != orig_class
    return cf_class != orig_class


def prediction_difference(orig_probs: np.ndarray, cf_probs: np.ndarray, class_idx: int) -> float:
    """Return how much ``class_idx``'s probability dropped from original to counterfactual."""
    return float(orig_probs[class_idx] - cf_probs[class_idx])


def is_word_token(token: str) -> bool:
    """True for plausible standalone word tokens.

    Rejects sub-word continuations (``##ing``), special/placeholder tokens (``[CLS]``), punctuation
    and numerics — substituting or removing any of these yields malformed text once detokenized.
    ``str.isalpha`` handles all of these in one check (``##ing``/``[CLS]``/``1b`` are not alphabetic).
    Shared by the token-substitution (HotFlip) and token-removal (ablation) generators.
    """
    return token.isalpha()
