"""Counterfactual generator interface + declarative config spec.

Design mirrors both LIT's ``Generator`` (``generate`` + ``config_spec`` + compatibility check) and
the shapash ``NlpBackend`` ABC style. The tiny :class:`Field` spec types let a generator declare its
tunable parameters once; the webapp renders controls from that declaration automatically, so adding
a new knob or a new generator needs no UI changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shapash.model.base import TextModel


@dataclass(frozen=True)
class Counterfactual:
    """One generated counterfactual for a single input text.

    Attributes
    ----------
    original_text : str
        The input text the counterfactual was derived from.
    new_text : str
        The perturbed text (reconstructed from the modified tokens).
    tokens : list[str]
        The original content tokens the generator operated on.
    flipped_positions : list[int]
        Positions (into ``tokens``) that were modified.
    substitutions : list[tuple[int, str, str]]
        ``(position, old_token, new_token)`` for each modification.
    orig_label : str
        Predicted label of ``original_text``.
    new_label : str
        Predicted label of ``new_text`` (differs from ``orig_label`` on a successful flip).
    orig_prob : float
        Probability of ``orig_label`` on the original text.
    new_prob : float
        Probability of ``new_label`` on the counterfactual text.
    prob_delta : float
        Drop in the original label's probability (``orig_prob`` minus its probability on the CF).
    """

    original_text: str
    new_text: str
    tokens: list[str]
    flipped_positions: list[int]
    substitutions: list[tuple[int, str, str]]
    orig_label: str
    new_label: str
    orig_prob: float
    new_prob: float
    prob_delta: float


@dataclass(frozen=True)
class Field:
    """Base declarative spec for a configurable generator parameter."""

    label: str
    default: Any


@dataclass(frozen=True)
class IntField(Field):
    """Integer parameter rendered as a bounded numeric control."""

    minimum: int = 1
    maximum: int = 10


@dataclass(frozen=True)
class TokenListField(Field):
    """List-of-strings parameter (e.g. tokens to ignore) rendered as a tag/multi-select input."""

    default: list[str] = field(default_factory=list)


class CounterfactualGenerator(ABC):
    """Abstract base for counterfactual generators.

    Concrete generators implement :meth:`config_spec`, :meth:`is_compatible` and :meth:`generate`.
    They depend on a :class:`~shapash.model.base.TextModel` through its declared capabilities, never
    on a concrete model class.
    """

    name: str = "counterfactual"

    def __init__(self, model: TextModel) -> None:
        self.model = model

    @abstractmethod
    def config_spec(self) -> dict[str, Field]:
        """Return the tunable parameters as ``name -> Field`` (drives the webapp controls)."""

    @classmethod
    @abstractmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Return ``True`` when ``model`` exposes the capabilities this generator needs."""

    @abstractmethod
    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate counterfactuals for ``text``.

        Parameters
        ----------
        text : str
            Input text to perturb.
        target_label : str or None
            Desired label to flip *towards*. When ``None``, any label change counts as success.
        config : dict or None
            Overrides for :meth:`config_spec` defaults (keys match the spec).

        Returns
        -------
        list[Counterfactual]
            Successful counterfactuals, minimal and ordered by increasing number of flips.
        """

    def resolve_config(self, config: dict | None) -> dict:
        """Merge user ``config`` over the spec defaults into a plain value dict."""
        resolved = {name: fld.default for name, fld in self.config_spec().items()}
        if config:
            resolved.update({k: v for k, v in config.items() if k in resolved})
        return resolved
