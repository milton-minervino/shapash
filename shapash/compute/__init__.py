"""Compute layer — heavy computation behind modality-agnostic interfaces.

Seeded by the NLP prototype ahead of the master refactoring. Today it holds the ``generators``
sub-package (counterfactual / what-if generation), the ``retrieval`` sub-package (similar-example
lookup), and :mod:`~shapash.compute.embedding_store` — the shared cache both retrieval and the 2-D
scatter draw their vectors from. The master plan's ``analyses/{local,global,diagnostics}``
interpreters will move here in later phases; ``generators`` is a *new* sibling axis for generative
components (LIT's ``Generator``, as opposed to explanation ``Interpreter``s).
"""
