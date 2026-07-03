"""Compute layer — heavy computation behind modality-agnostic interfaces.

Seeded by the NLP prototype ahead of the master refactoring. Today it holds only the
``generators`` sub-package (counterfactual / what-if generation). The master plan's
``analyses/{local,global,diagnostics}`` interpreters will move here in later phases;
``generators`` is a *new* sibling axis for generative components (LIT's ``Generator``,
as opposed to explanation ``Interpreter``s).
"""
