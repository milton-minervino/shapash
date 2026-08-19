"""Unit tests for the shared NLP backend layer — the punctuation helper.

``NlpContributions.word_importance`` filtering is covered in ``tests/unit_tests/nlp``; this file
covers :func:`~shapash.backend.nlp_backend.is_punctuation` itself, which word segmentation now
makes load-bearing (punctuation is emitted as its own unit rather than glued to a neighbour).
"""

import unittest

from shapash.backend.nlp_backend import is_punctuation


class TestIsPunctuation(unittest.TestCase):
    def test_pure_punctuation_units(self):
        for unit in [".", ",", "!", "!!!", "?!", "--", "\u2026", "(", ")", "\u201c", "/"]:
            with self.subTest(unit=unit):
                self.assertTrue(is_punctuation(unit))

    def test_words_are_not_punctuation(self):
        for unit in ["great", "don't", "state-of-the-art", "10/10", "u.s.", "_", "3"]:
            with self.subTest(unit=unit):
                self.assertFalse(is_punctuation(unit))

    def test_blank_is_not_punctuation(self):
        # Blanks are the special-token concern, handled by ``filter_special`` — not this helper.
        for unit in ["", "   ", "\n"]:
            with self.subTest(unit=unit):
                self.assertFalse(is_punctuation(unit))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertTrue(is_punctuation("  !  "))


if __name__ == "__main__":
    unittest.main()
