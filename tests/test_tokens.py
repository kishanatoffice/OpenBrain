"""Token estimator heuristics + the session savings counter."""

from __future__ import annotations

import unittest

from myagent import tokens


class TokensCase(unittest.TestCase):
    def test_text_estimate_scales_with_length(self):
        self.assertEqual(tokens.estimate_text_tokens(""), 0)
        self.assertGreaterEqual(tokens.estimate_text_tokens("a" * 400), 100)
        self.assertGreater(
            tokens.estimate_text_tokens("a" * 800),
            tokens.estimate_text_tokens("a" * 400),
        )

    def test_image_estimate_has_floor_and_grows(self):
        self.assertEqual(tokens.estimate_image_vision_tokens(0, 0),
                         tokens._MIN_VISION_TOKENS)
        small = tokens.estimate_image_vision_tokens(100, 100)
        big = tokens.estimate_image_vision_tokens(2000, 2000)
        self.assertGreaterEqual(small, tokens._MIN_VISION_TOKENS)
        self.assertGreater(big, small)

    def test_record_savings_accumulates_and_ignores_nonpositive(self):
        before = tokens.tokens_saved()
        tokens.record_savings(500)
        tokens.record_savings(0)
        tokens.record_savings(-100)
        self.assertEqual(tokens.tokens_saved(), before + 500)


if __name__ == "__main__":
    unittest.main()
