"""Local API token: creation, perms, idempotency, matching, extraction."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from myagent import auth


class TestToken(unittest.TestCase):
    def test_creates_token_with_0600_perms(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data"
            tok = auth.load_or_create_token(d)
            self.assertTrue(tok)
            self.assertGreaterEqual(len(tok), 32)
            mode = stat.S_IMODE(os.stat(auth.token_path(d)).st_mode)
            self.assertEqual(mode, 0o600, oct(mode))

    def test_idempotent_reuses_existing_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            first = auth.load_or_create_token(d)
            second = auth.load_or_create_token(d)
            self.assertEqual(first, second)

    def test_reharden_loosens_perms_back_to_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            auth.load_or_create_token(d)
            os.chmod(auth.token_path(d), 0o644)  # simulate a loosely-created file
            auth.load_or_create_token(d)          # should re-harden
            mode = stat.S_IMODE(os.stat(auth.token_path(d)).st_mode)
            self.assertEqual(mode, 0o600)

    def test_read_token_does_not_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(auth.read_token(d))
            self.assertFalse(auth.token_path(d).exists())

    def test_matches_is_constant_time_and_fails_closed(self):
        self.assertTrue(auth.token_matches("abc", "abc"))
        self.assertFalse(auth.token_matches("abc", "abd"))
        self.assertFalse(auth.token_matches("", "abc"))
        self.assertFalse(auth.token_matches("abc", None))
        self.assertFalse(auth.token_matches(None, "abc"))

    def test_extract_prefers_header_then_query_ignores_authorization(self):
        self.assertEqual(
            auth.extract_token({auth.HEADER: "h"}, {"token": "q"}), "h")
        self.assertEqual(auth.extract_token({}, {"token": "q"}), "q")
        # Authorization is reserved for the proxy's upstream key — never our token.
        self.assertIsNone(auth.extract_token({"authorization": "Bearer x"}, {}))


if __name__ == "__main__":
    unittest.main()
