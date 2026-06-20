"""Secret/PII redaction tests."""

from __future__ import annotations

import unittest

from myagent.redact import has_secret, redact, redaction_count


class TestRedact(unittest.TestCase):
    def test_openai_key(self):
        clean, found = redact("my key is sk-proj-ABCDEFGHIJKLMNOP1234567890 ok")
        self.assertIn("openai-key", found)
        self.assertNotIn("sk-proj-ABCDEFGHIJKLMNOP", clean)
        self.assertIn("[REDACTED:openai-key]", clean)

    def test_github_and_aws_and_jwt(self):
        self.assertTrue(has_secret("ghp_" + "a" * 36))
        self.assertTrue(has_secret("AKIAIOSFODNN7EXAMPLE"))
        self.assertTrue(has_secret("token eyJhbGciOiJIUzI1Niced.eyJzdWIiOiIxMjM.abcDEFghi"))

    def test_email_is_pii(self):
        clean, found = redact("ping me at jane.doe@example.com please")
        self.assertIn("email", found)
        self.assertNotIn("jane.doe@example.com", clean)

    def test_assigned_secret_keeps_key_name(self):
        clean, found = redact('password = "hunter2supersecret"')
        self.assertIn("assigned-secret", found)
        self.assertIn("password", clean)               # key name preserved
        self.assertNotIn("hunter2supersecret", clean)  # value gone

    def test_clean_text_untouched(self):
        text = "I prefer concise, production-ready answers and use Python + FastAPI."
        clean, found = redact(text)
        self.assertEqual(clean, text)
        self.assertEqual(found, [])

    def test_empty(self):
        self.assertEqual(redact(""), ("", []))

    def test_redaction_count_increments(self):
        before = redaction_count()
        redact("key sk-proj-ABCDEFGHIJKLMNOP1234567890 and a@b.com")
        self.assertGreaterEqual(redaction_count() - before, 2)


if __name__ == "__main__":
    unittest.main()
