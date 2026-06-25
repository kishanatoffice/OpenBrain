"""Claude Code capture hook: payload → approval-event translation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guardrails.hooks.claude_code_capture import (
    _result_from_response,
    _summarize_input,
    build_event,
    last_user_request,
)


class TestSummarize(unittest.TestCase):
    def test_bash_uses_command(self):
        self.assertEqual(_summarize_input("Bash", {"command": "npm start"}),
                         "Bash: npm start")

    def test_edit_uses_file_path(self):
        self.assertEqual(_summarize_input("Edit", {"file_path": "/a/b.py"}),
                         "Edit: /a/b.py")

    def test_falls_back_to_json(self):
        out = _summarize_input("Weird", {"foo": "bar"})
        self.assertTrue(out.startswith("Weird: "))

    def test_no_input(self):
        self.assertEqual(_summarize_input("Read", {}), "Read")


class TestResultFromResponse(unittest.TestCase):
    def test_dict_error_is_failure(self):
        r, _ = _result_from_response({"error": "boom"})
        self.assertEqual(r, "failure")

    def test_dict_ok_is_success(self):
        r, _ = _result_from_response({"stdout": "ok"})
        self.assertEqual(r, "success")

    def test_string_error_is_failure(self):
        r, _ = _result_from_response("Traceback (most recent call last)")
        self.assertEqual(r, "failure")

    def test_string_ok_is_success(self):
        r, _ = _result_from_response("done")
        self.assertEqual(r, "success")


class TestBuildEvent(unittest.TestCase):
    def test_post_tool_use_is_complete_and_allowed(self):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "systemctl restart app"},
            "tool_response": {"stdout": "OK"},
            "session_id": "s1", "cwd": "/repo",
        }
        ev = build_event(payload, repository="open_brain", branch="main",
                        user_request="Run the application")
        self.assertEqual(ev["ide"], "claude-code")
        self.assertEqual(ev["selected_option"], "allow")
        self.assertEqual(ev["result"], "success")
        self.assertEqual(ev["options"], ["allow", "deny"])
        self.assertEqual(ev["agent_action"], "Bash: systemctl restart app")
        self.assertEqual(ev["user_request"], "Run the application")
        self.assertEqual(ev["repository"], "open_brain")
        self.assertEqual(ev["metadata"]["prompt_source"], "synthesized")
        self.assertIn("systemctl restart app", ev["prompt_text"])

    def test_pre_tool_use_has_no_decision(self):
        ev = build_event({"hook_event_name": "PreToolUse", "tool_name": "Edit",
                          "tool_input": {"file_path": "x.py"}})
        self.assertNotIn("selected_option", ev)
        self.assertNotIn("result", ev)


class TestLastUserRequest(unittest.TestCase):
    def _transcript(self, entries):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            tmp.write(json.dumps(e) + "\n")
        tmp.close()
        return tmp.name

    def test_picks_last_user_string(self):
        path = self._transcript([
            {"type": "user", "message": {"role": "user", "content": "first"}},
            {"type": "assistant", "message": {"role": "assistant", "content": "ok"}},
            {"type": "user", "message": {"role": "user", "content": "second request"}},
        ])
        self.assertEqual(last_user_request(path), "second request")

    def test_handles_parts_content(self):
        path = self._transcript([
            {"type": "user", "message": {"role": "user",
             "content": [{"type": "text", "text": "please deploy"}]}},
        ])
        self.assertEqual(last_user_request(path), "please deploy")

    def test_missing_file_is_none(self):
        self.assertIsNone(last_user_request("/no/such/file.jsonl"))

    def test_none_path(self):
        self.assertIsNone(last_user_request(None))


if __name__ == "__main__":
    unittest.main()
