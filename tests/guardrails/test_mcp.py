"""Guard Rails MCP surface: initialize, tools/list, tools/call, notifications."""

from __future__ import annotations

import unittest

from guardrails.mcp import handle_mcp


def _ingest_ok(args):
    return {"id": 7, "status": "completed"}


class TestMcp(unittest.TestCase):
    def test_initialize_echoes_supported_version(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-03-26"}}, _ingest_ok)
        self.assertEqual(resp["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_initialize_falls_back_on_unknown_version(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "1999-01-01"}}, _ingest_ok)
        self.assertEqual(resp["result"]["protocolVersion"], "2025-06-18")

    def test_tools_list_exposes_log_approval(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, _ingest_ok)
        names = [t["name"] for t in resp["result"]["tools"]]
        self.assertEqual(names, ["log_approval"])
        self.assertIn("prompt_text", resp["result"]["tools"][0]["inputSchema"]["required"])

    def test_tools_call_logs_event(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": "log_approval",
                                      "arguments": {"prompt_text": "Allow restart?"}}}, _ingest_ok)
        self.assertFalse(resp["result"]["isError"])
        self.assertIn("#7", resp["result"]["content"][0]["text"])

    def test_tools_call_requires_prompt_text(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                           "params": {"name": "log_approval", "arguments": {}}}, _ingest_ok)
        self.assertTrue(resp["result"]["isError"])

    def test_unknown_tool_is_error(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "nope", "arguments": {}}}, _ingest_ok)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_notification_gets_no_response(self):
        self.assertIsNone(handle_mcp(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}, _ingest_ok))

    def test_unknown_method(self):
        resp = handle_mcp({"jsonrpc": "2.0", "id": 6, "method": "bogus"}, _ingest_ok)
        self.assertEqual(resp["error"]["code"], -32601)

    def test_ingest_failure_is_reported_not_raised(self):
        def boom(args):
            raise RuntimeError("db down")
        resp = handle_mcp({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                           "params": {"name": "log_approval",
                                      "arguments": {"prompt_text": "x"}}}, boom)
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("db down", resp["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
