"""MCP protocol dispatch tests."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from myagent.mcp import LATEST_PROTOCOL_VERSION, handle_message

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


def req(method, msg_id=1, **params):
    return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.deps = make_deps(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_initialize_echoes_supported_version(self):
        resp = run(handle_message(self.deps, req(
            "initialize", protocolVersion="2024-11-05",
            clientInfo={"name": "test-client", "version": "1"})))
        self.assertEqual(resp["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(resp["result"]["serverInfo"]["name"], "openbrain")
        self.assertEqual(self.deps.source, "test-client")

    def test_initialize_falls_back_to_latest_for_unknown_version(self):
        resp = run(handle_message(self.deps, req(
            "initialize", protocolVersion="1999-01-01")))
        self.assertEqual(resp["result"]["protocolVersion"],
                         LATEST_PROTOCOL_VERSION)

    def test_notifications_get_no_response(self):
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self.assertIsNone(run(handle_message(self.deps, msg)))

    def test_tools_list(self):
        resp = run(handle_message(self.deps, req("tools/list", msg_id=2)))
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertEqual(names, {"recall", "remember", "forget", "expand"})

    def test_remember_recall_forget_roundtrip(self):
        saved = run(handle_message(self.deps, req(
            "tools/call", name="remember",
            arguments={"content": "The launch code is 4242.",
                       "tags": ["secret"]})))
        text = saved["result"]["content"][0]["text"]
        self.assertIn("Saved as memory #1", text)

        dup = run(handle_message(self.deps, req(
            "tools/call", name="remember",
            arguments={"content": "The launch code is 4242!"})))
        self.assertIn("Already known", dup["result"]["content"][0]["text"])

        recalled = run(handle_message(self.deps, req(
            "tools/call", name="recall",
            arguments={"query": "launch code", "tag": "secret"})))
        self.assertIn("4242", recalled["result"]["content"][0]["text"])

        gone = run(handle_message(self.deps, req(
            "tools/call", name="forget", arguments={"memory_id": 1})))
        self.assertIn("deleted", gone["result"]["content"][0]["text"])
        self.assertEqual(self.deps.store.count(), 0)

    def test_unknown_tool_is_error_result(self):
        resp = run(handle_message(self.deps, req(
            "tools/call", name="nope", arguments={})))
        self.assertTrue(resp["result"]["isError"])

    def test_unknown_method_is_jsonrpc_error(self):
        resp = run(handle_message(self.deps, req("bogus/method")))
        self.assertEqual(resp["error"]["code"], -32601)

    def test_ping_and_probe_methods(self):
        self.assertEqual(
            run(handle_message(self.deps, req("ping")))["result"], {})
        self.assertEqual(
            run(handle_message(self.deps, req("prompts/list")))["result"],
            {"prompts": []})


if __name__ == "__main__":
    unittest.main()
