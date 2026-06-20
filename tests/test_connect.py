"""Onboarding auto-wiring: JSON merge safety, idempotency, detection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from myagent.connect import _json_tools, _merge_server, connect_tools


class TestMergeServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mcp.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_file_when_absent(self):
        status = _merge_server(self.path, "mcpServers", {"url": "u"})
        self.assertIn("wired", status)
        data = json.loads(self.path.read_text())
        self.assertEqual(data["mcpServers"]["openbrain"], {"url": "u"})

    def test_preserves_existing_servers_and_backs_up(self):
        self.path.write_text(json.dumps({"mcpServers": {"other": {"url": "x"}}}))
        _merge_server(self.path, "mcpServers", {"url": "u"})
        data = json.loads(self.path.read_text())
        self.assertEqual(set(data["mcpServers"]), {"other", "openbrain"})  # kept
        self.assertTrue(self.path.with_suffix(".json.bak").exists())       # backup

    def test_idempotent(self):
        _merge_server(self.path, "mcpServers", {"url": "u"})
        self.assertEqual(_merge_server(self.path, "mcpServers", {"url": "u"}),
                         "already wired")

    def test_dry_run_writes_nothing(self):
        status = _merge_server(self.path, "mcpServers", {"url": "u"}, dry_run=True)
        self.assertIn("would wire", status)
        self.assertFalse(self.path.exists())

    def test_unparseable_file_is_skipped_not_clobbered(self):
        self.path.write_text("{ not json")
        status = _merge_server(self.path, "mcpServers", {"url": "u"})
        self.assertIn("skipped", status)
        self.assertEqual(self.path.read_text(), "{ not json")  # untouched


class TestDetection(unittest.TestCase):
    def test_only_installed_tools_are_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bases = {"home": home, "appsupport": home / "AS", "xdg": home / "xdg"}
            (home / ".cursor").mkdir()                 # Cursor "installed"
            (bases["appsupport"] / "Code" / "User").mkdir(parents=True)  # VS Code
            tools = _json_tools("http://x/mcp", bases)
            active = {t.label for t in tools if t.detect.exists()}
            self.assertEqual(active, {"Cursor", "VS Code"})
            # entry shapes are tool-correct
            cur = next(t for t in tools if t.label == "Cursor")
            self.assertEqual(cur.entry, {"url": "http://x/mcp?client=cursor"})
            vsc = next(t for t in tools if t.label == "VS Code")
            self.assertEqual(vsc.entry["type"], "http")
            self.assertIn("client=vscode", vsc.entry["url"])  # provenance baked in

    def test_antigravity_detected_with_serverurl_shape(self):
        # Antigravity lives in its own subdir under ~/.gemini and wants the
        # `serverUrl` key (not url/httpUrl) for a remote HTTP server.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bases = {"home": home, "appsupport": home / "AS", "xdg": home / "xdg"}
            (home / ".gemini" / "antigravity").mkdir(parents=True)
            tools = _json_tools("http://x/mcp", bases)
            ag = next(t for t in tools if t.label == "Antigravity")
            self.assertTrue(ag.detect.exists())
            self.assertEqual(ag.servers_key, "mcpServers")
            self.assertEqual(set(ag.entry), {"serverUrl"})  # not url / httpUrl
            self.assertIn("client=antigravity", ag.entry["serverUrl"])
            self.assertTrue(str(ag.path).endswith(
                ".gemini/antigravity/mcp_config.json"))

    def test_claude_desktop_uses_stdio_bridge_not_url(self):
        # Claude Desktop's claude_desktop_config.json silently rejects {url:...}
        # entries with a popup ("not valid MCP server configurations"). The only
        # accepted shape is stdio (command + args), so we must bridge via
        # `npx mcp-remote`. Guard against accidentally regressing to {url:...}.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bases = {"home": home, "appsupport": home / "AS", "xdg": home / "xdg"}
            tools = _json_tools("http://x/mcp", bases)
            cd = next(t for t in tools if t.label == "Claude Desktop")
            self.assertNotIn("url", cd.entry)
            self.assertEqual(cd.entry.get("command"), "npx")
            self.assertIn("mcp-remote", cd.entry.get("args", []))
            self.assertTrue(any("client=claude-desktop" in a
                                for a in cd.entry["args"]))


class TestTokenBakedIntoUrls(unittest.TestCase):
    def test_token_appended_to_every_tool_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bases = {"home": home, "appsupport": home / "AS", "xdg": home / "xdg"}
            tools = _json_tools("http://x/mcp", bases, token="secret123")
            for t in tools:
                blob = json.dumps(t.entry)
                self.assertIn("token=secret123", blob, t.label)
                self.assertIn("client=", blob, t.label)  # provenance still there

    def test_no_token_means_no_token_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bases = {"home": home, "appsupport": home / "AS", "xdg": home / "xdg"}
            tools = _json_tools("http://x/mcp", bases)  # token defaults to ""
            self.assertNotIn("token=", json.dumps([t.entry for t in tools]))


class TestConnectTools(unittest.TestCase):
    def test_returns_structured_report_dry_run(self):
        # dry_run writes nothing; just assert the shape the UI/endpoint expects.
        r = connect_tools("http://127.0.0.1:3111/mcp", dry_run=True)
        self.assertIn("connected", r)
        self.assertIn("not_installed", r)
        self.assertIsInstance(r["connected"], list)
        self.assertIsInstance(r["not_installed"], list)
        for c in r["connected"]:
            self.assertIn("label", c)
            self.assertIn("status", c)


if __name__ == "__main__":
    unittest.main()
