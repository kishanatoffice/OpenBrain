"""Connector registry / platform contract.

Locks in the guarantees the platform layer must keep as connectors are added:
  * the registry is internally consistent (tool names map back to connectors),
  * built-in connectors cannot be disabled,
  * a disabled connector's tools disappear from tools/list AND are rejected on
    a direct tools/call (defense in depth at the protocol boundary).
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from myagent import connectors
from myagent.connectors import (
    REGISTRY,
    Connector,
    connector_for_tool,
    default_enabled_keys,
    is_enabled,
    tool_specs_for,
)
from myagent.mcp import call_tool, handle_message

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


def req(method, msg_id=1, **params):
    return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}


class RegistryContractCase(unittest.TestCase):
    def test_every_tool_maps_back_to_its_connector(self):
        for c in REGISTRY.values():
            for spec in c.tool_specs:
                self.assertIs(connector_for_tool(spec["name"]), c)

    def test_tool_names_are_globally_unique(self):
        seen: set[str] = set()
        for c in REGISTRY.values():
            for name in c.tool_names:
                self.assertNotIn(name, seen, f"duplicate tool name {name!r}")
                seen.add(name)

    def test_memory_is_a_non_toggleable_default(self):
        mem = REGISTRY["memory"]
        self.assertFalse(mem.toggleable)
        self.assertTrue(mem.default_enabled)
        self.assertIn("memory", default_enabled_keys())

    def test_none_resolves_to_defaults(self):
        names = {t["name"] for t in tool_specs_for(None)}
        self.assertEqual(names, {"recall", "remember", "forget", "expand"})

    def test_unknown_tool_has_no_connector(self):
        self.assertIsNone(connector_for_tool("does-not-exist"))


class ConnectorSwitchCase(unittest.TestCase):
    """A registered, toggleable connector that can be flipped off — proves the
    switch works end to end without depending on a specific add-on existing."""

    def setUp(self):
        self.calls: list[str] = []

        async def handler(deps, name, args):
            self.calls.append(name)
            return "stub ok"

        self.stub = Connector(
            key="stub",
            label="Stub",
            description="test-only connector",
            tool_specs=[{"name": "stub_tool", "description": "x",
                         "inputSchema": {"type": "object", "properties": {}}}],
            handler=handler,
            default_enabled=False,
            toggleable=True,
        )
        REGISTRY[self.stub.key] = self.stub
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        REGISTRY.pop("stub", None)
        self.tmp.cleanup()

    def _deps(self, enabled):
        d = make_deps(self.tmp.name)
        d.enabled_connectors = enabled
        return d

    def test_disabled_tool_absent_from_tools_list(self):
        names = {t["name"] for t in tool_specs_for(default_enabled_keys())}
        self.assertNotIn("stub_tool", names)
        # And present once enabled.
        names = {t["name"] for t in tool_specs_for({"memory", "stub"})}
        self.assertIn("stub_tool", names)

    def test_disabled_tool_rejected_on_direct_call(self):
        # Defense in depth: even a direct tools/call must not run a disabled
        # connector's handler (a client could have cached the schema).
        deps = self._deps(frozenset({"memory"}))  # stub OFF
        out = run(call_tool(deps, "stub_tool", {}))
        self.assertIn("switched off", out)
        self.assertEqual(self.calls, [])  # handler never ran

    def test_enabled_tool_runs(self):
        deps = self._deps(frozenset({"memory", "stub"}))
        out = run(call_tool(deps, "stub_tool", {}))
        self.assertEqual(out, "stub ok")
        self.assertEqual(self.calls, ["stub_tool"])

    def test_tools_list_over_jsonrpc_reflects_switch(self):
        deps = self._deps(frozenset({"memory", "stub"}))
        resp = run(handle_message(deps, req("tools/list")))
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertEqual(names,
                         {"recall", "remember", "forget", "expand", "stub_tool"})

    def test_is_enabled_respects_explicit_set(self):
        self.assertTrue(is_enabled("stub", {"stub"}))
        self.assertFalse(is_enabled("stub", {"memory"}))


if __name__ == "__main__":
    unittest.main()
