"""HTTP endpoint tests — drives the ASGI app in-process (no network, no Ollama).

We bypass the real lifespan (which would spin up Ollama + background loops) and
mount a temp store + FakeOllama-backed deps directly onto app.state, so these
tests exercise the actual route handlers fast and deterministically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import tempfile
import unittest
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from myagent.config import load_config
from myagent.server import create_app

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        # Real config, but redirected to temp paths so nothing touches ~/.myagent.
        self.cfg = dataclasses.replace(load_config(),
                                       db_path=tmp / "m.db", vault_path=tmp / "vault")
        self.app = create_app(self.cfg)
        self.deps = make_deps(self.tmp.name, vectors={})  # FakeOllama
        # Populate the state the lifespan normally would.
        self.app.state.config = self.cfg
        self.app.state.deps = self.deps
        self.app.state.stats = {"injections": 0, "last_query": None}
        self.app.state.settings = {"paused": False, "preflight": False,
                                   "autocapture": False, "min_relevance": 0.6}
        self.app.state.health = {"enrichment": None, "sync": None}

    def tearDown(self):
        self.tmp.cleanup()

    def client(self):
        return AsyncClient(transport=ASGITransport(app=self.app),
                           base_url="http://test")

    async def _post_memory(self, c, content, tags=None):
        r = await c.post("/memories", json={"content": content, "tags": tags or []},
                         headers={"x-source": "web-ui"})
        return r.json()

    def test_health_and_stats(self):
        async def go():
            async with self.client() as c:
                h = (await c.get("/health")).json()
                self.assertEqual(h["status"], "ok")
                self.assertIn("version", h)
                s = (await c.get("/stats")).json()
                for k in ("total", "core", "auto", "paused", "preflight",
                          "autocapture", "min_relevance", "redactions"):
                    self.assertIn(k, s)
        run(go())

    def test_memory_crud_favorite_archive(self):
        async def go():
            async with self.client() as c:
                m = await self._post_memory(c, "I prefer dark mode in editors", ["prefs"])
                mid = m["id"]
                got = (await c.get("/memories")).json()
                self.assertEqual(got["count"], 1)
                # favorite → appears in favorites filter
                await c.patch(f"/memories/{mid}", json={"favorite": True})
                self.assertEqual((await c.get("/memories?favorite=true")).json()["count"], 1)
                # archive → excluded from default, present in archived view
                await c.patch(f"/memories/{mid}", json={"archived": True})
                self.assertEqual((await c.get("/memories")).json()["count"], 0)
                self.assertEqual((await c.get("/memories?archived=true")).json()["count"], 1)
                # delete → gone
                await c.delete(f"/memories/{mid}")
                self.assertEqual((await c.get("/memories?archived=true")).json()["count"], 0)
        run(go())

    def test_pagination_cursor(self):
        # Distinct vectors so dedup (cosine>=0.95) doesn't collapse them into one.
        contents = [f"distinct memory {i} alpha bravo" for i in range(5)]
        vecs = {c: [1.0 if j == i else 0.0 for j in range(5)]
                for i, c in enumerate(contents)}
        self.app.state.deps = make_deps(self.tmp.name, vectors=vecs)

        async def go():
            async with self.client() as c:
                for content in contents:
                    await self._post_memory(c, content)
                p1 = (await c.get("/memories?limit=2")).json()
                self.assertEqual(len(p1["memories"]), 2)
                self.assertIsNotNone(p1["next"])
                p2 = (await c.get(f"/memories?limit=2&after={p1['next']}")).json()
                ids1 = {m["id"] for m in p1["memories"]}
                ids2 = {m["id"] for m in p2["memories"]}
                self.assertFalse(ids1 & ids2)  # no overlap
        run(go())

    def test_settings_put_and_facets(self):
        async def go():
            async with self.client() as c:
                await self._post_memory(c, "a fact about postgres", ["db", "work"])
                f = (await c.get("/facets")).json()
                self.assertEqual(f["total"], 1)
                self.assertTrue(any(s["source"] == "web-ui" for s in f["sources"]))
                put = (await c.put("/settings", json={"preflight": True,
                                                      "min_relevance": 0.75})).json()
                self.assertTrue(put["preflight"])
                self.assertEqual(put["min_relevance"], 0.75)
                self.assertTrue((await c.get("/settings")).json()["preflight"])
        run(go())

    def test_connectors_surface_in_stats(self):
        async def go():
            async with self.client() as c:
                s = (await c.get("/stats")).json()
                self.assertIn("connectors", s)
                mem = next(x for x in s["connectors"] if x["key"] == "memory")
                self.assertTrue(mem["enabled"])
                self.assertFalse(mem["toggleable"])  # the core can't be removed
                self.assertEqual(set(mem["tools"]),
                                 {"recall", "remember", "forget", "expand"})
        run(go())

    def test_builtin_connector_cannot_be_disabled(self):
        async def go():
            async with self.client() as c:
                # Hostile/ignorant client tries to switch the core off.
                await c.put("/settings", json={"connectors": {"memory": False}})
                s = (await c.get("/stats")).json()
                mem = next(x for x in s["connectors"] if x["key"] == "memory")
                self.assertTrue(mem["enabled"])  # ignored — still on
        run(go())

    def test_export_import_roundtrip(self):
        async def go():
            async with self.client() as c:
                await self._post_memory(c, "exportable knowledge about caching", ["k"])
                exp = (await c.get("/export")).json()
                self.assertEqual(exp["count"], 1)
                # re-import the same bundle → dedup catches it
                r = (await c.post("/import", json={"memories": exp["memories"]})).json()
                self.assertEqual(r["duplicates"], 1)
                self.assertEqual(r["imported"], 0)
        run(go())

    def test_mcp_provenance_via_client_param(self):
        async def go():
            async with self.client() as c:
                await c.post("/mcp?client=cursor", json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "remember", "arguments": {
                        "content": "The team standardized on TypeScript for frontend.",
                        "tags": ["decision"]}}})
                got = (await c.get("/memories?source=cursor")).json()
                self.assertEqual(got["count"], 1)
                self.assertEqual(got["memories"][0]["source"], "cursor")
        run(go())

    def test_bad_cursor_does_not_500(self):
        async def go():
            async with self.client() as c:
                r = await c.get("/memories?after=garbage")
                self.assertEqual(r.status_code, 200)
        run(go())


if __name__ == "__main__":
    unittest.main()
