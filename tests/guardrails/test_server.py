"""Guard Rails REST API: auth gate, ingestion, lifecycle, redaction, stats."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from guardrails.config import Config
from guardrails.server import create_app
from guardrails.auth import read_token


def _client(tmp: str):
    cfg = Config(port=3112, db_path=Path(tmp) / "g.db", max_field_chars=20000)
    app = create_app(cfg)
    client = TestClient(app)
    token = read_token(cfg.db_path.parent)
    return client, {"X-Guardrails-Token": token}


class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client, self.H = _client(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_health_and_dashboard_are_open(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_endpoints_require_token(self):
        self.assertEqual(self.client.post("/events", json={"prompt_text": "x"}).status_code, 401)
        self.assertEqual(self.client.get("/events").status_code, 401)
        self.assertEqual(self.client.get("/stats").status_code, 401)

    def test_create_and_get_event(self):
        body = {
            "user_request": "Run the application",
            "agent_action": "Need to restart the server.",
            "prompt_text": "Restart the server to apply changes?",
            "options": ["Restart", "Skip", "Cancel"],
            "selected_option": "Restart",
            "result": "success",
            "ide": "claude-code", "agent": "claude-code",
            "repository": "open_brain", "branch": "main",
            "session_id": "sess-123", "tool_name": "Bash",
        }
        r = self.client.post("/events", headers=self.H, json=body)
        self.assertEqual(r.status_code, 201)
        ev = r.json()
        self.assertEqual(ev["status"], "completed")
        self.assertEqual(ev["selected_option"], "Restart")
        self.assertEqual(ev["options"], ["Restart", "Skip", "Cancel"])

        got = self.client.get(f"/events/{ev['id']}", headers=self.H)
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["user_request"], "Run the application")

    def test_prompt_text_required(self):
        r = self.client.post("/events", headers=self.H, json={"user_request": "hi"})
        self.assertEqual(r.status_code, 422)

    def test_patch_fills_in_decision_and_result(self):
        ev = self.client.post("/events", headers=self.H,
                              json={"prompt_text": "Allow?", "options": ["Yes", "No"]}).json()
        self.assertEqual(ev["status"], "pending")
        r = self.client.patch(f"/events/{ev['id']}", headers=self.H,
                             json={"selected_option": "Yes", "result": "success"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "completed")

    def test_patch_unknown_404(self):
        r = self.client.patch("/events/999", headers=self.H, json={"result": "ok"})
        self.assertEqual(r.status_code, 404)

    def test_patch_empty_400(self):
        ev = self.client.post("/events", headers=self.H, json={"prompt_text": "x"}).json()
        r = self.client.patch(f"/events/{ev['id']}", headers=self.H, json={})
        self.assertEqual(r.status_code, 400)

    def test_delete(self):
        ev = self.client.post("/events", headers=self.H, json={"prompt_text": "x"}).json()
        self.assertEqual(self.client.delete(f"/events/{ev['id']}", headers=self.H).status_code, 200)
        self.assertEqual(self.client.get(f"/events/{ev['id']}", headers=self.H).status_code, 404)

    def test_secrets_are_redacted_on_ingest(self):
        r = self.client.post("/events", headers=self.H, json={
            "prompt_text": "Run: deploy --token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa now",
            "agent_action": "email jane.doe@example.com the build",
        })
        ev = r.json()
        self.assertNotIn("ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", ev["prompt_text"])
        self.assertIn("[REDACTED", ev["prompt_text"])
        self.assertNotIn("jane.doe@example.com", ev["agent_action"])
        self.assertTrue(ev["metadata"].get("_redacted"))

    def test_list_filter_and_stats(self):
        self.client.post("/events", headers=self.H, json={"prompt_text": "a", "ide": "cursor"})
        self.client.post("/events", headers=self.H, json={"prompt_text": "b", "ide": "vscode"})
        listed = self.client.get("/events?ide=cursor", headers=self.H).json()
        self.assertEqual(listed["count"], 1)
        stats = self.client.get("/stats", headers=self.H).json()
        self.assertEqual(stats["total"], 2)

    def test_export(self):
        for i in range(3):
            self.client.post("/events", headers=self.H, json={"prompt_text": f"p{i}"})
        exp = self.client.get("/export", headers=self.H).json()
        self.assertEqual(exp["count"], 3)


if __name__ == "__main__":
    unittest.main()
