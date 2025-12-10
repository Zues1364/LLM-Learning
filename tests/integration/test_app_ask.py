import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    # Reload app module fresh for isolation
    app_mod = importlib.reload(importlib.import_module("app"))

    # Redirect session cache to temp dir
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "ctx", "memory": "mem", "chunk_index": 1}
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def __init__(self):
            self.calls = []

        def run(self, query, context, source, memory_context):
            self.calls.append((query, context, source, memory_context))
            return f"answer:{query}:{source}"

    dummy_answer = DummyAnswerAgent()
    monkeypatch.setattr(app_mod, "answer_agent", dummy_answer)
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        # only memory_add and memory_get are used here
        if tool == "memory_get":
            return []
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    return app_mod


def test_ask_returns_answer(monkeypatch, app_module):
    client = TestClient(app_module.app)

    payload = {"query": "hello", "session_id": "s1", "file_ids": []}
    resp = client.post("/ask", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("answer:hello")


def test_ask_returns_friendly_on_planner_error(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "error", "context": "err_ctx"}
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "should_not_call"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(app_mod.mcp_client, "invoke", lambda tool, args: [])

    client = TestClient(app_mod.app)
    resp = client.post("/ask", json={"query": "hi"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "err_ctx"
