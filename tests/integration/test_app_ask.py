import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))

    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_prompts: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_prompts.append(prompt)
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
        if tool == "memory_get":
            return []
        if tool == "get_available_programs":
            return {
                "programs": [
                    {
                        "id": "cs_2022",
                        "name": "Khoa hoc may tinh",
                        "year": "2022",
                        "display_name": "Khoa hoc may tinh (QH-2022-2024)",
                    }
                ]
            }
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    app_mod._planner_prompts = planner_prompts
    app_mod._dummy_answer = dummy_answer
    return app_mod


def test_ask_requires_program_selection_when_missing(app_module):
    client = TestClient(app_module.app)

    payload = {"query": "hello", "session_id": "s1", "file_ids": []}
    resp = client.post("/ask", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_program_selection"] is True
    assert body["selected_program_id"] is None
    assert len(body["programs"]) == 1
    assert app_module._dummy_answer.calls == []
    assert app_module._planner_prompts == []


def test_ask_returns_answer_with_program_and_injects_prompt(app_module):
    client = TestClient(app_module.app)

    payload = {"query": "hello", "session_id": "s2", "file_ids": [], "program_id": "cs_2022"}
    resp = client.post("/ask", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("answer:hello")
    assert body["selected_program_id"] == "cs_2022"
    assert app_module._planner_prompts, "planner should be called"
    assert "[PROGRAM:cs_2022]" in app_module._planner_prompts[-1]


def test_ask_uses_cached_program_for_next_request(app_module):
    client = TestClient(app_module.app)
    session_id = "s3"

    first = client.post(
        "/ask",
        json={"query": "first", "session_id": session_id, "file_ids": [], "program_id": "cs_2022"},
    )
    assert first.status_code == 200
    assert first.json()["selected_program_id"] == "cs_2022"

    second = client.post("/ask", json={"query": "second", "session_id": session_id, "file_ids": []})
    assert second.status_code == 200
    assert second.json()["selected_program_id"] == "cs_2022"
    assert app_module._planner_prompts[-1].count("[PROGRAM:cs_2022]") == 1


def test_ask_maps_planner_program_selection_response(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "program_selection",
                "context": json.dumps(
                    {
                        "programs": [
                            {
                                "id": "it_2025",
                                "name": "Cong nghe thong tin",
                                "year": "2025",
                                "display_name": "Cong nghe thong tin (QH-2025)",
                            }
                        ]
                    }
                ),
                "memory": "mem",
                "chunk_index": None,
                "requires_selection": True,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "should_not_call"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "it_2025", "display_name": "Cong nghe thong tin (QH-2025)"}]}
        if tool == "memory_get":
            return []
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "hi", "session_id": "s4", "program_id": "it_2025", "file_ids": []},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_program_selection"] is True
    assert body["programs"][0]["id"] == "it_2025"
