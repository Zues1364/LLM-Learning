import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from conversation_state import default_conversation_state  # noqa: E402


def test_structured_state_resolves_mon_nay_followup(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))

    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_prompts: list[str] = []
    state_store: dict[str, dict] = {}

    class DummyPlanner:
        def run(self, prompt):
            planner_prompts.append(prompt)
            if "INT3412E" in prompt:
                payload = {
                    "source": "academic_advisor",
                    "context": "INT3412E dang mo lop ky 252.",
                    "memory": "",
                    "chunk_index": None,
                }
            else:
                payload = {
                    "source": "academic_advisor",
                    "context": "INT3412E - Thi giac may. Thu 3, Ca 2.",
                    "memory": "",
                    "chunk_index": None,
                }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return f"answer::{query}::{context}"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
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
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "memory_state_get":
            return state_store.get(args["session_id"], default_conversation_state())
        if tool == "memory_state_upsert":
            state_store[args["session_id"]] = args["state"]
            return args["state"]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    session_id = "e2e-state-1"

    first = client.post(
        "/ask",
        json={
            "query": "toi can lich hoc mon thi giac may o ki nay",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert first.status_code == 200
    assert "INT3412E" in first.json()["answer"]

    second = client.post(
        "/ask",
        json={
            "query": "mon nay co mo lop khong",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert second.status_code == 200
    assert "INT3412E" in second.json()["answer"]

    assert len(planner_prompts) >= 2
    assert "INT3412E" in planner_prompts[-1]

    saved_state = state_store.get(session_id) or {}
    assert saved_state.get("turn_index") == 2
    assert "INT3412E" in (saved_state.get("entities", {}).get("course_codes") or [])
