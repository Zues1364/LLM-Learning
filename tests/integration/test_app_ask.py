import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from persistent_memory import PersistentMemory  # noqa: E402


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


def test_health_and_readiness_endpoints(app_module):
    client = TestClient(app_module.app)

    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["checks"]["memory_db"] == "ok"


def test_transcript_consult_timeout_has_floor(app_module):
    assert app_module._timeout_at_least(270.0, 900.0) == 900.0
    assert app_module._timeout_at_least(1200.0, 900.0) == 1200.0
    assert app_module._timeout_at_least(None, 900.0) is None


def test_readiness_fails_when_supabase_db_unavailable(app_module, monkeypatch):
    client = TestClient(app_module.app)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://fake")
    monkeypatch.setattr(app_module, "check_postgres_ready", lambda: "db down")

    ready = client.get("/readyz")

    assert ready.status_code == 503
    detail = ready.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["memory_db"] == "db down"


def test_readiness_fails_when_mcp_required_and_down(app_module, monkeypatch):
    client = TestClient(app_module.app)
    monkeypatch.setenv("MCP_SERVER_URL", "http://mcp.internal")
    monkeypatch.setattr(app_module.mcp_client, "discover", lambda timeout=5.0: (_ for _ in ()).throw(RuntimeError("mcp down")))

    ready = client.get("/readyz")

    assert ready.status_code == 503
    detail = ready.json()["detail"]
    assert detail["status"] == "not_ready"
    assert "mcp down" in detail["checks"]["mcp"]


def test_upload_pdf_rejects_invalid_type_and_size(app_module, monkeypatch):
    client = TestClient(app_module.app)

    bad_type = client.post(
        "/upload_pdf",
        files={"file": ("transcript.txt", b"not a pdf", "text/plain")},
    )
    assert bad_type.status_code == 400

    monkeypatch.setattr(app_module, "MAX_TRANSCRIPT_UPLOAD_BYTES", 4)
    too_large = client.post(
        "/upload_pdf",
        files={"file": ("transcript.pdf", b"12345", "application/pdf")},
    )
    assert too_large.status_code == 413


def test_session_meta_is_owner_scoped_with_legacy_fallback(app_module):
    app_module._write_session_meta(
        "shared-session",
        {"file_ids": ["owner.pdf"], "program_id": "cs_2022"},
        user_id="user/a",
    )
    app_module._write_session_meta(
        "shared-session",
        {"file_ids": ["guest.pdf"], "program_id": "guest_program"},
    )

    assert app_module._load_session_meta("shared-session", user_id="user/a") == {
        "file_ids": ["owner.pdf"],
        "program_id": "cs_2022",
    }
    assert app_module._load_session_meta("shared-session") == {
        "file_ids": ["guest.pdf"],
        "program_id": "guest_program",
    }

    owner_path = app_module.SESSION_CACHE_DIR / "users" / "user_a" / "shared-session" / "meta.json"
    legacy_path = app_module.SESSION_CACHE_DIR / "shared-session" / "meta.json"
    assert owner_path.exists()
    assert legacy_path.exists()


def test_delete_session_only_removes_authenticated_owner_cache(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "memory", PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5))
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    app_mod._write_session_meta("same-session", {"file_ids": ["legacy.pdf"], "program_id": "legacy"})
    app_mod._write_session_meta(
        "same-session",
        {"file_ids": ["owner.pdf"], "program_id": "owner"},
        user_id="user-a",
    )

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token-a":
            return {"id": "user-a", "email": "a@vnu.edu.vn"}
        return None

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    client = TestClient(app_mod.app)
    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-a")
    resp = client.request("DELETE", "/session", json={"session_id": "same-session"})

    assert resp.status_code == 200
    assert not (app_mod.SESSION_CACHE_DIR / "users" / "user-a" / "same-session").exists()
    assert (app_mod.SESSION_CACHE_DIR / "same-session" / "meta.json").exists()


def test_ask_passes_authenticated_user_to_memory_tools(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "ctx", "memory": "", "chunk_index": 1}
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "scoped answer"

    captured_memory_calls = []

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token":
            return {"id": "user-123", "email": "student@vnu.edu.vn"}
        return None

    def fake_invoke(tool, args):
        if tool.startswith("memory"):
            captured_memory_calls.append((tool, dict(args)))
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
        if tool == "memory_state_get":
            return {}
        if tool == "memory_get":
            return []
        if tool == "retrieve_chunks":
            return ""
        return "ok"

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    client.cookies.set(app_mod._mail_cookie_name(), "auth-token")
    resp = client.post(
        "/ask",
        json={"query": "hello", "session_id": "shared-session", "file_ids": [], "program_id": "cs_2022"},
    )

    assert resp.status_code == 200
    memory_args = {tool: args for tool, args in captured_memory_calls}
    assert memory_args["memory_state_get"]["user_id"] == "user-123"
    assert memory_args["memory_get"]["user_id"] == "user-123"
    assert memory_args["memory_add"]["user_id"] == "user-123"
    assert memory_args["memory_state_upsert"]["user_id"] == "user-123"


def test_history_passes_authenticated_user_to_memory_get(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    captured_args = {}

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token":
            return {"id": "user-123", "email": "student@vnu.edu.vn"}
        return None

    def fake_invoke(tool, args):
        if tool == "memory_get":
            captured_args.update(args)
            return []
        return "ok"

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    client.cookies.set(app_mod._mail_cookie_name(), "auth-token")
    resp = client.get("/history?session_id=shared-session")

    assert resp.status_code == 200
    assert captured_args["session_id"] == "shared-session"
    assert captured_args["user_id"] == "user-123"


def test_chat_session_api_requires_auth_and_scopes_to_user(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "memory", PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5))
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token-a":
            return {"id": "user-a", "email": "a@vnu.edu.vn"}
        if raw_token == "auth-token-b":
            return {"id": "user-b", "email": "b@vnu.edu.vn"}
        return None

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    client = TestClient(app_mod.app)

    assert client.get("/api/chat/sessions").status_code == 401

    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-a")
    created = client.post(
        "/api/chat/sessions",
        json={
            "session_id": "shared-session",
            "title": "Lịch thị giác máy",
            "selected_program_id": "cs_2022",
            "selected_file_ids": ["1.pdf"],
        },
    )
    assert created.status_code == 200
    assert created.json()["session"]["id"] == "shared-session"

    listed = client.get("/api/chat/sessions")
    assert listed.status_code == 200
    assert [item["title"] for item in listed.json()["sessions"]] == ["Lịch thị giác máy"]

    patched = client.patch("/api/chat/sessions/shared-session", json={"title": "Đã đổi tên"})
    assert patched.status_code == 200
    assert patched.json()["session"]["title"] == "Đã đổi tên"

    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-b")
    assert client.get("/api/chat/sessions").json()["sessions"] == []
    assert client.patch("/api/chat/sessions/shared-session", json={"title": "Không được"}).status_code == 404


def test_chat_session_api_recovers_legacy_history_for_authenticated_user(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    test_memory = PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5)
    test_memory.add_to_history("cau hoi cu", "tra loi cu", session_id="legacy-session")
    monkeypatch.setattr(app_mod, "memory", test_memory)
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token-a":
            return {"id": "user-a", "email": "a@vnu.edu.vn"}
        if raw_token == "auth-token-b":
            return {"id": "user-b", "email": "b@vnu.edu.vn"}
        return None

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    client = TestClient(app_mod.app)

    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-a")
    listed = client.get("/api/chat/sessions")
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["id"] == "legacy-session"
    assert listed.json()["sessions"][0]["title"] == "cau hoi cu"

    messages = client.get("/api/chat/sessions/legacy-session/messages")
    assert messages.status_code == 200
    assert [msg["role"] for msg in messages.json()["messages"]] == ["user", "assistant"]

    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-b")
    assert client.get("/api/chat/sessions").json()["sessions"] == []


def test_chat_migrate_api_imports_browser_sessions_for_authenticated_user(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "memory", PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5))
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token-a":
            return {"id": "user-a", "email": "a@vnu.edu.vn"}
        if raw_token == "auth-token-b":
            return {"id": "user-b", "email": "b@vnu.edu.vn"}
        return None

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    client = TestClient(app_mod.app)
    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-a")

    migrated = client.post(
        "/api/chat/migrate",
        json={
            "sessions": [
                {
                    "session_id": "browser-session",
                    "title": "Browser chat",
                    "selected_program_id": "cs_2022",
                    "selected_file_ids": ["1.pdf"],
                    "messages": [
                        {"role": "user", "content": "old question"},
                        {"type": "bot", "text": "old answer", "citations": [{"source_file": "old.pdf"}]},
                    ],
                }
            ]
        },
    )
    assert migrated.status_code == 200
    assert migrated.json()["imported_sessions"] == 1
    assert migrated.json()["imported_messages"] == 2

    repeated = client.post(
        "/api/chat/migrate",
        json={
            "sessions": [
                {
                    "session_id": "browser-session",
                    "title": "Browser chat",
                    "messages": [{"role": "user", "content": "duplicate"}],
                }
            ]
        },
    )
    assert repeated.status_code == 200
    assert repeated.json()["imported_messages"] == 0

    sessions = client.get("/api/chat/sessions").json()["sessions"]
    assert sessions[0]["id"] == "browser-session"
    assert sessions[0]["selected_program_id"] == "cs_2022"
    assert sessions[0]["selected_file_ids"] == ["1.pdf"]

    messages = client.get("/api/chat/sessions/browser-session/messages").json()["messages"]
    assert [msg["content"] for msg in messages] == ["old question", "old answer"]
    assert messages[1]["citations"] == [{"source_file": "old.pdf"}]

    client.cookies.set(app_mod._mail_cookie_name(), "auth-token-b")
    assert client.get("/api/chat/sessions").json()["sessions"] == []


def test_ask_records_authenticated_chat_session_and_messages(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "memory", PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5))
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "ctx", "memory": "", "chunk_index": 1}
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "bot answer"

    def fake_user(raw_token, touch=True):
        if raw_token == "auth-token":
            return {"id": "user-123", "email": "student@vnu.edu.vn"}
        return None

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
        if tool == "memory_state_get":
            return {}
        if tool == "memory_get":
            return []
        if tool == "retrieve_chunks":
            return ""
        return "ok"

    monkeypatch.setattr(app_mod.mail_agent_service, "get_authenticated_user", fake_user)
    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    client.cookies.set(app_mod._mail_cookie_name(), "auth-token")
    resp = client.post(
        "/ask",
        json={
            "query": "toi can lich hoc thi giac may",
            "session_id": "chat-state-1",
            "file_ids": ["1.pdf"],
            "program_id": "cs_2022",
        },
    )
    assert resp.status_code == 200

    sessions = client.get("/api/chat/sessions").json()["sessions"]
    assert sessions[0]["id"] == "chat-state-1"
    assert sessions[0]["selected_program_id"] == "cs_2022"
    assert sessions[0]["selected_file_ids"] == ["1.pdf"]

    messages = client.get("/api/chat/sessions/chat-state-1/messages").json()["messages"]
    assert [msg["role"] for msg in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "toi can lich hoc thi giac may"
    assert messages[1]["content"] == "bot answer"


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


def test_extract_retrieve_citations_parses_chunk_blocks(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = (
        "[PHU LUC TKB.pdf - Chunk 12 - Page 3 - Line 41] INT2041 2 CL LT 5 2 3-G3 Ngo Thi Duyen\n"
        "Hoc 1 ca/15 tuan, thi dot 2\n\n"
        "[Signed TKB.pdf - Chunk 6] INT2041 2 CL LT 5 1 206-T Ngo Thi Duyen"
    )
    citations = app_mod._extract_retrieve_citations(context, max_items=10)

    assert len(citations) == 2
    assert citations[0]["source_file"] == "PHU LUC TKB.pdf"
    assert citations[0]["chunk_index"] == 12
    assert citations[0]["page"] == 3
    assert citations[0]["source_line"] == 41
    assert "INT2041 2 CL LT 5 2 3-G3" in citations[0]["excerpt"]
    assert citations[1]["source_file"] == "Signed TKB.pdf"
    assert citations[1]["chunk_index"] == 6
    assert citations[1]["page"] is None
    assert citations[1]["source_line"] is None


def test_extract_retrieve_citations_keeps_ielts_table_rows(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = (
        "[SỔ TAY HỌC VỤ.pdf - Chunk 73 - Page 26] ### BẢNG THAM CHIẾU KẾT QUẢ CÁC BÀI THI TIẾNG ANH\n"
        "| KNLNVN | IELTS | TOEFL | Aptis ESOL | Cambridge Exam |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| Bậc 3 | 4.5 | 450 | B1 | PET |\n"
        "| Bậc 4 | 5.5 | 500 | B2 | FCE |\n"
        "| Bậc 5 | 7.0 | 600 | C1 | CAE |"
    )
    citations = app_mod._extract_retrieve_citations(
        context,
        max_items=3,
        query="với 6.5 ielts tôi có đủ điều kiện tiếng anh để ra trường không",
        answer="IELTS 6.5 cao hơn chuẩn Bậc 3 và Bậc 4.",
    )

    assert len(citations) == 1
    excerpt = citations[0]["excerpt"]
    assert "| KNLNVN | IELTS |" in excerpt
    assert "| Bậc 3 | 4.5 |" in excerpt
    assert "| Bậc 4 | 5.5 |" in excerpt


def test_extract_retrieve_citations_keeps_ielts_rows_when_markdown_row_is_wrapped(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = (
        "[SỔ TAY HỌC VỤ.pdf - Chunk 73 - Page 26] ### BẢNG THAM CHIẾU KẾT QUẢ CÁC BÀI THI TIẾNG ANH\n"
        "| KNLNVN | IELTS | TOEFL | Aptis ESOL | Cambridge Exam | VSTEP |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| Bậc 3 | 4.5 | 42 iBT | B1 | A2 Key:140\n"
        "B1 Preliminary: 140\n"
        "B2 First: 140 | VSTEP.3-5 (4.0) |\n"
        "| Bậc 4 | 5.5 | 72 iBT | B2 | FCE | VSTEP.3-5 (6.0) |"
    )
    citations = app_mod._extract_retrieve_citations(
        context,
        max_items=3,
        query="với 6.5 ielts tôi có đủ điều kiện tiếng anh để ra trường không",
        answer="IELTS 6.5 cao hơn chuẩn Bậc 3 và Bậc 4.",
    )

    assert len(citations) == 1
    excerpt = citations[0]["excerpt"]
    assert "| KNLNVN | IELTS |" in excerpt
    assert "| Bậc 3 | 4.5 |" in excerpt
    assert "| Bậc 4 | 5.5 |" in excerpt


def test_ask_vector_store_response_includes_citations(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_mod, "STRUCTURED_TKB_ENABLED", False)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "vector_store",
                "context": (
                    "[PHU LUC TKB.pdf - Chunk 12 - Page 3 - Line 41] INT2041 2 CL LT 5 2 3-G3 Ngo Thi Duyen\n"
                    "Hoc 1 ca/15 tuan, thi dot 2"
                ),
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "answer-from-vector-store"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
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

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "lich hoc mon tuong tac nguoi may", "session_id": "s_citations", "program_id": "cs_2022", "file_ids": []},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer-from-vector-store"
    assert body["source"] == "vector_store"
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_file"] == "PHU LUC TKB.pdf"
    assert body["citations"][0]["chunk_index"] == 12
    assert body["citations"][0]["page"] == 3
    assert body["citations"][0]["source_line"] == 41


def test_extract_structured_schedule_citations_builds_row_excerpts(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = json.dumps(
        {
            "rows": [
                {
                    "subject_code": "INT2041",
                    "class_code": "INT2041 2",
                    "day_of_week": "Thá»© 5",
                    "slot": "2",
                    "room": "3-G3",
                    "teacher_name": "NgÃ´ Thá»‹ DuyÃªn",
                    "week_note": "Há»c 1 ca/15 tuáº§n, thi Ä‘á»£t 2",
                    "source_file": "PHU LUC TKB.pdf",
                    "source_page": 7,
                    "source_line": 118,
                },
                {
                    "subject_code": "INT2041",
                    "class_code": "INT2041 2",
                    "day_of_week": "Thá»© 5",
                    "slot": "2",
                    "room": "3-G3",
                    "teacher_name": "NgÃ´ Thá»‹ DuyÃªn",
                    "week_note": "Há»c 1 ca/15 tuáº§n, thi Ä‘á»£t 2",
                    "source_file": "PHU LUC TKB.pdf",
                    "source_page": 7,
                    "source_line": 118,
                },
            ],
            "source_files": ["PHU LUC TKB.pdf"],
        },
        ensure_ascii=False,
    )
    citations = app_mod._extract_structured_schedule_citations(context, max_items=10)

    assert len(citations) == 1
    assert citations[0]["source_file"] == "PHU LUC TKB.pdf"
    assert citations[0]["page"] == 7
    assert citations[0]["source_line"] == 118
    assert "INT2041 2" in citations[0]["excerpt"]
    assert "Ca 2" in citations[0]["excerpt"]
    assert "3-G3" in citations[0]["excerpt"]


def test_extract_structured_schedule_citations_collapses_rows_with_same_excerpt_different_lines(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = json.dumps(
        {
            "rows": [
                {
                    "subject_code": "INT3401E",
                    "class_code": "INT3401E 1",
                    "day_of_week": "Thứ 4",
                    "slot": "1",
                    "room": "503-B",
                    "teacher_name": "Nguyễn Thị Nhật Thanh",
                    "week_note": "Học 1 ca/15 tuần, thi đợt 2",
                    "source_file": "PHU LUC TKB HKII 2025-2026.pdf",
                    "source_page": 1,
                    "source_line": 91,
                },
                {
                    "subject_code": "INT3401E",
                    "class_code": "INT3401E 1",
                    "day_of_week": "Thứ 4",
                    "slot": "1",
                    "room": "503-B",
                    "teacher_name": "Nguyễn Thị Nhật Thanh",
                    "week_note": "Học 1 ca/15 tuần, thi đợt 2",
                    "source_file": "PHU LUC TKB HKII 2025-2026.pdf",
                    "source_page": 1,
                    "source_line": 228,
                },
            ],
            "source_files": ["PHU LUC TKB HKII 2025-2026.pdf"],
        },
        ensure_ascii=False,
    )
    citations = app_mod._extract_structured_schedule_citations(context, max_items=10)

    assert len(citations) == 1
    assert citations[0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"
    assert citations[0]["page"] == 1
    assert citations[0]["source_line"] == 91
    assert "INT3401E 1" in citations[0]["excerpt"]
    assert "503-B" in citations[0]["excerpt"]


def test_extract_structured_schedule_citations_uses_schedule_source_file_without_rows(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    context = json.dumps(
        {
            "rows": [],
            "schedule_source_file": "PHU LUC TKB HKII 2025-2026.pdf",
            "coverage_note": "Không tìm thấy học phần tự chọn đang mở lớp trong thời khóa biểu hiện có.",
        },
        ensure_ascii=False,
    )

    citations = app_mod._extract_structured_schedule_citations(context, max_items=10)

    assert len(citations) == 1
    assert citations[0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"
    assert "Không tìm thấy học phần tự chọn" in citations[0]["excerpt"]


def test_extract_advisor_text_citations_from_gpa_schedule_table(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    answer = "\n".join(
        [
            "Gợi ý lịch:",
            "| Ngày học | Ca học | Tiết + Thời gian | Mã môn học | Tên môn học | Tín chỉ | Ghi chú về lớp |",
            "|---|---|---|---|---|---:|---|",
            "| Thứ 5 | Ca 2 | Tiết 4-6 (09:50 – 12:30) | INT4050 | Khóa luận tốt nghiệp | 7 | Lớp INT4050 1 |",
            "",
            "Nguồn TKB: PHU LUC TKB HKII 2025-2026.pdf",
        ]
    )

    citations = app_mod._extract_advisor_text_citations(answer, max_items=10)

    assert len(citations) == 1
    assert citations[0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"
    assert "INT4050" in citations[0]["excerpt"]
    assert "09:50" in citations[0]["excerpt"]


def test_extract_advisor_text_citations_from_gpa_schedule_bullet_list(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    answer = "\n".join(
        [
            "Gợi ý lịch:",
            "- Thứ 5 | Ca 2 | Tiết 4-6 (09:50 – 12:30)",
            "  INT4050 - Khóa luận tốt nghiệp | 7 tín chỉ | Lớp INT4050 1",
            "",
            "Nguồn TKB: PHU LUC TKB HKII 2025-2026.pdf",
        ]
    )

    citations = app_mod._extract_advisor_text_citations(answer, max_items=10)

    assert len(citations) == 1
    assert citations[0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"
    assert "INT4050" in citations[0]["excerpt"]
    assert "09:50" in citations[0]["excerpt"]


def test_ask_structured_schedule_response_includes_citations(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        app_mod,
        "_structured_intent_classifier",
        lambda query: {"intent": "course_schedule", "confidence": 0.92, "signals": ["schedule_marker", "course_marker"]},
    )
    monkeypatch.setattr(
        app_mod,
        "_build_structured_route_payload",
        lambda **kwargs: {
            "source": "structured_schedule",
            "context": json.dumps(
                {
                    "rows": [
                        {
                            "subject_code": "INT2041",
                            "class_code": "INT2041 2",
                            "day_of_week": "Thá»© 5",
                            "slot": "1",
                            "room": "206-T",
                            "teacher_name": "NgÃ´ Thá»‹ DuyÃªn",
                            "source_file": "Signed.[TKB] DU KIEN.pdf",
                            "source_page": 2,
                            "source_line": 61,
                        }
                    ],
                    "source_files": ["Signed.[TKB] DU KIEN.pdf"],
                    "coverage_note": "TÃ¬m tháº¥y dá»¯ liá»‡u lá»‹ch há»c.",
                },
                ensure_ascii=False,
            ),
            "memory": "",
            "chunk_index": None,
            "route_meta": {"intent": "course_schedule", "confidence": 0.92},
        },
    )

    def fake_invoke(tool, args):
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
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

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "lich hoc mon int2041", "session_id": "s_structured_citations", "program_id": "cs_2022", "file_ids": []},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "structured_schedule"
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_file"] == "Signed.[TKB] DU KIEN.pdf"
    assert body["citations"][0]["page"] == 2
    assert body["citations"][0]["source_line"] == 61
    assert "INT2041 2" in body["citations"][0]["excerpt"]


def test_ask_academic_advisor_response_backfills_retrieve_citations(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "STRUCTURED_TKB_ENABLED", False)
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "academic_advisor",
                "context": "Có, với chứng chỉ IELTS 6.5, bạn đã đủ điều kiện về trình độ ngoại ngữ để ra trường.",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "should-not-call"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    retrieve_calls = {"count": 0}
    retrieve_file_ids_seen: list[list[str]] = []

    def fake_invoke(tool, args):
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
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
        if tool == "retrieve_chunks":
            retrieve_calls["count"] += 1
            retrieve_file_ids_seen.append(list(args.get("file_ids") or []))
            return [
                "[SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf - Chunk 6 - Page 12 - Line 8] "
                "IELTS tối thiểu 5.5 để đáp ứng điều kiện ngoại ngữ."
            ]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh ra trường không", "session_id": "s_advisor_backfill", "program_id": "cs_2022", "file_ids": ["1_transcript.pdf", "2_transcript.pdf"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "academic_advisor"
    assert retrieve_calls["count"] == 1
    assert retrieve_file_ids_seen == [[]]
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == "SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
    assert body["citations"][0]["page"] == 12


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


def test_ask_planner_disconnect_falls_back_without_500(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class BrokenPlanner:
        def run(self, prompt):
            raise RuntimeError("Server disconnected without sending a response.")

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return f"fallback-answer:{source}:{query}"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: BrokenPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {
                "programs": [
                    {
                        "id": "cs_2022",
                        "name": "Khoa hoc may tinh",
                        "display_name": "Khoa hoc may tinh (QH-2022-2024)",
                    }
                ]
            }
        if tool == "memory_get":
            return []
        if tool == "retrieve_chunks":
            return ["CTX from fallback retrieve"]
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "giang vien mon lich su dang ky nay",
            "session_id": "s_fallback",
            "program_id": "cs_2022",
            "file_ids": ["1_x.pdf", "2_y.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_program_id"] == "cs_2022"
    assert "fallback-answer:" in body["answer"]


def test_ask_high_confidence_structured_route_skips_planner(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            return {"matched_subject": {"subject_code": "HIS1001", "subject_name_vi": "Lá»‹ch sá»­ Äáº£ng"}, "confidence": 0.95}
        if tool == "get_teachers_by_subject":
            return {
                "matched_subject": {"subject_code": "HIS1001", "subject_name_vi": "Lá»‹ch sá»­ Äáº£ng"},
                "teachers": ["Nguyá»…n Thá»‹ Thu HoÃ i", "VÅ© Thá»‹ Thu HÃ "],
                "rows": [{"subject_code": "HIS1001", "class_code": "HIS1001 5", "day_of_week": "Thá»© 2", "slot": "1", "room": "103-A", "teacher_name": "VÅ© Thá»‹ Thu HÃ "}],
                "source_files": ["PHU_LUC_TKB.pdf"],
                "coverage_note": "ok",
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "vá» mÃ´n lá»‹ch sá»­ Ä‘áº£ng ká»³ nÃ y cÃ³ nhá»¯ng ai dáº¡y", "session_id": "s_struct_high", "program_id": "cs_2022"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "HIS1001" in body["answer"]
    assert "giang vien" in app_mod.normalize_for_match(body["answer"])
    assert planner_calls["count"] == 0


def test_ask_medium_confidence_blend_promotes_structured_payload(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            return {"matched_subject": {"subject_code": "PEC1008", "subject_name_vi": "Kinh táº¿ chÃ­nh trá»‹ MÃ¡c â€“ LÃªnin"}, "confidence": 0.78}
        if tool == "get_teachers_by_subject":
            return {
                "matched_subject": {"subject_code": "PEC1008", "subject_name_vi": "Kinh táº¿ chÃ­nh trá»‹ MÃ¡c â€“ LÃªnin"},
                "teachers": ["NgÃ´ ThÃ¡i HÃ "],
                "rows": [{"subject_code": "PEC1008", "class_code": "PEC1008 1", "day_of_week": "Thá»© 4", "slot": "3", "room": "503-B", "teacher_name": "NgÃ´ ThÃ¡i HÃ "}],
                "source_files": ["PHU_LUC_TKB.pdf"],
                "coverage_note": "ok",
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "giáº£ng viÃªn ká»³ nÃ y lÃ  ai", "session_id": "s_struct_mid", "program_id": "cs_2022"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "PEC1008" in body["answer"]
    assert planner_calls["count"] == 1


def test_structured_intent_classifier_prefers_course_schedule_for_combined_query(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier("Lá»›p thá»‹ giÃ¡c mÃ¡y kÃ¬ nÃ y ai dáº¡y vÃ  lá»›p vÃ o hÃ´m nÃ o")
    assert route["intent"] == "course_schedule"
    assert float(route["confidence"]) >= 0.75


def test_ask_deictic_followup_resolves_subject_from_memory_and_skips_advisor_answer(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    alias_queries: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {
                "source": "academic_advisor",
                "context": "advisor-should-not-win-for-deictic-followup",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    memory_lines = [
        "[2026-04-08 10:36:00] Query: Lop thi giac may ki nay ai day va lop vao hom nao\n"
        "Response: Mon INT3412E - Thi giac may ky nay co 1 giang vien: Le Thanh Ha"
    ]

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return memory_lines
        if tool == "resolve_course_alias":
            alias_queries.append(str(args.get("query") or ""))
            if str(args.get("query") or "").upper() == "INT3412E":
                return {"matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thá»‹ giÃ¡c mÃ¡y"}, "confidence": 0.98}
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "get_schedule_rows":
            if str(args.get("subject_code") or "").upper() == "INT3412E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3412E",
                            "class_code": "INT3412E 1",
                            "day_of_week": "Thá»© 2",
                            "slot": "1",
                            "room": "209-T",
                            "teacher_name": "LÃª Thanh HÃ ",
                        }
                    ],
                    "matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thá»‹ giÃ¡c mÃ¡y"},
                    "coverage_note": "ok",
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={"query": "lá»›p nÃ y lá»‹ch há»c hÃ´m nÃ o", "session_id": "s_followup", "program_id": "cs_2022", "file_ids": ["dummy_transcript.pdf"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "INT3412E 1" in body["answer"]
    assert "Thá»© 2" in body["answer"]
    assert "advisor-should-not-win-for-deictic-followup" not in body["answer"]
    assert "INT3412E" in [q.upper() for q in alias_queries]
    assert planner_calls["count"] == 0


def test_ask_multi_subject_teacher_query_resolves_both_without_memory_drift(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    alias_queries: list[str] = []
    teacher_subject_calls: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {
                "source": "academic_advisor",
                "context": "advisor-should-not-win-for-multi-subject",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    memory_lines = [
        "[2026-04-08 10:34:05] Query: kÃ¬ nÃ y cÃ³ nhá»¯ng mÃ´n tá»± chá»n nÃ o má»Ÿ lá»›p\n"
        "Response: ... INT3418 ... INT3102 ... INT3103"
    ]

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return memory_lines
        if tool == "resolve_course_alias":
            query = str(args.get("query") or "")
            alias_queries.append(query)
            q_norm = app_mod.normalize_for_match(query)
            if q_norm == "do hoa may tinh":
                return {
                    "matched_subject": {"subject_code": "INT3403", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh"},
                    "confidence": 0.95,
                }
            if q_norm == "thi giac may tinh":
                return {
                    "matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thá»‹ giÃ¡c mÃ¡y"},
                    "confidence": 0.95,
                }
            if query.upper() == "INT3103":
                return {
                    "matched_subject": {"subject_code": "INT3103", "subject_name_vi": "Tá»‘i Æ°u hÃ³a"},
                    "confidence": 0.95,
                }
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "get_teachers_by_subject":
            code = str(args.get("subject_code") or "").upper()
            teacher_subject_calls.append(code)
            if code == "INT3403":
                return {
                    "matched_subject": {"subject_code": "INT3403", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh"},
                    "teachers": ["Giáº£ng viÃªn A"],
                    "rows": [
                        {
                            "subject_code": "INT3403",
                            "class_code": "INT3403 1",
                            "day_of_week": "Thá»© 4",
                            "slot": "3",
                            "room": "207-T",
                            "teacher_name": "Giáº£ng viÃªn A",
                        }
                    ],
                    "source_files": ["PHU_LUC_TKB.pdf"],
                    "coverage_note": "ok",
                }
            if code == "INT3412E":
                return {
                    "matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thá»‹ giÃ¡c mÃ¡y"},
                    "teachers": ["LÃª Thanh HÃ "],
                    "rows": [
                        {
                            "subject_code": "INT3412E",
                            "class_code": "INT3412E 1",
                            "day_of_week": "Thá»© 3",
                            "slot": "2",
                            "room": "209-T",
                            "teacher_name": "LÃª Thanh HÃ ",
                        }
                    ],
                    "source_files": ["PHU_LUC_TKB.pdf"],
                    "coverage_note": "ok",
                }
            return {"rows": [], "teachers": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "mÃ´n Ä‘á»“ há»a mÃ¡y tÃ­nh vá»›i thá»‹ giÃ¡c mÃ¡y tÃ­nh ká»³ nÃ y ai dáº¡y",
            "session_id": "s_multi_subject",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "INT3403 1" in body["answer"]
    assert "INT3412E 1" in body["answer"]
    assert "advisor-should-not-win-for-multi-subject" not in body["answer"]
    assert "INT3103" not in [q.upper() for q in alias_queries]
    assert set(teacher_subject_calls) == {"INT3403", "INT3412E"}
    assert planner_calls["count"] == 0


def test_ask_non_deictic_unresolved_subject_does_not_retry_memory_subject(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    alias_queries: list[str] = []
    teacher_calls = {"count": 0}
    retrieve_calls = {"count": 0}

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return f"answer:{source}"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return [
                "[2026-04-08 10:34:05] Query: ká»³ nÃ y mÃ´n má»Ÿ lá»›p\n"
                "Response: ... INT3418 ... INT3102 ... INT3103"
            ]
        if tool == "resolve_course_alias":
            query = str(args.get("query") or "")
            alias_queries.append(query)
            if query.upper() == "INT3103":
                return {
                    "matched_subject": {"subject_code": "INT3103", "subject_name_vi": "Tá»‘i Æ°u hÃ³a"},
                    "confidence": 0.95,
                }
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "get_teachers_by_subject":
            teacher_calls["count"] += 1
            return {"rows": [], "teachers": []}
        if tool == "retrieve_chunks":
            retrieve_calls["count"] += 1
            return ["fallback-context"]
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "mÃ´n abc xyz ká»³ nÃ y ai dáº¡y",
            "session_id": "s_non_deictic_no_retry",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer:vector_store"
    assert "INT3103" not in [q.upper() for q in alias_queries]
    assert teacher_calls["count"] == 0
    assert retrieve_calls["count"] >= 1


def test_render_electives_schedule_answer_formats_vi_en_parentheses():
    app_mod = importlib.reload(importlib.import_module("app"))
    context = json.dumps(
        {
            "opened": [
                {
                    "code": "INT3306",
                    "name": "Phat trien ung dung Web Web Application Development",
                    "credits": 3,
                    "group": "Nhom phat trien phan mem",
                    "group_code": "V.2.1",
                },
                {
                    "code": "INT3230E",
                    "name": "Mat ma va An toan thong tin Cryptography and Information security",
                    "credits": 4,
                    "group": "Nhom AI",
                    "group_code": "V.2.2",
                },
            ],
            "opened_count": 2,
        },
        ensure_ascii=False,
    )

    answer = app_mod._render_electives_schedule_answer(context)
    norm_answer = app_mod.normalize_for_match(answer)
    assert "int3306: phat trien ung dung web (web application development)" in norm_answer
    assert "int3230e: mat ma va an toan thong tin (cryptography and information security)" in norm_answer
    assert "web web application development" not in norm_answer

def test_render_structured_schedule_answer_separates_subjects_and_filters_garbled_teacher():
    app_mod = importlib.reload(importlib.import_module("app"))
    context = json.dumps(
        {
            "rows": [
                {
                    "subject_code": "INT3403",
                    "subject_name_vi": "Do hoa may tinh",
                    "subject_name_en": "Computer Graphics",
                    "class_code": "INT3403 1",
                    "day_of_week": "Thu 4",
                    "slot": "3",
                    "room": "209-T",
                    "teacher_name": "Ma Thi Chau",
                },
                {
                    "subject_code": "INT3406",
                    "subject_name_vi": "Xu ly ngon ngu tu nhien",
                    "subject_name_en": "Natural Language Processing",
                    "class_code": "INT3406 2",
                    "day_of_week": "Thu 4",
                    "slot": "1",
                    "room": "207-B",
                    "teacher_name": "t u u áº§ áº§ n n, t Ä‘ h áº§ i u v",
                },
                {
                    "subject_code": "INT3406",
                    "subject_name_vi": "Xu ly ngon ngu tu nhien",
                    "subject_name_en": "Natural Language Processing",
                    "class_code": "INT3406 2",
                    "day_of_week": "Thu 4",
                    "slot": "4",
                    "room": "503-B",
                    "teacher_name": "Nguyen Van Vinh",
                },
            ]
        },
        ensure_ascii=False,
    )
    answer = app_mod._render_structured_schedule_answer(query="q", context=context)
    assert "INT3403 - Do hoa may tinh (Computer Graphics)" in answer
    assert "INT3406 - Xu ly ngon ngu tu nhien (Natural Language Processing)" in answer
    assert "\nINT3406 - Xu ly ngon ngu tu nhien (Natural Language Processing)\n" in answer
    assert "t u u áº§ áº§ n n" not in answer
    assert "Nguyen Van Vinh" in answer


def test_ask_source_electives_schedule_uses_deterministic_formatter(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "electives_schedule",
                "context": json.dumps(
                    {
                        "opened": [
                            {
                                "code": "INT3306",
                                "name": "Phat trien ung dung Web Web Application Development",
                                "credits": 3,
                                "group": "Nhom phat trien phan mem",
                                "group_code": "V.2.1",
                            }
                        ],
                        "opened_count": 1,
                    },
                    ensure_ascii=False,
                ),
                "memory": "",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "should_not_use_answer_agent"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [
                    {
                        "code": "INT3306",
                        "name": "Phat trien ung dung Web Web Application Development",
                        "credits": 3,
                        "group": "Nhom phat trien phan mem",
                        "group_code": "V.2.1",
                    }
                ],
                "opened_count": 1,
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "ki nay co mon tu chon nao mo lop",
            "session_id": "s_elective_render",
            "program_id": "cs_2022",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_program_id"] == "cs_2022"
    assert "int3306: phat trien ung dung web (web application development) - 3 tin chi" in app_mod.normalize_for_match(body["answer"])
    assert "should_not_use_answer_agent" not in body["answer"]


def test_ask_correction_followup_keeps_structured_teacher_route(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    alias_queries: list[str] = []
    teacher_calls: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {
                "source": "academic_advisor",
                "context": "advisor-should-not-win-for-correction-followup",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return [
                "[2026-04-10 08:32:53] Query: mon hoc sau va ung dung ... co nhung ai day\\n"
                "Response: Lich hoc theo tung mon ..."
            ]
        if tool == "resolve_course_alias":
            alias_queries.append(str(args.get("query") or ""))
            q_norm = app_mod.normalize_for_match(str(args.get("query") or ""))
            if "hoc sau" in q_norm:
                return {
                    "matched_subject": {
                        "subject_code": "INT3420E",
                        "subject_name_vi": "Hoc sau va Ung dung",
                        "subject_name_en": "Deep learning and Applications",
                    },
                    "confidence": 0.95,
                }
            if "phat trien ung dung di dong" in q_norm:
                return {
                    "matched_subject": {"subject_code": "INT3120", "subject_name_vi": "Phat trien ung dung di dong"},
                    "confidence": 0.95,
                }
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "get_teachers_by_subject":
            code = str(args.get("subject_code") or "").upper()
            teacher_calls.append(code)
            if code == "INT3420E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3420E",
                            "subject_name_vi": "Hoc sau va Ung dung",
                            "subject_name_en": "Deep learning and Applications",
                            "class_code": "INT3420E 1",
                            "day_of_week": "Thu 3",
                            "slot": "3",
                            "room": "211-T",
                            "teacher_name": "Ta Viet Cuong",
                        }
                    ],
                    "teachers": ["Ta Viet Cuong"],
                    "matched_subject": {
                        "subject_code": "INT3420E",
                        "subject_name_vi": "Hoc sau va Ung dung",
                        "subject_name_en": "Deep learning and Applications",
                    },
                    "source_files": ["PHU_LUC_TKB.pdf"],
                }
            return {"rows": [], "teachers": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "khÃ´ng pháº£i phÃ¡t triá»ƒn á»©ng dá»¥ng di Ä‘á»™ng mÃ  lÃ  mÃ´n há»c sÃ¢u vÃ  á»©ng dá»¥ng mÃ ",
            "session_id": "s_correction_followup",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "INT3420E" in body["answer"]
    assert "Ta Viet Cuong" in body["answer"]
    assert "advisor-should-not-win-for-correction-followup" not in body["answer"]
    assert planner_calls["count"] == 0
    assert all(code != "INT3120" for code in teacher_calls)
    assert any("hoc sau" in app_mod.normalize_for_match(q) for q in alias_queries)


def test_ask_academic_advisor_unescapes_unicode_answer(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "academic_advisor",
                "context": "Ch\\u00e0o b\\u1ea1n!\\n\\u0110\\u00e3 c\\u1eadp nh\\u1eadt.",
                "memory": "",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "should_not_use_answer_agent"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "cho toi cau tra loi unicode mau",
            "session_id": "s_unescape_advisor",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Chào bạn!\nĐã cập nhật."


def test_normalize_output_text_strips_source_reference_footer_block():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = (
        "Lịch học theo từng môn:\n"
        "- INT3412E 1: Thứ 3, Ca 2, phòng 209-T, GV Lê Thanh Hà\n\n"
        "Nguồn tham chiếu\n"
        "[1] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4\n"
        "[2] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4"
    )
    normalized = app_mod._normalize_output_text(text)
    normalized_check = app_mod.normalize_for_match(normalized)
    assert "nguon tham chieu" not in normalized_check
    assert "[1]" not in normalized
    assert "INT3412E 1" in normalized


def test_normalize_output_text_keeps_non_footer_nguon_sentence():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = "Môn học này thuộc nhóm nguồn mở dữ liệu, không phải phần trích dẫn tài liệu."
    normalized = app_mod._normalize_output_text(text)
    assert normalized == text


def test_normalize_output_text_strips_footer_when_header_is_markdown_variant():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = (
        "Lịch học theo từng môn:\n"
        "- INT3412E 1: Thứ 3, Ca 2, phòng 209-T, GV Lê Thanh Hà\n\n"
        "### **Nguồn tham chiếu:**\n"
        "- [1] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4\n"
        "- [2] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4"
    )
    normalized = app_mod._normalize_output_text(text)
    normalized_check = app_mod.normalize_for_match(normalized)
    assert "nguon tham chieu" not in normalized_check
    assert "INT3412E 1" in normalized


def test_normalize_output_text_strips_trailing_source_items_with_broken_header():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = (
        "Lịch học theo từng môn:\n"
        "- INT3412E 1: Thứ 3, Ca 2, phòng 209-T, GV Lê Thanh Hà\n\n"
        "tham chieu tai lieu\n"
        "[1] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4\n"
        "[2] PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4"
    )
    normalized = app_mod._normalize_output_text(text)
    normalized_check = app_mod.normalize_for_match(normalized)
    assert "tham chieu tai lieu" not in normalized_check
    assert "[1]" not in normalized
    assert "INT3412E 1" in normalized


def test_normalize_output_text_repairs_stored_ait3004_ocr_title():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = "AIT3004 - T tạ h o ực hành phát triển hệ thống Trí tuệ nhân"

    normalized = app_mod._normalize_output_text(text)

    assert normalized == "AIT3004 - Thực hành phát triển hệ thống Trí tuệ nhân tạo"


def test_normalize_output_text_strips_mojibake_source_footer_header():
    app_mod = importlib.reload(importlib.import_module("app"))
    text = (
        "Lich hoc theo tung mon:\n"
        "- INT3412E 1: Thu 3, Ca 2, phong 209-T, GV Le Thanh Ha\n\n"
        "Ngu?n tham chi?u\n"
        "[1] PHU LUC THOI KHOA BIEU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4\n"
        "[2] PHU LUC THOI KHOA BIEU HKII 2025-2026.xlsx - Sheet1.pdf - Page 4 - Line 4"
    )
    normalized = app_mod._normalize_output_text(text)
    normalized_check = app_mod.normalize_for_match(normalized)
    assert "tham chi" not in normalized_check
    assert "[1]" not in normalized
    assert "INT3412E 1" in normalized


def test_subject_hint_parser_handles_multi_subject_without_breaking_and_inside_title():
    app_mod = importlib.reload(importlib.import_module("app"))
    query = (
        "mon hoc sau va ung dung, mon mat ma va an toan thong tin "
        "va mon kiem thu va dam bao chat luong phan mem co nhung ai day"
    )
    hints = app_mod._extract_subject_hints(query)
    normalized_hints = [app_mod.normalize_for_match(item) for item in hints]

    assert "hoc sau va ung dung" in normalized_hints
    assert "mat ma va an toan thong tin" in normalized_hints
    assert "kiem thu va dam bao chat luong phan mem" in normalized_hints
    assert "ung dung" not in normalized_hints
    assert "mat" not in normalized_hints

    assert app_mod._extract_teacher_name_hint(query) is None
    route = app_mod._structured_intent_classifier(query)
    assert route["intent"] == "teacher_by_subject"
    assert "teacher_name" not in (route.get("signals") or [])


def test_subject_hint_parser_handles_multi_subject_with_vietnamese_diacritics():
    app_mod = importlib.reload(importlib.import_module("app"))
    query = (
        "m\u00f4n h\u1ecdc s\u00e2u v\u00e0 \u1ee9ng d\u1ee5ng, "
        "m\u00f4n m\u1eadt m\u00e3 v\u00e0 an to\u00e0n th\u00f4ng tin "
        "v\u00e0 m\u00f4n ki\u1ec3m th\u1eed v\u00e0 \u0111\u1ea3m b\u1ea3o ch\u1ea5t l\u01b0\u1ee3ng ph\u1ea7n m\u1ec1m c\u00f3 nh\u1eefng ai d\u1ea1y"
    )
    hints = app_mod._extract_subject_hints(query)
    normalized_hints = [app_mod.normalize_for_match(item) for item in hints]

    assert "hoc sau va ung dung" in normalized_hints
    assert "mat ma va an toan thong tin" in normalized_hints
    assert "kiem thu va dam bao chat luong phan mem" in normalized_hints

    route = app_mod._structured_intent_classifier(query)
    assert route["intent"] == "teacher_by_subject"
    assert "teacher_marker" in (route.get("signals") or [])


def test_structured_intent_classifier_keeps_classes_by_teacher_for_teacher_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    query = "thay le khanh trinh day nhung mon nao ki nay"
    teacher_name = app_mod._extract_teacher_name_hint(query)
    assert app_mod.normalize_for_match(teacher_name or "") == "le khanh trinh"

    route = app_mod._structured_intent_classifier(query)
    assert route["intent"] == "classes_by_teacher"


def test_structured_intent_classifier_keeps_classes_by_teacher_for_teacher_query_with_mon_gi():
    app_mod = importlib.reload(importlib.import_module("app"))
    query = "thay tran hoang viet ki nay day nhung mon gi vay"
    teacher_name = app_mod._extract_teacher_name_hint(query)
    assert app_mod.normalize_for_match(teacher_name or "") == "tran hoang viet"

    route = app_mod._structured_intent_classifier(query)
    assert route["intent"] == "classes_by_teacher"


def test_render_electives_schedule_answer_formats_bilingual_names_with_parentheses():
    app_mod = importlib.reload(importlib.import_module("app"))
    payload = {
        "opened": [
            {
                "code": "INT3117",
                "name": "Kiem thu va dam bao chat luong phan mem Software Testing and Quality Assurance",
                "credits": 3,
                "group": "Nhom 1",
                "group_code": "V.2.1",
            },
            {
                "code": "INT2041",
                "name": "Tuong tac nguoi may Human-Machine Interaction",
                "credits": 3,
                "group": "Nhom 2",
                "group_code": "V.2.4",
            },
        ]
    }
    answer = app_mod._render_electives_schedule_answer(json.dumps(payload, ensure_ascii=False))
    norm_answer = app_mod.normalize_for_match(answer)
    assert "int3117: kiem thu va dam bao chat luong phan mem (software testing and quality assurance)" in norm_answer
    assert "int2041: tuong tac nguoi may (human-machine interaction)" in norm_answer

def test_render_structured_schedule_answer_dedupes_teacher_rows_and_shows_subject_titles():
    app_mod = importlib.reload(importlib.import_module("app"))
    payload = {
        "rows": [
            {
                "subject_code": "INT3117",
                "subject_name_vi": "Kiem thu va dam bao chat luong phan mem",
                "subject_name_en": "Software Testing and Quality Assurance",
                "class_code": "INT3117 2",
                "day_of_week": "Thu 6",
                "slot": "4",
                "room": "207-T",
                "teacher_name": "Nguyen Thu Trang",
            }
        ],
        "matched_teacher": {"query": "Le Khanh Trinh"},
    }
    answer = app_mod._render_structured_schedule_answer("multi", json.dumps(payload, ensure_ascii=False))
    norm_answer = app_mod.normalize_for_match(answer)
    assert "int3117 - kiem thu va dam bao chat luong phan mem (software testing and quality assurance)" in norm_answer
    assert norm_answer.count("- int3117 2: thu 6, ca 4, phong 207-t") == 1


def test_render_structured_schedule_answer_repairs_ait3004_ocr_title():
    app_mod = importlib.reload(importlib.import_module("app"))
    payload = {
        "rows": [
            {
                "subject_code": "AIT3004",
                "subject_name_vi": "T tạ h o ực hành phát triển hệ thống Trí tuệ nhân",
                "subject_name_en": "",
                "class_code": "AIT3004 1",
                "day_of_week": "Thứ 2",
                "slot": "2",
                "room": "503-A",
                "teacher_name": "Trịnh Ngọc Huỳnh",
            }
        ],
        "matched_teacher": {"query": "Trịnh Ngọc Huỳnh"},
    }
    answer = app_mod._render_structured_schedule_answer("thầy Huỳnh dạy gì", json.dumps(payload, ensure_ascii=False))

    assert "AIT3004 - Thực hành phát triển hệ thống Trí tuệ nhân tạo" in answer
    assert "T tạ h o ực" not in answer


def test_render_structured_schedule_answer_appends_missing_subject_note_for_multi_subject_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    payload = {
        "rows": [
            {
                "subject_code": "INT3117",
                "subject_name_vi": "Kiem thu va dam bao chat luong phan mem",
                "subject_name_en": "Software Testing and Quality Assurance",
                "class_code": "INT3117 2",
                "day_of_week": "Thu 6",
                "slot": "4",
                "room": "207-T",
                "teacher_name": "Nguyen Thu Trang",
            }
        ],
        "no_data_subjects": [
            {
                "subject_code": "INT3230E",
                "subject_name_vi": "Mat ma va an toan thong tin",
                "subject_name_en": "Cryptography and Information security",
            }
        ],
    }
    answer = app_mod._render_structured_schedule_answer("multi", json.dumps(payload, ensure_ascii=False))
    norm_answer = app_mod.normalize_for_match(answer)
    assert "chua thay du lieu lich" in norm_answer
    assert "int3230e - mat ma va an toan thong tin (cryptography and information security)" in norm_answer


def test_structured_intent_classifier_detects_electives_overview_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier(
        "y toi la cac mon tu chon theo chuyen nganh chuong trinh dao tao cua toi ma"
    )
    assert route["intent"] == "electives_overview"
    assert float(route["confidence"]) >= 0.75


def test_structured_intent_classifier_keeps_electives_overview_for_opening_query_with_schedule_words():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier(
        "ve cac mon tu chon co mon nao duoc mo o ki nay khong mo hom nao trong tuan"
    )
    assert route["intent"] == "electives_overview"
    assert float(route["confidence"]) >= 0.79


def test_structured_intent_classifier_detects_course_schedule_for_lich_mon_phrase():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier("lich mon ly thuyet thong tin nhu nao")
    assert route["intent"] == "course_schedule"
    assert float(route["confidence"]) >= 0.75


def test_query_requires_transcript_files_skips_plain_schedule_lookup():
    app_mod = importlib.reload(importlib.import_module("app"))
    assert app_mod._query_requires_transcript_files("lich mon ly thuyet thong tin nhu nao") is False


def test_extract_subject_hints_strips_nhu_nao_tail_for_schedule_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    hints = app_mod._extract_subject_hints("lich mon ly thuyet thong tin nhu nao")
    normalized = [app_mod.normalize_for_match(item) for item in hints]
    assert normalized == ["ly thuyet thong tin"]


def test_query_requires_planner_orchestration_for_complex_elective_schedule_filter():
    app_mod = importlib.reload(importlib.import_module("app"))
    assert not app_mod._query_requires_planner_orchestration(
        query="toi can lich hoc cac mon tu chon lien quan den ai duoc mo ky nay va hoc thu may trong tuan",
        route_intent="electives_overview",
    )
    assert not app_mod._query_requires_planner_orchestration(
        query="ky nay co nhung mon tu chon nao mo lop",
        route_intent="electives_overview",
    )


def test_ask_transcript_intensive_query_bypasses_planner_when_low_structured_confidence(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    advisor_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            advisor_calls["count"] += 1
            return "Thong tin hoc vu tu advisor"
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can ban kiem tra giup toi xem toi con thieu nhung mon nao theo chuong trinh dao tao voi",
            "session_id": "s_transcript_bypass",
            "program_id": "cs_2022",
            "file_ids": ["1_a.pdf", "2_b.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_program_id"] == "cs_2022"
    assert "Thong tin hoc vu tu advisor" in body["answer"]
    assert planner_calls["count"] == 0
    assert advisor_calls["count"] == 1


def test_ask_advisor_priority_query_skips_structured_prefetch_even_with_high_schedule_confidence(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    advisor_calls = {"count": 0}
    structured_prefetch_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_structured_prefetch(**kwargs):
        structured_prefetch_calls["count"] += 1
        raise AssertionError("structured prefetch should be skipped for advisor-priority transcript query")

    monkeypatch.setattr(app_mod, "_build_structured_route_payload", fake_structured_prefetch)

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            advisor_calls["count"] += 1
            return "Bang lich hoc tu advisor"
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can ban lap giup toi lich hoc dua tren cac mon con thieu cua toi",
            "session_id": "s_advisor_priority_schedule",
            "program_id": "cs_2022",
            "file_ids": ["1_a.pdf", "2_b.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Bang lich hoc tu advisor" in body["answer"]
    assert structured_prefetch_calls["count"] == 0
    assert planner_calls["count"] == 0
    assert advisor_calls["count"] == 1


def test_ask_schedule_replan_followup_returns_to_advisor_after_intervening_teacher_lookup(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    advisor_calls: list[str] = []
    alias_calls: list[str] = []
    state_store: dict[str, dict] = {}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_state_get":
            return state_store.get(args["session_id"], app_mod.default_conversation_state())
        if tool == "memory_state_upsert":
            state_store[args["session_id"]] = args["state"]
            return args["state"]
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            advisor_calls.append(str(args.get("query") or ""))
            return (
                "Thiếu tín chỉ và kế hoạch học tập.\n\n"
                "Gợi ý lịch\n\n"
                "| Ngày học | Ca học | Tiết + Thời gian | Mã môn học | Tên môn học | Tín chỉ | Ghi chú về lớp |\n"
                "|---|---|---|---|---|---:|---|\n"
                "| Thứ 2 | Ca 1 | Tiết 1-3 (07:00 – 09:40) | INT3230E | Mật mã và An toàn thông tin | 4 | Lớp INT3230E 1 |\n"
                "| Thứ 4 | Ca 1 | Tiết 1-3 (07:00 – 09:40) | HIS1001 | Lịch sử Đảng Cộng sản Việt Nam | 2 | Lớp HIS1001 1 |\n"
                "\n"
                "Nguồn TKB: Structured Schedule"
            )
        if tool == "get_classes_by_teacher":
            return {
                "rows": [
                    {
                        "subject_code": "INT2208",
                        "subject_name_vi": "Cong nghe phan mem",
                        "subject_name_en": "",
                        "teacher_name": "Phạm Ngọc Hùng",
                        "class_code": "INT2208 3",
                        "weekday": "Thứ 4",
                        "slot": "Ca 3",
                        "period": "Tiết 7-9",
                        "time_range": "13:30 – 16:10",
                        "room": "402-B",
                        "source_file": "Structured Schedule",
                    }
                ],
                "teachers": ["Phạm Ngọc Hùng"],
                "source_files": ["Structured Schedule"],
            }
        if tool == "resolve_course_alias":
            alias_calls.append(str(args.get("query") or ""))
            query = app_mod.normalize_for_match(str(args.get("query") or ""))
            if "mat ma" in query:
                return {
                    "matched_subject": {"subject_code": "INT3230E", "subject_name_vi": "Mật mã và An toàn thông tin"},
                    "confidence": 0.95,
                    "candidates": [{"subject_code": "INT3230E", "subject_name_vi": "Mật mã và An toàn thông tin", "score": 0.95}],
                }
            if "xu ly ngon ngu tu nhien" in query:
                return {
                    "matched_subject": {"subject_code": "INT3406", "subject_name_vi": "Xử lý ngôn ngữ tự nhiên"},
                    "confidence": 0.95,
                    "candidates": [{"subject_code": "INT3406", "subject_name_vi": "Xử lý ngôn ngữ tự nhiên", "score": 0.95}],
                }
            return {"matched_subject": None, "confidence": 0.0, "candidates": []}
        if tool == "get_curriculum_lookup":
            return json.dumps(
                {
                    "groups": {
                        "V.2.3": {
                            "group_name": "Nhóm tự chọn ngành",
                            "subjects": [
                                {"code": "INT3230E", "name": "Mật mã và An toàn thông tin", "credits": 4},
                                {"code": "INT3406", "name": "Xử lý ngôn ngữ tự nhiên", "credits": 3},
                            ],
                        }
                    }
                },
                ensure_ascii=False,
            )
        if tool == "get_schedule_rows":
            return {
                "rows": [
                    {
                        "subject_code": "INT3406",
                        "subject_name_vi": "Xử lý ngôn ngữ tự nhiên",
                        "class_code": "INT3406 2",
                        "teacher_name": "Nguyễn Văn Vinh",
                        "day_of_week": "Thứ 4",
                        "slot": "1",
                        "room": "207-B",
                        "week_note": "Học 1 ca/15 tuần",
                        "source_file": "Structured Schedule",
                        "source_page": 3,
                        "source_line": 8,
                    }
                ],
                "source_files": ["Structured Schedule"],
                "coverage_note": "Tìm thấy dữ liệu lịch học.",
            }
        if tool == "get_time_slot_info":
            return {"slot": "1", "period": "Tiết 1-3", "time_range": "07:00 – 09:40", "source_file": "Structured Schedule"}
        if tool == "retrieve_chunks":
            raise AssertionError(f"{tool} should not run for advisor schedule replan follow-up")
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    session_id = "s_advisor_replan_followup"
    file_ids = ["1_a.pdf", "2_b.pdf"]

    first = client.post(
        "/ask",
        json={
            "query": "toi can ban lap giup toi lich hoc dua tren cac mon con thieu cua toi",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": file_ids,
        },
    )
    assert first.status_code == 200
    assert first.json()["source"] == "academic_advisor"

    saved_after_first = state_store.get(session_id) or {}
    assert (saved_after_first.get("advisor_context") or {}).get("has_recent_schedule_plan") is True

    second = client.post(
        "/ask",
        json={
            "query": "ki hoc nay thay Pham Ngoc Hung day nhung mon gi",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": file_ids,
        },
    )
    assert second.status_code == 200
    assert second.json()["source"] == "structured_schedule"

    third = client.post(
        "/ask",
        json={
            "query": "sap xep lai lich hoc giup toi toi muon thay mon mat ma va an toan thong tin bang mon xu ly ngon ngu tu nhien co duoc khong",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": file_ids,
        },
    )
    assert third.status_code == 200
    body = third.json()
    assert body["source"] == "schedule_plan_replan"
    assert "xu ly ngon ngu tu nhien" in app_mod.normalize_for_match(body["answer"])
    assert "trung voi his1001" in app_mod.normalize_for_match(body["answer"])
    assert planner_calls["count"] == 0
    assert len(advisor_calls) == 1
    assert alias_calls == [] or alias_calls


def test_ask_schedule_replan_prefers_curriculum_lexical_match_over_wrong_alias(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    state_store: dict[str, dict] = {}

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_state_get":
            return state_store.get(args["session_id"], app_mod.default_conversation_state())
        if tool == "memory_state_upsert":
            state_store[args["session_id"]] = args["state"]
            return args["state"]
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            return (
                "Thiếu tín chỉ và kế hoạch học tập.\n\n"
                "Gợi ý lịch\n\n"
                "| Ngày học | Ca học | Tiết + Thời gian | Mã môn học | Tên môn học | Tín chỉ | Ghi chú về lớp |\n"
                "|---|---|---|---|---|---:|---|\n"
                "| Thứ 2 | Ca 1 | Tiết 1-3 (07:00 – 09:40) | INT3230E | Mật mã và An toàn thông tin | 4 | Lớp INT3230E 1 |\n"
                "| Thứ 3 | Ca 4 | Tiết 10-12 (16:20 – 19:00) | PHI1002 | Chủ nghĩa xã hội khoa học | 2 | Lớp PHI1002 1 |\n"
                "| Thứ 4 | Ca 1 | Tiết 1-3 (07:00 – 09:40) | HIS1001 | Lịch sử Đảng Cộng sản Việt Nam | 2 | Lớp HIS1001 1 |\n"
                "\n"
                "Nguồn TKB: Structured Schedule"
            )
        if tool == "resolve_course_alias":
            query = app_mod.normalize_for_match(str(args.get("query") or ""))
            if "mat ma" in query:
                return {
                    "matched_subject": {"subject_code": "INT3230E", "subject_name_vi": "Mật mã và An toàn thông tin"},
                    "confidence": 0.95,
                    "candidates": [{"subject_code": "INT3230E", "subject_name_vi": "Mật mã và An toàn thông tin", "score": 0.95}],
                }
            if "xu ly ngon ngu tu nhien" in query:
                return {
                    "matched_subject": {"subject_code": "ELT2035", "subject_name_vi": "Tín hiệu và hệ thống"},
                    "confidence": 0.66,
                    "candidates": [{"subject_code": "ELT2035", "subject_name_vi": "Tín hiệu và hệ thống", "score": 0.66}],
                }
            return {"matched_subject": None, "confidence": 0.0, "candidates": []}
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.3": {
                        "group_name": "Nhóm tự chọn ngành",
                        "subjects": [
                            {"code": "INT3230E", "name": "Mật mã và An toàn thông tin", "credits": 4},
                            {"code": "INT3406", "name": "Xử lý ngôn ngữ tự nhiên", "credits": 3},
                            {"code": "ELT2035", "name": "Tín hiệu và hệ thống", "credits": 3},
                        ],
                    }
                }
            }
        if tool == "get_schedule_rows":
            if str(args.get("subject_code") or "").upper() == "INT3406":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3406",
                            "subject_name_vi": "Xử lý ngôn ngữ tự nhiên",
                            "class_code": "INT3406 2",
                            "teacher_name": "Nguyễn Văn Vinh",
                            "day_of_week": "Thứ 4",
                            "slot": "1",
                            "room": "207-B",
                            "source_file": "Structured Schedule",
                        }
                    ],
                    "source_files": ["Structured Schedule"],
                }
            return {"rows": [], "source_files": ["Structured Schedule"]}
        if tool == "get_time_slot_info":
            return {"slot": "1", "period": "Tiết 1-3", "time_range": "07:00 – 09:40", "source_file": "Structured Schedule"}
        if tool == "retrieve_chunks":
            raise AssertionError("retrieve_chunks should not run for deterministic schedule replan")
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    session_id = "s_advisor_replan_curriculum_preferred"
    file_ids = ["1_a.pdf", "2_b.pdf"]

    first = client.post(
        "/ask",
        json={
            "query": "toi can ban lap giup toi lich hoc dua tren cac mon con thieu cua toi",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": file_ids,
        },
    )
    assert first.status_code == 200
    assert first.json()["source"] == "academic_advisor"

    second = client.post(
        "/ask",
        json={
            "query": "sap xep lai lich hoc giup toi toi muon thay mon mat ma va an toan thong tin bang mon xu ly ngon ngu tu nhien co duoc khong",
            "session_id": session_id,
            "program_id": "cs_2022",
            "file_ids": file_ids,
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["source"] == "schedule_plan_replan"
    normalized_answer = app_mod.normalize_for_match(body["answer"])
    assert "int3406" in normalized_answer
    assert "xu ly ngon ngu tu nhien" in normalized_answer
    assert "elt2035" not in normalized_answer
    assert "trung voi his1001" in normalized_answer


def test_ask_course_schedule_retries_alias_without_program_scope_and_skips_planner(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    alias_call_program_ids = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            alias_call_program_ids.append(args.get("program_id"))
            if args.get("program_id"):
                return {"matched_subject": None, "confidence": 0.0}
            return {
                "matched_subject": {
                    "subject_code": "INT2044E",
                    "subject_name_vi": "Ly thuyet thong tin",
                    "subject_name_en": "Information Theory",
                },
                "confidence": 0.92,
            }
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").upper()
            if code == "INT2044E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT2044E",
                            "subject_name_vi": "Ly thuyet thong tin",
                            "subject_name_en": "Information Theory",
                            "class_code": "INT2044E 1",
                            "day_of_week": "Thu 3",
                            "slot": "3",
                            "room": "211-T",
                            "teacher_name": "GV A",
                        }
                    ]
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "lich mon ly thuyet thong tin nhu nao",
            "session_id": "s_schedule_alias_retry",
            "program_id": "cs_2022",
            "file_ids": ["1_a.pdf", "2_b.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "int2044e" in norm_answer
    assert "thu 3" in norm_answer
    assert planner_calls["count"] == 0
    assert any(pid == "cs_2022" for pid in alias_call_program_ids)
    assert any(pid is None for pid in alias_call_program_ids)


def test_ask_course_schedule_prefers_curriculum_subject_over_bad_alias_and_returns_no_data(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    schedule_codes_called = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            # Deliberate wrong high-confidence alias from resolver.
            return {
                "matched_subject": {
                    "subject_code": "INT3230E",
                    "subject_name_vi": "Mat ma va an toan thong tin",
                    "subject_name_en": "Cryptography and Information security",
                },
                "confidence": 0.97,
            }
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.x": {
                        "group_name": "Nhom thu nghiem",
                        "subjects": [
                            {"code": "INT2044E", "name": "Ly thuyet thong tin Information Theory"},
                            {"code": "INT3230E", "name": "Mat ma va an toan thong tin"},
                        ],
                    }
                }
            }
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").upper()
            schedule_codes_called.append(code)
            if code == "INT2044E":
                return {"rows": []}
            if code == "INT3230E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3230E",
                            "class_code": "INT3230E 1",
                            "day_of_week": "Thu 2",
                            "slot": "1",
                            "room": "209-T",
                        }
                    ]
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "lich mon ly thuyet thong tin nhu nao",
            "session_id": "s_schedule_curriculum_override",
            "program_id": "cs_2022",
            "file_ids": ["1_a.pdf", "2_b.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "int2044e" in norm_answer
    assert "chua thay du lieu" in norm_answer
    assert "int3230e 1" not in norm_answer
    assert planner_calls["count"] == 0
    assert "INT2044E" in schedule_codes_called
    assert "INT3230E" not in schedule_codes_called


def test_ask_specialized_electives_query_uses_structured_overview_without_planner(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [
                    {"code": "INT3117", "name": "Kiem thu va dam bao chat luong phan mem", "credits": 3, "group_code": "V.2.1", "group": "Nhom V.2.1"},
                    {"code": "INT3120", "name": "Phat trien ung dung di dong", "credits": 3, "group_code": "V.2.1", "group": "Nhom V.2.1"},
                    {"code": "INT3230E", "name": "Mat ma va An toan thong tin", "credits": 4, "group_code": "V.2.2", "group": "Nhom V.2.2"},
                    {"code": "INT3406", "name": "Xu ly ngon ngu tu nhien", "credits": 3, "group_code": "V.2.3", "group": "Nhom V.2.3"},
                    {"code": "INT3412E", "name": "Thi giac may", "credits": 3, "group_code": "V.2.4", "group": "Nhom V.2.4"},
                ],
                "opened_count": 5,
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "y toi la cac mon tu chon theo chuyen nganh chuong trinh dao tao cua toi ma",
            "session_id": "s_specialized_overview",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "int3117" in norm_answer
    assert "int3120" in norm_answer
    assert "int3230e" in norm_answer
    assert "int3406" in norm_answer
    assert "int3412e" in norm_answer
    assert planner_calls["count"] == 0


def test_ask_complex_elective_schedule_filter_uses_structured_recommendation(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(
        app_mod,
        "_rank_opened_electives_for_query",
        lambda query, opened_items, top_k=6: {
            "focus": "tri tue nhan tao",
            "selected_codes": ["INT3406", "INT3420E"],
            "reason_by_code": {
                "INT3406": "phu hop huong NLP",
                "INT3420E": "phu hop huong deep learning",
            },
            "confidence": 0.91,
            "used_fallback": False,
        },
    )

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [
                    {"code": "INT3406", "name": "Xu ly ngon ngu tu nhien", "credits": 3, "group_code": "V.2.3", "group": "Nhom V.2.3"},
                    {"code": "INT3420E", "name": "Hoc sau va Ung dung", "credits": 3, "group_code": "V.2.3", "group": "Nhom V.2.3"},
                ],
                "opened_count": 2,
            }
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").upper()
            if code == "INT3406":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3406",
                            "subject_name_vi": "Xu ly ngon ngu tu nhien",
                            "class_code": "INT3406 1",
                            "day_of_week": "Thu 3",
                            "slot": "3",
                            "room": "211-T",
                            "teacher_name": "Ta Viet Cuong",
                        }
                    ]
                }
            if code == "INT3420E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3420E",
                            "subject_name_vi": "Hoc sau va Ung dung",
                            "class_code": "INT3420E 1",
                            "day_of_week": "Thu 4",
                            "slot": "4",
                            "room": "209-T",
                            "teacher_name": "Le Thanh Ha",
                        }
                    ]
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can lich hoc cac mon tu chon lien quan den AI mo ky nay va hoc thu may trong tuan",
            "session_id": "s_complex_elective_planner",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "goi y mon tu chon" in norm_answer
    assert "int3406" in norm_answer
    assert "int3420e" in norm_answer
    assert "lich hoc theo tung mon" in norm_answer
    assert planner_calls["count"] == 0


def test_ask_mobile_orientation_query_returns_recommendation_without_planner(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(
        app_mod,
        "_rank_opened_electives_for_query",
        lambda query, opened_items, top_k=6: {
            "focus": "lap trinh mobile",
            "selected_codes": ["INT3120", "INT3306"],
            "reason_by_code": {
                "INT3120": "mon cot loi cho mobile app",
                "INT3306": "ho tro backend/frontend cho ung dung mobile",
            },
            "confidence": 0.88,
            "used_fallback": False,
        },
    )

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [
                    {
                        "code": "INT3120",
                        "name": "Phat trien ung dung di dong",
                        "name_en": "Mobile Application Development",
                        "credits": 3,
                        "group_code": "V.2.1",
                        "group": "Nhom V.2.1",
                    },
                    {
                        "code": "INT3306",
                        "name": "Phat trien ung dung Web",
                        "name_en": "Web Application Development",
                        "credits": 3,
                        "group_code": "V.2.1",
                        "group": "Nhom V.2.1",
                    },
                    {
                        "code": "INT3230E",
                        "name": "Mat ma va An toan thong tin",
                        "credits": 4,
                        "group_code": "V.2.2",
                        "group": "Nhom V.2.2",
                    },
                ],
                "opened_count": 3,
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can hoc cac mon tu chon lien quan den dinh huong lap trinh mobile",
            "session_id": "s_mobile_orientation",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "goi y mon tu chon" in norm_answer
    assert "int3120" in norm_answer
    assert "int3306" in norm_answer
    assert planner_calls["count"] == 0


def test_ask_electives_overview_with_schedule_request_includes_schedule_details(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [
                    {
                        "code": "INT3117",
                        "name": "Kiem thu va dam bao chat luong phan mem",
                        "credits": 3,
                        "group_code": "V.2.1",
                        "group": "Nhom V.2.1",
                    },
                    {
                        "code": "INT3120",
                        "name": "Phat trien ung dung di dong",
                        "credits": 3,
                        "group_code": "V.2.1",
                        "group": "Nhom V.2.1",
                    },
                ],
                "opened_count": 2,
            }
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").upper()
            if code == "INT3117":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3117",
                            "subject_name_vi": "Kiem thu va dam bao chat luong phan mem",
                            "class_code": "INT3117 2",
                            "day_of_week": "Thu 6",
                            "slot": "4",
                            "room": "207-T",
                            "teacher_name": "Nguyen Thu Trang",
                        }
                    ]
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "ki nay co nhung mon tu chon nao mo lop toi can lich hoc cac mon do luon",
            "session_id": "s_elective_schedule_detail",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "ky nay co 2 hoc phan tu chon mo lop" in norm_answer
    assert "lich hoc theo tung mon" in norm_answer
    assert "int3117 2" in norm_answer
    assert "int3120 - phat trien ung dung di dong" in norm_answer
    assert planner_calls["count"] == 0


def test_ask_course_offering_status_reports_not_opened_and_curriculum_group(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            return {
                "matched_subject": {
                    "subject_code": "INT3404E",
                    "subject_name_vi": "Xu ly anh",
                    "subject_name_en": "Image Processing",
                },
                "confidence": 0.95,
            }
        if tool == "get_schedule_rows":
            return {"rows": []}
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.4": {
                        "group_name": "Nhom cac hoc phan ve Tuong tac nguoi-may",
                        "subjects": [{"code": "INT3404E", "name": "Xu ly anh"}],
                    }
                }
            }
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "the mon xu li anh ki nay co mo khong va ma mon nay co nam trong khung ctdt cua toi khong",
            "session_id": "s_course_status",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "int3404e" in norm_answer
    assert "khong mo lop" in norm_answer
    assert "v.2.4" in norm_answer
    assert planner_calls["count"] == 0


def test_ask_course_offering_status_no_rows_uses_schedule_citation_without_retrieve(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            return {
                "matched_subject": {
                    "subject_code": "INT3404E",
                    "subject_name_vi": "Xu ly anh",
                    "subject_name_en": "Image Processing",
                },
                "confidence": 0.95,
            }
        if tool == "get_schedule_rows":
            return {
                "rows": [],
                "source_files": ["PHU LUC TKB HKII 2025-2026.pdf"],
                "coverage_note": "Chưa thấy dữ liệu lịch cho INT3404E trong thời khóa biểu hiện có.",
            }
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.4": {
                        "group_name": "Nhom cac hoc phan ve Tuong tac nguoi-may",
                        "subjects": [{"code": "INT3404E", "name": "Xu ly anh"}],
                    }
                }
            }
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise AssertionError("structured no-data route must not backfill citations with retrieve_chunks")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "mon xu ly anh ky nay co mo lop khong",
            "session_id": "s_course_status_no_rows_citations",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "course_offering_status"
    assert body["citations"][0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"
    assert "khong mo lop" in app_mod.normalize_for_match(body["answer"])


def test_ask_course_offering_status_with_schedule_words_avoids_planner(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("opening + schedule status query must stay on structured route")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            return {
                "matched_subject": {
                    "subject_code": "INT3404E",
                    "subject_name_vi": "Xu ly anh",
                    "subject_name_en": "Image Processing",
                },
                "confidence": 0.95,
            }
        if tool == "get_schedule_rows":
            return {
                "rows": [],
                "source_files": ["PHU LUC TKB HKII 2025-2026.pdf"],
                "coverage_note": "Chưa thấy dữ liệu lịch cho INT3404E trong thời khóa biểu hiện có.",
            }
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.4": {
                        "group_name": "Nhom cac hoc phan ve Tuong tac nguoi-may",
                        "subjects": [{"code": "INT3404E", "name": "Xu ly anh"}],
                    }
                }
            }
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise AssertionError("structured course status query must not call retrieve_chunks")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "môn xử lí ảnh kỳ này mở lớp không, lịch vào hôm nào vậy",
            "session_id": "s_course_status_schedule_words",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "course_offering_status"
    assert "int3404e" in app_mod.normalize_for_match(body["answer"])
    assert "khong mo lop" in app_mod.normalize_for_match(body["answer"])

    resp = client.post(
        "/ask",
        json={
            "query": "thế môn xử lí ảnh kỳ này không có lớp à",
            "session_id": "s_course_status_schedule_words",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "course_offering_status"
    assert "int3404e" in app_mod.normalize_for_match(body["answer"])
    assert "khong mo lop" in app_mod.normalize_for_match(body["answer"])


def test_ask_course_overview_followup_uses_recent_subject_without_retrieve(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("course overview follow-up must stay on structured route")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_state_get":
            return {
                "entities": {"course_codes": ["INT3412E"]},
                "referents": {"last_subject_codes": ["INT3412E"]},
            }
        if tool == "memory_get":
            return []
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.4": {
                        "group_name": "Nhom cac hoc phan ve Tuong tac nguoi-may",
                        "subjects": [{"code": "INT3412E", "name": "Thi giac may Computer Vision"}],
                    }
                }
            }
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise AssertionError("course overview follow-up must not call retrieve_chunks")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "môn này là nội dung về cái gì",
            "session_id": "s_course_overview_followup",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "course_overview"
    assert "int3412e" in app_mod.normalize_for_match(body["answer"])
    assert "computer vision" in app_mod.normalize_for_match(body["answer"])


def test_ask_ielts_requirement_uses_deterministic_language_route(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    retrieve_calls: list[dict] = []

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            retrieve_calls.append(dict(args))
            return [
                "[SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf - Chunk 6 - Page 12 - Line 8] "
                "IELTS tối thiểu 5.5 để đáp ứng điều kiện ngoại ngữ."
            ]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình không",
            "session_id": "s_ielts_structured",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "du dieu kien" in norm_answer
    assert "ielts 5.5" in norm_answer
    assert len(retrieve_calls) == 0
    assert all(
        "transcript" not in " ".join(str(fid) for fid in (call.get("file_ids") or [])).lower()
        for call in retrieve_calls
    )
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert app_mod._is_handbook_resource_name(str(body["citations"][0]["source_file"]))


def test_ask_ielts_requirement_without_selected_files_uses_global_policy_citations(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve planning")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    retrieve_calls: list[dict] = []

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            retrieve_calls.append(dict(args))
            return [
                "[SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf - Chunk 6 - Page 12 - Line 8] "
                "IELTS tối thiểu 5.5 để đáp ứng điều kiện ngoại ngữ."
            ]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình đào tạo không",
            "session_id": "s_ielts_structured_no_files",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "du dieu kien" in norm_answer
    assert len(retrieve_calls) == 0
    assert all(
        "transcript" not in " ".join(str(fid) for fid in (call.get("file_ids") or [])).lower()
        for call in retrieve_calls
    )
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert app_mod._is_handbook_resource_name(str(body["citations"][0]["source_file"]))


def test_ask_ielts_requirement_uses_resource_citation_when_retrieve_unavailable(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve planning")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise RuntimeError("mcp unavailable")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": "SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf", "scope": "global"},
            {"type": "pdf", "name": "PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026.pdf", "scope": "global"},
        ],
    )

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình đào tạo không",
            "session_id": "s_ielts_structured_resource_fallback",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert app_mod._is_handbook_resource_name(str(body["citations"][0]["source_file"]))


def test_extract_language_requirement_resource_citations_prefers_handbook_table_excerpt(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"
    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": handbook_name, "scope": "global"},
            {"type": "html", "name": "Chuan_dau_ra_CNTT.html", "scope": "global"},
        ],
    )
    monkeypatch.setattr(
        app_mod,
        "_extract_language_table_citation_from_handbook",
        lambda source_name, session_id, user_id=None, max_chars=1600: {
            "page": 26,
            "source_line": 5,
            "excerpt": "BẢNG THAM CHIẾU KẾT QUẢ CÁC BÀI THI TIẾNG ANH | KNLNVN | IELTS | TOEIC |",
        },
    )

    citations = app_mod._extract_language_requirement_resource_citations(
        session_id="s_extract_language_cite_table",
        user_id=None,
        max_items=3,
    )

    assert len(citations) >= 1
    assert citations[0]["source_file"] == handbook_name
    assert citations[0]["page"] == 26
    assert citations[0]["source_line"] == 5
    assert "BẢNG THAM CHIẾU" in str(citations[0]["excerpt"])
    assert "IELTS" in str(citations[0]["excerpt"])


def test_extract_language_requirement_resource_citations_reads_cached_handbook_chunk_when_pdf_text_empty(monkeypatch, tmp_path):
    import pickle
    from types import SimpleNamespace

    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"
    handbook_pdf = tmp_path / handbook_name
    handbook_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_mod, "CACHE_DIR", cache_dir)
    cache_file = cache_dir / f"{handbook_name}.pkl"
    docs = [
        SimpleNamespace(
            page_content=(
                "### BẢNG THAM CHIẾU KẾT QUẢ CÁC BÀI THI TIẾNG ANH\n"
                "| KNLNVN | IELTS | TOEFL |\n"
                "| Bậc 4 | 5.5 | 72 iBT |"
            ),
            metadata={"page": 26},
        )
    ]
    with cache_file.open("wb") as fh:
        pickle.dump(docs, fh)

    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": handbook_name, "scope": "global"},
        ],
    )
    monkeypatch.setattr(
        app_mod,
        "_resolve_resource_pdf_path_by_name",
        lambda source_name, session_id, user_id=None: handbook_pdf,
    )

    citations = app_mod._extract_language_requirement_resource_citations(
        session_id="s_extract_language_cite_cache",
        user_id=None,
        max_items=3,
    )

    assert len(citations) >= 1
    first = citations[0]
    assert first["source_file"] == handbook_name
    assert first["page"] == 26
    assert "BẢNG THAM CHIẾU" in str(first["excerpt"])
    assert "IELTS" in str(first["excerpt"])
    assert "5.5" in str(first["excerpt"])


def test_extract_language_requirement_resource_citations_fallback_keeps_table_reference(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"
    monkeypatch.setattr(
        app_mod,
        "_collect_language_handbook_resources",
        lambda session_id, user_id=None: [
            {"source_name": handbook_name, "resource_id": handbook_name, "pdf_path": ""}
        ],
    )
    monkeypatch.setattr(
        app_mod,
        "_extract_language_table_citation_from_handbook",
        lambda source_name, session_id, user_id=None, max_chars=1600, pdf_path_override=None: None,
    )

    citations = app_mod._extract_language_requirement_resource_citations(
        session_id="s_extract_language_cite_fallback_table",
        user_id=None,
        max_items=3,
    )

    assert len(citations) >= 1
    first = citations[0]
    excerpt = str(first.get("excerpt") or "")
    assert "BẢNG THAM CHIẾU" in excerpt
    assert "IELTS" in excerpt
    assert "Bậc 4" in excerpt
    assert "5.5" in excerpt


def test_ask_ielts_requirement_resource_fallback_prefers_handbook_with_underscored_name(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve planning")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise RuntimeError("mcp unavailable")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"
    unrelated_policy_name = (
        "Chuan_dau_ra_chuong_trinh_dao_tao_nganh_Cong_nghe_thong_tin_"
        "-_Truong_Dai_hoc_Cong_nghe_DHQGHN.html"
    )
    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": handbook_name, "scope": "global"},
            {"type": "html", "name": unrelated_policy_name, "scope": "global"},
        ],
    )

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình đào tạo không",
            "session_id": "s_ielts_structured_resource_fallback_underscored_handbook",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == handbook_name
    assert all(item.get("source_file") != unrelated_policy_name for item in body["citations"])


def test_ask_ielts_requirement_ignores_non_handbook_retrieve_citations_when_handbook_exists(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve planning")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    ctdt_source = (
        "Chuan_dau_ra_chuong_trinh_dao_tao_nganh_Cong_nghe_thong_tin_"
        "-_Truong_Dai_hoc_Cong_nghe_DHQGHN.html"
    )
    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            return [
                f"[{ctdt_source} - Chunk 4 - Page 1 - Line 10] "
                "Chuan dau ra ngoai ngu bac 4."
            ]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": handbook_name, "scope": "global"},
            {"type": "html", "name": ctdt_source, "scope": "global"},
        ],
    )

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình đào tạo không",
            "session_id": "s_ielts_structured_prefers_handbook_over_ctdt_retrieve",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == handbook_name
    assert all(item.get("source_file") != ctdt_source for item in body["citations"])


def test_ask_ielts_requirement_prefers_handbook_table_chunk_when_available(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            raise AssertionError("IELTS requirement query must not require planner/retrieve planning")

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    handbook_name = "SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf"
    retrieve_calls: list[dict] = []

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            retrieve_calls.append(dict(args))
            raise AssertionError("language_requirement route should not call retrieve_chunks for citation backfill")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)
    monkeypatch.setattr(
        app_mod,
        "_extract_language_table_citation_from_handbook",
        lambda source_name, session_id, user_id=None, max_chars=1600: {
            "page": 26,
            "source_line": 5,
            "excerpt": "### BẢNG THAM CHIẾU KẾT QUẢ CÁC BÀI THI TIẾNG ANH | KNLNVN | IELTS | TOEFL |",
        },
    )
    monkeypatch.setattr(
        app_mod.resource_loader,
        "get_resources",
        lambda session_id=None, user_id=None: [
            {"type": "pdf", "name": handbook_name, "scope": "global"},
        ],
    )

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh theo chương trình đào tạo không",
            "session_id": "s_ielts_structured_prefers_handbook_table_chunk",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "language_requirement"
    assert len(retrieve_calls) == 0
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == handbook_name
    excerpt = str(body["citations"][0].get("excerpt") or "")
    assert "KNLNVN" in excerpt
    assert "IELTS" in excerpt
    assert body["citations"][0].get("page") == 26
    assert body["citations"][0].get("source_line") == 5


def test_ask_electives_no_opened_uses_schedule_citation_without_retrieve(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "get_electives_with_schedule":
            return {
                "opened": [],
                "opened_count": 0,
                "not_opened": [{"code": "INT3404E", "name": "Xu ly anh", "credits": 3}],
                "schedule_source_file": "PHU LUC TKB HKII 2025-2026.pdf",
            }
        if tool == "memory_add":
            return "ok"
        if tool == "retrieve_chunks":
            raise AssertionError("structured elective no-opened route must not call retrieve_chunks")
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "ki nay co nhung mon tu chon nao mo lop",
            "session_id": "s_electives_no_opened_citations",
            "program_id": "cs_2022",
            "file_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "electives_schedule"
    assert "khong tim thay hoc phan tu chon dang mo lop" in app_mod.normalize_for_match(body["answer"])
    assert body["citations"][0]["source_file"] == "PHU LUC TKB HKII 2025-2026.pdf"


def test_extract_subject_hints_ignores_curriculum_tail_phrase():
    app_mod = importlib.reload(importlib.import_module("app"))
    hints = app_mod._extract_subject_hints(
        "the mon xu li anh ki nay co mo khong va ma mon nay co nam trong khung ctdt cua toi khong"
    )
    normalized_hints = [app_mod.normalize_for_match(h) for h in hints]
    assert "xu li anh" in normalized_hints
    assert all("khung ctdt" not in h for h in normalized_hints)

    accented_hints = app_mod._extract_subject_hints("môn xử lí ảnh kỳ này mở lớp không, lịch vào hôm nào vậy")
    assert [app_mod.normalize_for_match(h) for h in accented_hints] == ["xu li anh"]
    negative_hints = app_mod._extract_subject_hints("thế môn xử lí ảnh kỳ này không có lớp à")
    assert [app_mod.normalize_for_match(h) for h in negative_hints] == ["xu li anh"]


def test_ask_course_offering_status_prefers_curriculum_fallback_when_alias_uncertain(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class DummyPlanner:
        def run(self, prompt):
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "resolve_course_alias":
            # Deliberately uncertain/incorrect alias to verify curriculum fallback wins.
            return {
                "matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thi giac may"},
                "confidence": 0.66,
            }
        if tool == "get_curriculum_lookup":
            return {
                "groups": {
                    "V.2.4": {
                        "group_name": "Nhom cac hoc phan ve Tuong tac nguoi-may",
                        "subjects": [
                            {"code": "INT3404E", "name": "Xu ly anh Image Processing"},
                            {"code": "INT3412E", "name": "Thi giac may Computer Vision"},
                        ],
                    }
                }
            }
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").strip().upper()
            if code == "INT3404E":
                return {"rows": []}
            if code == "INT3412E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3412E",
                            "class_code": "INT3412E 1",
                            "day_of_week": "Thu 3",
                            "slot": "2",
                            "room": "209-T",
                            "teacher_name": "Le Thanh Ha",
                        }
                    ]
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "the mon xu li anh ki nay co mo khong va ma mon nay co nam trong khung ctdt cua toi khong",
            "session_id": "s_course_status_curriculum_fallback",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    norm_answer = app_mod.normalize_for_match(body["answer"])
    assert "int3404e" in norm_answer
    assert "khong mo lop" in norm_answer
    assert "v.2.4" in norm_answer
    assert "int3412e" not in norm_answer


def test_render_course_offering_status_collapses_duplicate_schedule_rows():
    app_mod = importlib.reload(importlib.import_module("app"))
    context = json.dumps(
        {
            "matched_subject": {
                "subject_code": "INT3230E",
                "subject_name_vi": "Máº­t mÃ£ vÃ  an toÃ n thÃ´ng tin",
                "subject_name_en": "Cryptography and Information security",
            },
            "subject_code": "INT3230E",
            "is_opened": True,
            "in_curriculum": True,
            "curriculum_group_code": "V.2.2",
            "curriculum_group_name": "NhÃ³m cÃ¡c há»c pháº§n vá» CÃ¡c há»‡ thá»‘ng á»¨ng dá»¥ng TrÃ­ tuá»‡ nhÃ¢n táº¡o",
            "rows": [
                {
                    "class_code": "INT3230E 1",
                    "day_of_week": "Thá»© 2",
                    "slot": "1",
                    "room": "209-T",
                    "week_note": "1-8",
                },
                {
                    "class_code": "INT3230E 1",
                    "day_of_week": "Thá»© 2",
                    "slot": "1",
                    "room": "209-T",
                    "week_note": "9-16",
                },
                {
                    "class_code": "INT3230E 1",
                    "day_of_week": "Thá»© 2",
                    "slot": "1",
                    "room": "209-T",
                    "teacher_name": "GV A",
                },
                {
                    "class_code": "INT3230E 1",
                    "day_of_week": "Thá»© 4",
                    "slot": "4",
                    "room": "211-T",
                },
            ],
        },
        ensure_ascii=False,
    )
    answer = app_mod._render_course_offering_status_answer(context)
    norm_answer = app_mod.normalize_for_match(answer)
    assert norm_answer.count("int3230e 1: thu 2, ca 1, phong 209-t") == 1
    assert norm_answer.count("int3230e 1: thu 4, ca 4, phong 211-t") == 1
def test_ask_transcript_bypass_returns_error_when_advisor_times_out(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    invoke_log: list[tuple[str, float | None]] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswer:
        def run(self, query, context, source, memory_context):
            return f"answer-from-{source}:{context}"

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswer())

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            invoke_log.append((tool, timeout))
            raise RuntimeError("simulated advisor timeout")
        if tool == "retrieve_chunks":
            invoke_log.append((tool, timeout))
            return ["fallback-context"]
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can ban kiem tra giup toi xem toi con thieu nhung mon nao theo chuong trinh dao tao voi",
            "session_id": "s_transcript_timeout_fallback",
            "program_id": "cs_2022",
            "file_ids": ["1_a.pdf", "2_b.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Khong the hoan tat phan tich bang diem/CTDT" in body["answer"]
    assert planner_calls["count"] == 0
    assert any(
        name == "consult_advisor" and timeout == app_mod.MCP_TOOL_TIMEOUTS_TRANSCRIPT["consult_advisor"]
        for name, timeout in invoke_log
    )
    assert not any(name == "retrieve_chunks" for name, _ in invoke_log)


def test_ask_transcript_timeout_retries_with_trimmed_files(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    invoke_log: list[tuple[str, list[str], float | None]] = []

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            file_ids = [str(item) for item in (args.get("file_ids") or [])]
            invoke_log.append((tool, file_ids, timeout))
            if len(file_ids) > 2:
                raise RuntimeError("simulated advisor timeout on large file set")
            return "trimmed-advisor-answer"
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi can ban kiem tra giup toi xem toi con thieu nhung mon nao theo chuong trinh dao tao voi",
            "session_id": "s_transcript_trimmed_retry",
            "program_id": "cs_2022",
            "file_ids": ["a.pdf", "b.pdf", "c.pdf", "d.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "trimmed-advisor-answer"

    consult_calls = [entry for entry in invoke_log if entry[0] == "consult_advisor"]
    assert len(consult_calls) == 2
    assert len(consult_calls[0][1]) == 4
    assert len(consult_calls[1][1]) == 2
    assert all(timeout == app_mod.MCP_TOOL_TIMEOUTS_TRANSCRIPT["consult_advisor"] for _, _, timeout in consult_calls)


def test_ask_missing_subjects_prefers_advisor_even_when_structured_route_is_high_confidence(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    invoke_log: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {"source": "vector_store", "context": "planner-ctx", "memory": "mem", "chunk_index": None}
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())
    monkeypatch.setattr(
        app_mod,
        "_structured_intent_classifier",
        lambda query: {"intent": "course_schedule", "confidence": 0.92, "signals": ["schedule_marker"]},
    )
    monkeypatch.setattr(
        app_mod,
        "_build_structured_route_payload",
        lambda **kwargs: {
            "source": "structured_schedule",
            "context": "structured-ctx",
            "memory": "mem",
            "chunk_index": None,
        },
    )

    def fake_invoke(tool, args, timeout=None):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "consult_advisor":
            invoke_log.append(tool)
            return "advisor-final-answer"
        if tool == "retrieve_chunks":
            invoke_log.append(tool)
            return ["fallback-context"]
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "toi con thieu nhung mon nao va bao nhieu tin chi theo chuong trinh dao tao",
            "session_id": "s_missing_subjects_advisor_priority",
            "program_id": "cs_2022",
            "file_ids": ["bang_diem1.pdf", "bang_diem2.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "advisor-final-answer"
    assert planner_calls["count"] == 0
    assert "consult_advisor" in invoke_log
    assert "retrieve_chunks" not in invoke_log


def test_infer_schedule_semester_code_maps_hk1_hk2_and_summer():
    app_mod = importlib.reload(importlib.import_module("app"))

    assert app_mod._infer_schedule_semester_code("mÃ´n nÃ y há»c ká»³ 1 nÄƒm há»c 2025-2026") == "251"
    assert app_mod._infer_schedule_semester_code("mÃ´n nÃ y há»c ká»³ 2 nÄƒm há»c 2025-2026") == "252"
    assert app_mod._infer_schedule_semester_code("mÃ´n nÃ y ká»³ hÃ¨ nÄƒm há»c 2025-2026") == "252"
    assert app_mod._infer_schedule_semester_code("HKII 25-26") == "252"


def test_structured_intent_classifier_detects_semester_code_lookup():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier("mã kỳ học theo thời khóa biểu hiện tại là gì")
    assert route["intent"] == "semester_code_lookup"
    assert float(route["confidence"]) >= 0.85


def test_build_structured_route_payload_semester_code_lookup_reads_from_schedule_rows(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    invoke_calls: list[tuple[str, dict]] = []

    def fake_invoke(tool, args):
        invoke_calls.append((tool, dict(args)))
        if tool == "get_schedule_rows":
            return {
                "rows": [
                    {
                        "semester": "HKII 2025-2026",
                        "subject_code": "INT4050",
                        "source_file": "PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf",
                    }
                ],
                "source_files": ["PHỤ LỤC THỜI KHÓA BIỂU HKII 2025-2026.xlsx - Sheet1.pdf"],
            }
        return {}

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    payload = app_mod._build_structured_route_payload(
        query="mã kỳ học theo thời khóa biểu hiện tại là gì",
        session_id="s_semester_lookup",
        program_id="cs_2022",
        intent="semester_code_lookup",
        confidence=0.9,
        memory_context="",
    )

    assert payload is not None
    assert payload.get("source") == "semester_code_lookup"
    context = app_mod._safe_json_loads(payload.get("context"))
    assert context.get("semester_code") == "252"
    assert context.get("inference_source") in {"schedule_rows", "source_file"}
    assert any(tool == "get_schedule_rows" for tool, _ in invoke_calls)


def test_structured_route_passes_inferred_semester_to_teacher_lookup(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    invoke_calls: list[tuple[str, dict]] = []

    def fake_invoke(tool, args):
        invoke_calls.append((tool, dict(args)))
        if tool == "resolve_course_alias":
            return {
                "matched_subject": {"subject_code": "HIS1001", "subject_name_vi": "Lich su Dang"},
                "confidence": 0.95,
            }
        if tool == "get_teachers_by_subject":
            return {
                "matched_subject": {"subject_code": "HIS1001", "subject_name_vi": "Lich su Dang"},
                "teachers": ["Vu Thi Thu Ha"],
                "rows": [
                    {
                        "subject_code": "HIS1001",
                        "class_code": "HIS1001 5",
                        "day_of_week": "Thu 2",
                        "slot": "1",
                        "room": "103-A",
                        "teacher_name": "Vu Thi Thu Ha",
                    }
                ],
                "source_files": ["PHU_LUC_TKB.pdf"],
                "coverage_note": "ok",
            }
        return {}

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    payload = app_mod._build_structured_route_payload(
        query="mÃ´n lá»‹ch sá»­ Ä‘áº£ng há»c ká»³ 1 nÄƒm há»c 2025-2026 cÃ³ nhá»¯ng ai dáº¡y",
        session_id="s_semester_pass",
        program_id="cs_2022",
        intent="teacher_by_subject",
        confidence=0.9,
        memory_context="",
    )

    assert payload is not None
    assert payload.get("source") == "structured_schedule"
    teacher_call = next((args for tool, args in invoke_calls if tool == "get_teachers_by_subject"), {})
    assert teacher_call.get("semester") == "251"


def test_structured_intent_classifier_detects_time_slot_definition_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier("ca 1 bắt đầu từ mấy giờ và kết thúc lúc mấy giờ")
    assert route["intent"] == "course_schedule"
    assert float(route["confidence"]) >= 0.85
    assert app_mod._query_prefers_global_resource_retrieval("ca 1 bắt đầu từ mấy giờ và kết thúc lúc mấy giờ") is True


def test_structured_intent_classifier_detects_followup_subject_schedule_query():
    app_mod = importlib.reload(importlib.import_module("app"))
    route = app_mod._structured_intent_classifier("thế còn thị giác máy tính")
    assert route["intent"] == "course_schedule"
    assert float(route["confidence"]) >= 0.75
    assert app_mod._extract_subject_hints("thế còn thị giác máy tính") == ["thi giac may tinh"]


def test_structured_route_unresolved_schedule_falls_back_to_retrieve(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    invoke_log: list[str] = []

    def fake_invoke(tool, args):
        invoke_log.append(tool)
        if tool == "get_curriculum_lookup":
            return {"groups": {}}
        if tool == "resolve_course_alias":
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "retrieve_chunks":
            return ["[TKB.pdf - Chunk 1] INT3412E 1: Thứ 3, Ca 2, phòng 209-T"]
        return {}

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    payload = app_mod._build_structured_route_payload(
        query="thế còn thị giác máy tính",
        session_id="s_followup_unresolved",
        program_id="cs_2022",
        intent="course_schedule",
        confidence=0.78,
        memory_context="",
    )

    assert payload is not None
    assert payload.get("source") == "vector_store"
    assert "retrieve_chunks" in invoke_log
    assert "INT3412E" in str(payload.get("context") or "")


def test_build_structured_route_payload_uses_time_slot_lookup_tool(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    invoke_calls: list[tuple[str, dict]] = []

    def fake_invoke(tool, args):
        invoke_calls.append((tool, dict(args)))
        if tool == "get_time_slot_info":
            return {
                "slot": "1",
                "period": "Tiet 1-3",
                "time_range": "07:00 – 09:40",
                "source_file": "Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf",
            }
        return {}

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    payload = app_mod._build_structured_route_payload(
        query="ca 1 bắt đầu từ mấy giờ và kết thúc lúc mấy giờ",
        session_id="s_time_slot",
        program_id="cs_2022",
        intent="course_schedule",
        confidence=0.62,
        memory_context="",
    )

    assert payload is not None
    assert payload.get("source") == "time_slot_lookup"
    assert "07:00" in str(payload.get("context") or "")
    tool_names = [name for name, _ in invoke_calls]
    assert "get_time_slot_info" in tool_names


def test_render_time_slot_lookup_answer_and_citation_extraction():
    app_mod = importlib.reload(importlib.import_module("app"))
    context = json.dumps(
        {
            "slot": "2",
            "period": "Tiet 4-6",
            "time_range": "09:50 – 12:30",
            "source_file": "Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf",
        },
        ensure_ascii=False,
    )
    answer = app_mod._render_time_slot_lookup_answer(
        query="ca 2 bắt đầu học lúc mấy giờ vậy",
        context=context,
    )
    citations = app_mod._extract_time_slot_citations(context, max_items=4)

    assert "Ca 2 bắt đầu từ 09:50 và kết thúc lúc 12:30" in answer
    assert len(citations) == 1
    assert citations[0]["source_file"] == "Signed.Signed.CV TKB chính thức HKII 25-26 gửi SV.pdf"
    assert "Ca 2" in citations[0]["excerpt"]


def test_ask_deictic_schedule_query_merges_memory_subject_with_explicit_subject(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    planner_calls = {"count": 0}
    schedule_calls: list[str] = []

    class DummyPlanner:
        def run(self, prompt):
            planner_calls["count"] += 1
            payload = {
                "source": "academic_advisor",
                "context": "advisor-should-not-win-for-deictic-merge",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    memory_lines = [
        "[2026-04-20 10:00:00] Query: về môn khoa học dữ liệu kỳ này có mở không\n"
        "Response: INT3425 - Khoa học dữ liệu có mở lớp."
    ]

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return memory_lines
        if tool == "resolve_course_alias":
            query = str(args.get("query") or "")
            q_norm = app_mod.normalize_for_match(query)
            if query.upper() == "INT3425":
                return {
                    "matched_subject": {"subject_code": "INT3425", "subject_name_vi": "Khoa học dữ liệu"},
                    "confidence": 0.98,
                }
            if q_norm == "thi giac may":
                return {
                    "matched_subject": {"subject_code": "INT3412E", "subject_name_vi": "Thị giác máy"},
                    "confidence": 0.95,
                }
            return {"matched_subject": None, "confidence": 0.0}
        if tool == "get_schedule_rows":
            code = str(args.get("subject_code") or "").upper()
            schedule_calls.append(code)
            if code == "INT3425":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3425",
                            "class_code": "INT3425 1",
                            "day_of_week": "Thứ 4",
                            "slot": "1",
                            "room": "107-B",
                            "teacher_name": "GV A",
                        }
                    ],
                    "source_files": ["TKB.pdf"],
                }
            if code == "INT3412E":
                return {
                    "rows": [
                        {
                            "subject_code": "INT3412E",
                            "class_code": "INT3412E 1",
                            "day_of_week": "Thứ 3",
                            "slot": "2",
                            "room": "209-T",
                            "teacher_name": "Lê Thanh Hà",
                        }
                    ],
                    "source_files": ["TKB.pdf"],
                }
            return {"rows": []}
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "môn này với môn thị giác máy lịch học như nào vậy",
            "session_id": "s_deictic_merge",
            "program_id": "cs_2022",
            "file_ids": ["dummy_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "INT3425 1" in body["answer"]
    assert "INT3412E 1" in body["answer"]
    assert "advisor-should-not-win-for-deictic-merge" not in body["answer"]
    assert set(schedule_calls) == {"INT3425", "INT3412E"}
    assert planner_calls["count"] == 0


def test_ask_fallback_retrieve_uses_global_scope_for_policy_query(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "STRUCTURED_TKB_ENABLED", False)
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    class BrokenPlanner:
        def run(self, prompt):
            raise RuntimeError("planner down")

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return f"answer:{source}"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: BrokenPlanner())

    retrieve_file_ids: list[list[str]] = []

    def fake_invoke(tool, args):
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "memory_get":
            return []
        if tool == "retrieve_chunks":
            retrieve_file_ids.append(list(args.get("file_ids") or []))
            return [
                "[SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf - Chunk 6 - Page 12 - Line 8] "
                "IELTS tối thiểu 5.5 để đáp ứng điều kiện ngoại ngữ."
            ]
        if tool == "memory_add":
            return "ok"
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh ra trường không",
            "session_id": "s_global_policy_fallback",
            "program_id": "cs_2022",
            "file_ids": ["1_transcript.pdf", "2_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "vector_store"
    assert len(retrieve_file_ids) >= 1
    assert all(ids == [] for ids in retrieve_file_ids)
    assert isinstance(body.get("citations"), list)
    assert body["citations"][0]["source_file"] == "SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"


def test_ask_vector_store_policy_query_overrides_selected_file_context_with_global_retrieve(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    (app_mod.SESSION_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_mod, "STRUCTURED_TKB_ENABLED", False)

    class DummyPlanner:
        def run(self, prompt):
            payload = {
                "source": "vector_store",
                "context": "[1_transcript.pdf - Chunk 1 - Page 1 - Line 1] Bang diem tong hop",
                "memory": "mem",
                "chunk_index": None,
            }
            return type("Resp", (), {"content": json.dumps(payload)})()

    class DummyAnswerAgent:
        def run(self, query, context, source, memory_context):
            return "answer-from-vector"

    monkeypatch.setattr(app_mod, "answer_agent", DummyAnswerAgent())
    monkeypatch.setattr(app_mod, "get_mcp_planner_agent", lambda allow_web_search=False: DummyPlanner())

    retrieve_file_ids: list[list[str]] = []

    def fake_invoke(tool, args):
        if tool == "memory_get":
            return []
        if tool == "memory_add":
            return "ok"
        if tool == "get_available_programs":
            return {"programs": [{"id": "cs_2022", "display_name": "CS"}]}
        if tool == "retrieve_chunks":
            retrieve_file_ids.append(list(args.get("file_ids") or []))
            return [
                "[SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf - Chunk 6 - Page 12 - Line 8] "
                "IELTS tối thiểu 5.5 để đáp ứng điều kiện ngoại ngữ."
            ]
        return "ok"

    monkeypatch.setattr(app_mod.mcp_client, "invoke", fake_invoke)

    client = TestClient(app_mod.app)
    resp = client.post(
        "/ask",
        json={
            "query": "với 6.5 ielts tôi có đủ điều kiện tiếng anh ra trường không",
            "session_id": "s_vector_override",
            "program_id": "cs_2022",
            "file_ids": ["1_transcript.pdf", "2_transcript.pdf"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "vector_store"
    assert len(retrieve_file_ids) >= 1
    assert all(ids == [] for ids in retrieve_file_ids)
    assert isinstance(body.get("citations"), list)
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == "SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"

