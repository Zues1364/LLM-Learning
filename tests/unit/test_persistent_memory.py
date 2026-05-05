import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from persistent_memory import PersistentMemory  # noqa: E402


def test_add_and_get_context_respects_limit(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=2)

    mem.add_to_history("q1", "a1", session_id="s")
    mem.add_to_history("q2", "a2", session_id="s")
    mem.add_to_history("q3", "a3", session_id="s")

    ctx = mem.get_context("", session_id="s", max_rows=5)
    # Only two entries should remain (max_history=2)
    assert ctx.count("Query:") == 2


def test_history_prune_is_scoped_to_session(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=2)

    mem.add_to_history("a-q1", "a-a1", session_id="session-a")
    mem.add_to_history("b-q1", "b-a1", session_id="session-b")
    mem.add_to_history("a-q2", "a-a2", session_id="session-a")
    mem.add_to_history("a-q3", "a-a3", session_id="session-a")

    ctx_a = mem.get_context("", session_id="session-a", max_rows=5)
    ctx_b = mem.get_context("", session_id="session-b", max_rows=5)

    assert "a-q1" not in ctx_a
    assert "a-q2" in ctx_a
    assert "a-q3" in ctx_a
    assert "b-q1" in ctx_b


def test_history_can_be_scoped_by_user_and_session(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)

    mem.add_to_history("same-session-user-a", "answer-a", session_id="shared", user_id="user-a")
    mem.add_to_history("same-session-user-b", "answer-b", session_id="shared", user_id="user-b")
    mem.add_to_history("anonymous", "answer-anon", session_id="shared")

    ctx_a = mem.get_context("", session_id="shared", user_id="user-a", max_rows=5)
    ctx_b = mem.get_context("", session_id="shared", user_id="user-b", max_rows=5)
    ctx_anon = mem.get_context("", session_id="shared", max_rows=5)

    assert "same-session-user-a" in ctx_a
    assert "same-session-user-b" not in ctx_a
    assert "anonymous" not in ctx_a

    assert "same-session-user-b" in ctx_b
    assert "same-session-user-a" not in ctx_b

    assert "anonymous" in ctx_anon
    assert "same-session-user-a" not in ctx_anon


def test_clear_session(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)
    mem.add_to_history("q1", "a1", session_id="s")
    mem.clear_session("s")
    ctx = mem.get_context("", session_id="s", max_rows=5)
    assert ctx == ""


def test_summary_save_get(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)
    mem.save_summary("file1", "summary text")
    assert mem.get_summary("file1") == "summary text"


def test_get_context_respects_max_rows(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=10)
    for i in range(5):
        mem.add_to_history(f"q{i}", f"a{i}", session_id="s")

    ctx = mem.get_context("", session_id="s", max_rows=3)
    # Respect max_rows limit
    assert ctx.count("Query:") == 3


def test_chat_sessions_and_messages_are_scoped_by_user(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)

    mem.ensure_chat_session(
        session_id="shared",
        user_id="user-a",
        title="Chat A",
        selected_program_id="cs_2022",
        selected_file_ids=["1.pdf", "2.pdf"],
    )
    mem.ensure_chat_session(session_id="shared", user_id="user-b", title="Chat B")
    mem.add_chat_message("shared", "user-a", "user", "hello")
    mem.add_chat_message("shared", "user-a", "assistant", "answer", citations=[{"source_file": "a.pdf"}])
    mem.add_chat_message("shared", "user-b", "user", "other")

    sessions_a = mem.list_chat_sessions("user-a")
    sessions_b = mem.list_chat_sessions("user-b")
    messages_a = mem.get_chat_messages("shared", "user-a")
    messages_b = mem.get_chat_messages("shared", "user-b")

    assert len(sessions_a) == 1
    assert sessions_a[0]["id"] == "shared"
    assert sessions_a[0]["title"] == "Chat A"
    assert sessions_a[0]["selected_program_id"] == "cs_2022"
    assert sessions_a[0]["selected_file_ids"] == ["1.pdf", "2.pdf"]
    assert sessions_b[0]["title"] == "Chat B"

    assert [msg["content"] for msg in messages_a] == ["hello", "answer"]
    assert messages_a[1]["citations"] == [{"source_file": "a.pdf"}]
    assert [msg["content"] for msg in messages_b] == ["other"]

    updated = mem.update_chat_session("shared", "user-a", title="Renamed", selected_file_ids=["3.pdf"])
    assert updated["title"] == "Renamed"
    assert updated["selected_file_ids"] == ["3.pdf"]

    assert mem.archive_chat_session("shared", "user-a") is True
    assert mem.list_chat_sessions("user-a") == []
    assert len(mem.list_chat_sessions("user-a", include_archived=True)) == 1
    assert len(mem.list_chat_sessions("user-b")) == 1


def test_migrate_legacy_history_to_chat_sessions_claims_raw_sessions(tmp_path):
    db_path = tmp_path / "mem.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)

    mem.add_to_history("legacy q1", "legacy a1", session_id="legacy-session")
    mem.add_to_history("legacy q2", "legacy a2", session_id="legacy-session")
    mem.add_to_history("other q", "other a", session_id="other-session")
    mem.add_to_history("scoped q", "scoped a", session_id="legacy-session", user_id="existing-user")

    assert mem.migrate_legacy_history_to_chat_sessions("user-a") == 2
    sessions = mem.list_chat_sessions("user-a")
    assert {item["id"] for item in sessions} == {"legacy-session", "other-session"}
    assert sessions[0]["title"] in {"legacy q1", "other q"}

    messages = mem.get_chat_messages("legacy-session", "user-a")
    assert [msg["role"] for msg in messages] == ["user", "assistant", "user", "assistant"]
    assert [msg["content"] for msg in messages] == ["legacy q1", "legacy a1", "legacy q2", "legacy a2"]

    assert mem.migrate_legacy_history_to_chat_sessions("user-a") == 0
    assert mem.migrate_legacy_history_to_chat_sessions("user-b") == 0
    assert mem.list_chat_sessions("user-b") == []
    assert mem.get_context("", session_id="legacy-session", user_id="existing-user") != ""
