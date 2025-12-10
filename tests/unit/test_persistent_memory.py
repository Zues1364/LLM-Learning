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
