import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from persistent_memory import PersistentMemory  # noqa: E402


def test_structured_state_roundtrip(tmp_path):
    db_path = tmp_path / "memory.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)

    initial = mem.get_structured_state("s1")
    assert initial["turn_index"] == 0

    payload = {
        "turn_index": 3,
        "last_query": "mon nay la mon gi",
        "entities": {"course_codes": ["INT3403"], "semester_code": "252"},
        "referents": {"last_subject_codes": ["INT3403"]},
    }
    saved = mem.save_structured_state("s1", payload)
    assert saved["turn_index"] == 3
    assert "INT3403" in saved["entities"]["course_codes"]

    loaded = mem.get_structured_state("s1")
    assert loaded["turn_index"] == 3
    assert loaded["entities"]["semester_code"] == "252"


def test_clear_session_also_clears_structured_state(tmp_path):
    db_path = tmp_path / "memory.db"
    mem = PersistentMemory(db_path=str(db_path), max_history=5)
    mem.add_to_history("q1", "a1", "s2")
    mem.save_structured_state("s2", {"turn_index": 2, "entities": {"course_codes": ["INT2041"]}})

    mem.clear_session("s2")
    loaded = mem.get_structured_state("s2")
    assert loaded["turn_index"] == 0
    assert loaded["entities"]["course_codes"] == []
