import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from conversation_state import (  # noqa: E402
    default_conversation_state,
    resolve_query_with_state,
    update_state_after_turn,
)


def test_resolve_query_with_subject_reference():
    state = default_conversation_state()
    state["referents"]["last_subject_codes"] = ["INT3412E"]

    resolved = resolve_query_with_state("môn này có mở lớp không", state)
    assert "INT3412E" in resolved["resolved_query"]
    assert resolved["applied_referents"]


def test_update_state_extracts_course_code_from_context():
    prev = default_conversation_state()
    next_state = update_state_after_turn(
        previous_state=prev,
        raw_query="toi can lich hoc mon thi giac may",
        resolved_query="toi can lich hoc mon thi giac may",
        answer="INT3412E 1: Thu 3, Ca 2, phong 209-T",
        planner_source="academic_advisor",
        planner_context="INT3412E - Thi giac may",
        selected_program_id="cs_2022",
    )
    assert "INT3412E" in next_state["entities"]["course_codes"]
    assert "INT3412E" in next_state["referents"]["last_subject_codes"]
    assert next_state["turn_index"] == 1
    assert next_state["selected_program_id"] == "cs_2022"
