import sys
import types
from pathlib import Path

# Provide lightweight stubs so we can import server helpers without FastAPI in test envs.
if "fastapi" not in sys.modules:
    class _DummyApp:
        def get(self, *args, **kwargs):
            def deco(fn): return fn
            return deco
        def post(self, *args, **kwargs):
            def deco(fn): return fn
            return deco
        def on_event(self, *args, **kwargs):
            def deco(fn): return fn
            return deco
        def middleware(self, *args, **kwargs):
            def deco(fn): return fn
            return deco
    class _DummyRequest: ...
    fastapi_stub = types.SimpleNamespace(FastAPI=lambda *a, **k: _DummyApp(), HTTPException=Exception, Request=_DummyRequest)
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.responses"] = types.SimpleNamespace(JSONResponse=lambda *a, **k: None)
if "pydantic" not in sys.modules:
    class _DummyBaseModel: ...
    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_DummyBaseModel)
if "agents" not in sys.modules:
    class _DummyAgent:
        def run(self, _prompt):
            return type("Resp", (), {"content": ""})()
    sys.modules["agents"] = types.SimpleNamespace(get_academic_advisor_agent=lambda: _DummyAgent())

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mcp_server.server import (  # noqa: E402
    _extract_target_gpa,
    _query_limits_to_remaining_credits,
    calculate_gpa_feasibility,
    compute_missing_subjects,
)


def _sample_transcript():
    return {
        "semesters": [
            {
                "semester_code": "241",
                "subjects": [
                    {"code": "INT1001", "credits": 3, "grade_4": 3.0},
                    {"code": "INT1002", "credits": 3, "grade_4": 1.0},
                ],
            }
        ],
        "overview": {"total_credits_accumulated": 6},
    }


def _sample_curriculum():
    return {
        "program_name": "CS",
        "subjects": [
            {"code": "INT1001", "credits": 3},
            {"code": "INT1002", "credits": 3},
            {"code": "INT1003", "credits": 2},
        ],
        "total_credits": 8,
    }


def test_compute_missing_subjects_flags_missing_and_low_grade():
    transcript = _sample_transcript()
    curriculum = _sample_curriculum()

    result = compute_missing_subjects(transcript, curriculum)
    missing_codes = {m["code"] for m in result["missing"]}
    low_grade_codes = [m["code"] for m in result["low_grades"]]

    assert "INT1003" in missing_codes
    assert "INT1001" not in missing_codes
    assert "INT1002" in low_grade_codes  # grade 1.0 should be flagged


def test_calculate_gpa_feasibility_estimates_max_and_feasibility():
    transcript = _sample_transcript()
    curriculum = _sample_curriculum()

    projection = calculate_gpa_feasibility(
        transcript,
        curriculum_total_credits=curriculum["total_credits"],
        target_gpa=3.2,
    )

    # Current GPA = (3*3.0 + 3*1.0) / 6 = 2.0
    assert projection["current_gpa"] == 2.0
    # Remaining credits = 2 (curriculum total 8 - earned 6); max possible GPA should be below 3.2
    assert projection["remaining_credits"] == 2
    assert projection["max_possible_gpa"] is not None
    assert projection["feasible"] is False


def test_gpa_feasibility_infers_distinction_target_and_remaining_scope():
    query = "liệu với số tín chỉ còn lại tôi có thể lên bằng giỏi không"
    target_gpa = _extract_target_gpa(query)

    projection = calculate_gpa_feasibility(
        _sample_transcript(),
        curriculum_total_credits=_sample_curriculum()["total_credits"],
        target_gpa=target_gpa,
    )

    assert target_gpa == 3.2
    assert _query_limits_to_remaining_credits(query) is True
    assert projection["max_gpa_no_retakes"] < target_gpa
    assert projection["max_possible_gpa"] >= target_gpa
    assert projection["feasible_no_retakes"] is False
    assert projection["feasible_with_retakes"] is True
    assert projection["feasible"] is False
