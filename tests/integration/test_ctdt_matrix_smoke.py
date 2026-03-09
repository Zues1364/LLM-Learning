import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402


TARGET_IDS = [
    "ckt_2019",
    "aut_2022",
    "ktr_2022",
    "network_2019",
    "cnhkvt_2019",
    "cs_2022",
    "it_2025",
    "ai_2025",
]


def _extract_subjects(groups):
    seen = set()
    items = []
    for group_data in (groups or {}).values():
        for subj in group_data.get("subjects", []) or []:
            code = str(subj.get("code") or "").strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            items.append(
                {
                    "code": code,
                    "name": str(subj.get("name") or code),
                    "credits": int(subj.get("credits") or 3),
                }
            )
    return sorted(items, key=lambda x: x["code"])


def _build_fake_transcript(program_id, subjects):
    if not subjects:
        subjects = [{"code": "DUMMY1001", "name": "Dummy", "credits": 3}]
    cut = max(1, int(len(subjects) * 0.65))
    completed = subjects[:cut]
    if len(completed) >= len(subjects):
        completed = completed[:-1] or completed

    rows = []
    for i, subj in enumerate(completed):
        grade = 3.3 if i % 5 else 1.2
        rows.append(
            {
                "code": subj["code"],
                "name": subj["name"],
                "credits": int(subj["credits"]),
                "grade_4": float(grade),
            }
        )
    total_credits = sum(int(r["credits"]) for r in rows)
    return {
        "student_info": {
            "name": "Smoke Student",
            "id": f"SMK-{program_id}",
            "class": "KXX",
            "major": program_id,
            "program_hint": program_id,
        },
        "semesters": [
            {"semester_code": "251", "subjects": rows[: len(rows) // 2 or 1]},
            {"semester_code": "252", "subjects": rows[len(rows) // 2 or 1 :]},
        ],
        "completed_subjects": rows,
        "overview": {
            "total_credits_accumulated": total_credits,
            "raw_gpa_4": 2.5,
        },
    }


def _choose_program_ids():
    data = json.loads(server.get_available_programs(refresh=True))
    programs = [p.get("id") for p in data.get("programs", []) if p.get("id")]
    if not programs:
        pytest.skip("No curriculum programs discovered.")
    selected = [pid for pid in TARGET_IDS if pid in programs]
    if not selected:
        selected = programs[:8]
    return selected


def test_ctdt_matrix_smoke_subset(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT1001\nINT4050\nINT3412E\n", "fake_tkb.pdf"),
    )

    program_ids = _choose_program_ids()
    registry = server._scan_curriculum_programs(force_refresh=False)

    for pid in program_ids:
        lookup = json.loads(server.get_curriculum_lookup(program_id=pid))
        assert "error" not in lookup, f"lookup failed for {pid}: {lookup.get('error')}"
        assert "total_credits_required" in lookup
        assert "groups" in lookup and isinstance(lookup["groups"], dict)

        curriculum = server.analyze_curriculum(program_hint=pid)
        expected = registry.get(pid)
        if expected and curriculum.get("source_path"):
            assert Path(curriculum["source_path"]).name == Path(expected["file_path"]).name

        subjects = _extract_subjects(lookup["groups"])
        transcript = _build_fake_transcript(pid, subjects)
        missing_info = server.compute_missing_subjects(transcript, curriculum)
        credit_summary = missing_info.get("credit_summary") or {}
        missing_credits = int(credit_summary.get("total_missing_credits") or 0)
        assert missing_credits >= 0

        electives = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id=pid))
        assert "error" not in electives, f"elective failed for {pid}: {electives.get('error')}"
        assert "opened" in electives and "not_opened" in electives
        assert isinstance(electives["opened"], list)
        assert isinstance(electives["not_opened"], list)
        assert electives.get("selection_mode") in {"token_matched_groups", "all_leaf_groups_fallback"}
        assert isinstance(electives.get("selected_group_codes") or [], list)
