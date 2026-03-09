import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402


def _mock_lookup_with_subject(code: str, name: str = "Mock Subject") -> str:
    payload = {
        "total_credits_required": 0,
        "groups": {
            "V.2.1": {
                "group_code": "V.2.1",
                "group_name": "Hoc phan tu chon",
                "subjects": [{"code": code, "name": name, "credits": 3}],
                "credits_required": 3,
            }
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def test_e_suffix_does_not_match_non_e_code(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_curriculum_lookup",
        lambda group_hint=None, program_id=None: _mock_lookup_with_subject("INT3404E", "Xu ly anh"),
    )
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT3404 Lop 1", "fake_schedule.pdf"),
    )

    data = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id="mock"))
    assert "error" not in data
    assert int(data.get("opened_count") or 0) == 0
    assert int(data.get("not_opened_count") or 0) == 1
    assert (data.get("not_opened") or [{}])[0].get("code") == "INT3404E"


def test_exact_e_code_matches_when_present(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_curriculum_lookup",
        lambda group_hint=None, program_id=None: _mock_lookup_with_subject("INT3404E", "Xu ly anh"),
    )
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT3404E Lop 1", "fake_schedule.pdf"),
    )

    data = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id="mock"))
    assert "error" not in data
    assert int(data.get("opened_count") or 0) == 1
    assert int(data.get("not_opened_count") or 0) == 0
    assert (data.get("opened") or [{}])[0].get("code") == "INT3404E"

