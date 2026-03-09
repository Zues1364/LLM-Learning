import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402


def test_get_curriculum_lookup_parses_old_program_subject_rows():
    data = json.loads(server.get_curriculum_lookup(program_id="ckt_2019"))
    assert "error" not in data, data.get("error")
    groups = data.get("groups") or {}
    assert isinstance(groups, dict)

    total_subjects = sum(len((group.get("subjects") or [])) for group in groups.values())
    assert total_subjects > 0, "Expected ckt_2019 to expose parsed subjects from curriculum."


def test_get_electives_with_schedule_aut_2022_has_no_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT1001\nINT4050\nINT3412E\n", "fake_tkb.pdf"),
    )

    data = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id="aut_2022"))
    assert "error" not in data, data.get("error")
    assert "opened" in data and "not_opened" in data
    assert data.get("selection_mode") in {"token_matched_groups", "all_leaf_groups_fallback"}
    assert isinstance(data.get("selected_group_codes") or [], list)

