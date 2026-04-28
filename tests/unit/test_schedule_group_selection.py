from src.mcp_server.server import _select_groups_for_schedule


def test_select_groups_propagates_parent_elective_tokens_and_excludes_non_elective_blocks():
    groups_data = {
        "I": {
            "group_name": "Khoi kien thuc chung",
            "notes": [{"text": "Ky nang bo tro"}],
            "subjects": [{"code": "PHI1002"}],
        },
        "V.2": {
            "group_name": "Cac hoc phan tu chon theo cac dinh huong",
            "notes": [],
            "subjects": [],
        },
        "V.2.1": {
            "group_name": "Dinh huong he thong thong tin",
            "notes": [],
            "subjects": [{"code": "INT3306"}],
        },
        "V.2.2": {
            "group_name": "Dinh huong khoa hoc du lieu",
            "notes": [],
            "subjects": [{"code": "INT3229E"}],
        },
        "V.4": {
            "group_name": "Khoa luan tot nghiep",
            "notes": [{"text": "3 tin chi tu danh sach tu chon"}],
            "subjects": [{"code": "INT4050"}],
        },
    }

    mode, selected = _select_groups_for_schedule(groups_data)

    assert mode == "token_matched_groups"
    assert set(selected) == {"V.2.1", "V.2.2"}


def test_select_groups_falls_back_to_all_leaf_groups_if_no_token_match():
    groups_data = {
        "II.1": {
            "group_name": "Khoi mon hoc co so",
            "notes": [],
            "subjects": [{"code": "MAT1093"}],
        },
        "II.2": {
            "group_name": "Khoi mon hoc theo linh vuc",
            "notes": [],
            "subjects": [{"code": "INT1008"}],
        },
    }

    mode, selected = _select_groups_for_schedule(groups_data)

    assert mode == "all_leaf_groups_fallback"
    assert set(selected) == {"II.1", "II.2"}
