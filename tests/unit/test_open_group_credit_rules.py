import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import utils  # noqa: E402
from mcp_server.server import compute_missing_subjects  # noqa: E402


def _curriculum_structure(with_open_group_note: bool):
    open_note = []
    if with_open_group_note:
        open_note = [
            {
                "text": "Các học phần thuộc các nhóm ngành Điện tử-viễn thông, Kinh tế, Luật",
                "norm": "cac hoc phan thuoc cac nhom nganh dien tu vien thong kinh te luat",
            }
        ]

    return [
        {
            "id": "I",
            "name": "Khối kiến thức ngành",
            "required_credits": 0,
            "subjects": [],
            "sub_blocks": [
                {
                    "id": "I.1",
                    "name": "Các học phần bắt buộc",
                    "required_credits": 21,
                    "subjects": [
                        {
                            "code": "INT9999",
                            "name": "Mon bat buoc chua hoc",
                            "credits": 21,
                        }
                    ],
                    "notes": [],
                },
                {
                    "id": "I.2",
                    "name": "Các học phần bổ trợ",
                    "required_credits": 3,
                    "subjects": [],
                    "notes": open_note,
                },
            ],
        }
    ]


def _completed_map_with_bsa2002():
    return {
        "BSA2002": {
            "code": "BSA2002",
            "name": "Nguyen ly marketing",
            "credits": 3,
            "grade_4": 3.0,
        }
    }


def test_parse_curriculum_attaches_open_group_note_to_active_sub_block():
    html = """
    <html>
      <body>
        <table>
          <tr>
            <td>STT</td><td>Mã học phần</td><td>Tên học phần</td><td>Số tín chỉ</td>
          </tr>
          <tr><td>I</td><td>Khối kiến thức ngành</td><td>24</td></tr>
          <tr><td>I.1</td><td>Các học phần bắt buộc</td><td>21</td></tr>
          <tr><td>1</td><td>INT9999</td><td>Mon bat buoc chua hoc</td><td>21</td></tr>
          <tr><td>I.2</td><td>Các học phần bổ trợ</td><td>3</td></tr>
          <tr><td>72</td><td>Các học phần thuộc các nhóm ngành Điện tử-viễn thông, Kinh tế, Luật</td></tr>
        </table>
      </body>
    </html>
    """

    structure = utils.parse_curriculum_from_html_content(html)
    assert structure, "expected parsed structure"
    assert structure[0]["sub_blocks"], "expected sub-blocks"

    sub_i2 = next(sb for sb in structure[0]["sub_blocks"] if sb.get("id") == "I.2")
    notes = sub_i2.get("notes") or []
    assert notes, "expected open-group note on sub-block I.2"
    assert "kinh te" in notes[0]["norm"]
    assert "luat" in notes[0]["norm"]


def test_compute_curriculum_missing_credits_applies_external_subject_when_note_allows():
    structure = _curriculum_structure(with_open_group_note=True)
    completed_map = _completed_map_with_bsa2002()

    details = utils.compute_curriculum_missing_credits(structure, completed_map)
    sub_i2 = next(item for item in details if item.get("block_id") == "I.2")

    assert sub_i2["missing_credits"] == 0
    assert sub_i2["completed_credits"] == 3
    assert sub_i2["applied_external_subjects"], "expected external subject application"
    assert sub_i2["applied_external_subjects"][0]["code"] == "BSA2002"
    assert sub_i2["applied_external_subjects"][0]["counted_credits"] == 3


def test_compute_curriculum_missing_credits_does_not_apply_external_without_note():
    structure = _curriculum_structure(with_open_group_note=False)
    completed_map = _completed_map_with_bsa2002()

    details = utils.compute_curriculum_missing_credits(structure, completed_map)
    sub_i2 = next(item for item in details if item.get("block_id") == "I.2")

    assert sub_i2["missing_credits"] == 3
    assert sub_i2["completed_credits"] == 0
    assert sub_i2["applied_external_subjects"] == []


def test_compute_missing_subjects_credit_summary_counts_open_group_external_credit():
    transcript = {
        "semesters": [
            {
                "semester_code": "251",
                "subjects": [
                    {
                        "code": "BSA2002",
                        "name": "Nguyen ly marketing",
                        "credits": 3,
                        "grade_4": 3.0,
                    }
                ],
            }
        ],
        "overview": {"total_credits_accumulated": 115},
    }
    curriculum = {
        "program_name": "cs_2022",
        "subjects": [],
        "structure": _curriculum_structure(with_open_group_note=True),
        "total_credits": 136,
    }

    result = compute_missing_subjects(transcript, curriculum)
    credit_summary = result["credit_summary"]

    assert credit_summary["transcript_total_credits"] == 115
    assert credit_summary["total_required_credits"] == 136
    assert credit_summary["total_missing_credits"] == 21
    assert credit_summary["total_completed_applicable_credits"] == 115
    assert credit_summary["external_credits_applied"], "expected external credit list"
    assert credit_summary["external_credits_applied"][0]["code"] == "BSA2002"
