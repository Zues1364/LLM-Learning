import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from mcp_server import server  # noqa: E402


def test_merge_transcript_with_deterministic_supplement_restores_missing_subjects_and_credits():
    transcript_payload = {
        "student_info": {"class": "QH-2022-I/CQ-I-CS2"},
        "semesters": [
            {
                "semester_code": "242",
                "semester_title": "Học kỳ 242",
                "subjects": [
                    {"code": "INT3117", "name": "Kiểm thử và đảm bảo chất lượng phần mềm", "credits": 3, "grade_letter": "C", "grade_4": 2.0},
                    {"code": "INT3231", "name": "Công nghệ Blockchain", "credits": 4, "grade_letter": "A", "grade_4": 3.7},
                ],
            },
            {
                "semester_code": "241",
                "semester_title": "Học kỳ 241",
                "subjects": [
                    {"code": "INT3105", "name": "Kiến trúc phần mềm", "credits": 3, "grade_letter": "C", "grade_4": 2.0},
                ],
            },
        ],
        "overview": {"total_credits_accumulated": 95},
    }
    text_entries = [
        {
            "file_id": "1.pdf",
            "text": "\n".join(
                [
                    "HỌC KỲ 2 - 2024-2025. MÃ HỌC KỲ 242 |  |  |  |  |  |  |  |  |",
                    "1 | INT3117 |  | Kiểm thử và đảm bảo chất lượng phần mềm | 3 | 6 | C | 2 |  |",
                    "2 | INT3401E |  | Trí tuệ nhân tạo | 3 | 9 | A+ | 4 |  |",
                    "3 | INT3231 |  | Công nghệ Blockchain | 4 | 8.6 | A | 3.7 |  |",
                    "4 | INT2214 |  | Nguyên lý hệ điều hành | 4 | 9.2 | A+ | 4 |  |",
                    "5 | INT3011E |  | Các vấn đề hiện đại trong KHMT | 3 | 9 | A+ | 4 |  |",
                    "6 | INT3425 |  | Khoa học dữ liệu | 3 | 6.6 | C+ | 2.5 |  |",
                    "Sinh viên: Nguyễn Tuấn Dương Mã số: 22028230 Lớp quản lý: QH-2022-I/CQ-I-CS2",
                    "Tổng tín chỉ: 115",
                    "Tổng tín chỉ tích lũy: 115",
                ]
            ),
        }
    ]

    merged = server._merge_transcript_with_deterministic_supplement(transcript_payload, text_entries)
    completed = server._build_completed_subjects(merged.get("semesters") or [])

    assert int((merged.get("overview") or {}).get("total_credits_accumulated") or 0) == 115
    assert "INT2214" in completed
    assert "INT3401E" in completed
    assert "INT3011E" in completed
    assert "INT3425" in completed
