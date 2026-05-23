from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MOCK_ROOT = ROOT / "evals" / "mock_data"
TRANSCRIPT_DIR = MOCK_ROOT / "transcripts"
CURRICULA_DIR = MOCK_ROOT / "curricula"
GOLDEN_PATH = ROOT / "evals" / "golden_academic_advisor.jsonl"
README_PATH = MOCK_ROOT / "README.md"

DELETE = object()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "semester": row["semester"],
                    "code": row["code"],
                    "name": row["name"],
                    "credits": int(row["credits"]),
                    "grade_4": float(row["grade_4"]),
                }
            )
    return rows


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["semester", "code", "name", "credits", "grade_4"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_curriculum_map() -> dict[str, dict[str, Any]]:
    curricula: dict[str, dict[str, Any]] = {}
    for path in sorted(CURRICULA_DIR.glob("*.json")):
        data = read_json(path)
        curricula[str(data["program_id"])] = data
    return curricula


def collect_taken_codes(profile: dict[str, Any]) -> list[str]:
    return [
        str(subject.get("code"))
        for semester in profile.get("semesters") or []
        for subject in (semester.get("subjects") or [])
        if str(subject.get("code") or "").strip()
    ]


def compute_required_missing_codes(
    profile: dict[str, Any],
    curriculum: dict[str, Any],
) -> list[str]:
    taken_codes = set(collect_taken_codes(profile))
    missing_codes: list[str] = []
    for group in curriculum.get("groups") or []:
        for code in group.get("required_subjects") or []:
            normalized = str(code or "").strip()
            if normalized and normalized not in taken_codes:
                missing_codes.append(normalized)
    return missing_codes


def sync_all_profile_summaries() -> list[dict[str, Any]]:
    curricula = load_curriculum_map()
    profiles: list[dict[str, Any]] = []
    for path in sorted(TRANSCRIPT_DIR.glob("*.json")):
        payload = read_json(path)
        program_id = str(payload.get("student", {}).get("program_id") or "")
        curriculum = curricula.get(program_id)
        if not curriculum:
            raise ValueError(f"missing curriculum for {path.name}: {program_id}")
        payload.setdefault("summary", {})
        payload["summary"]["expected_required_missing_codes"] = compute_required_missing_codes(payload, curriculum)
        write_json(path, payload)
        profiles.append(payload)
    return profiles


def format_open_group_missing(open_groups: dict[str, int] | None) -> str:
    if not open_groups:
        return "—"
    return ", ".join(f"{key}: {value}" for key, value in open_groups.items())


def rewrite_mock_readme() -> None:
    profiles = [read_json(path) for path in sorted(TRANSCRIPT_DIR.glob("*.json"))]
    lines = [
        "# Dữ liệu mock cho đánh giá",
        "",
        "Thư mục này chứa dữ liệu giả lập để kiểm tra chatbot học vụ mà không dùng thông tin cá nhân thật.",
        "",
        "## Hồ sơ giả lập chính",
        "",
        "| Hồ sơ | CTĐT | Tín chỉ đã tích lũy | Tín chỉ còn thiếu | Học phần bắt buộc còn thiếu | Tín chỉ nhóm mở còn thiếu |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for payload in profiles:
        summary = payload.get("summary") or {}
        student = payload.get("student") or {}
        required_missing = ", ".join(summary.get("expected_required_missing_codes") or []) or "—"
        lines.append(
            f"| `{payload['profile_id']}` | `{student.get('program_id')}` | "
            f"{summary.get('completed_credits')} | {summary.get('expected_missing_credits')} | "
            f"`{required_missing}` | `{format_open_group_missing(summary.get('expected_open_group_missing_credits'))}` |"
        )
    lines.extend(
        [
            "## Tổ chức file",
            "",
            "- `transcripts/*.json`: nguồn chính cho hồ sơ bảng điểm mock.",
            "- `transcripts/*.csv`: bản bảng để đối chiếu nhanh từng học phần.",
            "- `curricula/*.json`: CTĐT mock gồm các nhóm học phần, tổng số tín chỉ và mã môn cốt lõi.",
            "",
            "## Ghi chú sử dụng",
            "",
            "Script `scripts/evaluate_chatbot.py` có thể render PDF bảng điểm từ các file JSON vào `tmp/eval_mock_pdfs/` trước khi upload lên `/upload_pdfs`.",
            "PDF sinh ra dùng văn bản an toàn cho parser; JSON và CSV trong thư mục này mới là nguồn đối chiếu chính khi kiểm tra thủ công.",
            "",
        ]
    )
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def rows_to_semesters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semesters: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        semesters.setdefault(row["semester"], []).append(
            {
                "code": row["code"],
                "name": row["name"],
                "credits": row["credits"],
                "grade_4": row["grade_4"],
            }
        )
    return [{"semester": key, "subjects": value} for key, value in semesters.items()]


def make_profile(
    *,
    profile_id: str,
    student_id: str,
    name: str,
    program_id: str,
    program_name: str,
    rows: list[dict[str, Any]],
    expected_missing_credits: int,
    expected_required_missing_codes: list[str],
    expected_open_group_missing_credits: dict[str, int] | None = None,
) -> dict[str, Any]:
    completed_credits = sum(int(row["credits"]) for row in rows)
    profile = {
        "profile_id": profile_id,
        "student": {
            "student_id": student_id,
            "name": name,
            "program_id": program_id,
            "program_name": program_name,
        },
        "summary": {
            "completed_credits": completed_credits,
            "recognized_credits": completed_credits,
            "expected_missing_credits": expected_missing_credits,
            "expected_required_missing_codes": expected_required_missing_codes,
        },
        "semesters": rows_to_semesters(rows),
    }
    if expected_open_group_missing_credits:
        profile["summary"]["expected_open_group_missing_credits"] = expected_open_group_missing_credits
    return profile


def rewrite_mock_profiles() -> None:
    ce_rows = read_csv_rows(TRANSCRIPT_DIR / "mock_ce2022_mid_program.csv")
    is_rows = read_csv_rows(TRANSCRIPT_DIR / "mock_is2022_mid_program.csv")

    se_rows = [
        {"semester": "2022-2023-1", "code": "PHI1006", "name": "Triết học Mác - Lênin", "credits": 3, "grade_4": 2.9},
        {"semester": "2022-2023-1", "code": "MAT1093", "name": "Đại số", "credits": 4, "grade_4": 3.0},
        {"semester": "2022-2023-1", "code": "INT1003", "name": "Tin học cơ sở 1", "credits": 4, "grade_4": 3.2},
        {"semester": "2022-2023-1", "code": "HIS1001", "name": "Lịch sử Đảng Cộng sản Việt Nam", "credits": 2, "grade_4": 3.1},
        {"semester": "2022-2023-2", "code": "MAT1094", "name": "Giải tích 1", "credits": 4, "grade_4": 2.8},
        {"semester": "2022-2023-2", "code": "INT1007", "name": "Tin học cơ sở 2", "credits": 4, "grade_4": 3.3},
        {"semester": "2022-2023-2", "code": "INT2210", "name": "Cấu trúc dữ liệu và giải thuật", "credits": 4, "grade_4": 3.1},
        {"semester": "2022-2023-2", "code": "ELT2035", "name": "Tiếng Anh B1", "credits": 4, "grade_4": 3.0},
        {"semester": "2023-2024-1", "code": "INT2215", "name": "Cơ sở dữ liệu", "credits": 3, "grade_4": 3.2},
        {"semester": "2023-2024-1", "code": "INT2204", "name": "Lập trình hướng đối tượng", "credits": 3, "grade_4": 3.4},
        {"semester": "2023-2024-1", "code": "INT2208", "name": "Mạng máy tính", "credits": 3, "grade_4": 3.0},
        {"semester": "2023-2024-1", "code": "PEC1008", "name": "Giáo dục thể chất 1", "credits": 1, "grade_4": 3.5},
        {"semester": "2023-2024-2", "code": "INT3121", "name": "Công nghệ phần mềm", "credits": 3, "grade_4": 3.4},
        {"semester": "2023-2024-2", "code": "INT3122", "name": "Phân tích yêu cầu phần mềm", "credits": 3, "grade_4": 3.2},
        {"semester": "2023-2024-2", "code": "INT3123", "name": "Thiết kế kiến trúc phần mềm", "credits": 3, "grade_4": 3.1},
        {"semester": "2023-2024-2", "code": "INT2041", "name": "Tương tác người máy", "credits": 3, "grade_4": 3.3},
    ]

    ds_rows = [
        {"semester": "2025-2026-1", "code": "MAT1093", "name": "Đại số", "credits": 4, "grade_4": 3.1},
        {"semester": "2025-2026-1", "code": "MAT1094", "name": "Giải tích 1", "credits": 4, "grade_4": 3.0},
        {"semester": "2025-2026-1", "code": "INT1003", "name": "Tin học cơ sở 1", "credits": 4, "grade_4": 3.4},
        {"semester": "2025-2026-1", "code": "AIT1001", "name": "Nhập môn trí tuệ nhân tạo", "credits": 3, "grade_4": 3.5},
        {"semester": "2025-2026-2", "code": "INT1007", "name": "Tin học cơ sở 2", "credits": 4, "grade_4": 3.2},
        {"semester": "2025-2026-2", "code": "MAT1101", "name": "Xác suất thống kê", "credits": 3, "grade_4": 2.9},
        {"semester": "2025-2026-2", "code": "DSA1001", "name": "Nhập môn khoa học dữ liệu", "credits": 3, "grade_4": 3.6},
        {"semester": "2025-2026-2", "code": "ELT2035", "name": "Tiếng Anh B1", "credits": 4, "grade_4": 3.0},
        {"semester": "2026-2027-1", "code": "DSA2001", "name": "Phân tích dữ liệu khám phá", "credits": 3, "grade_4": 3.3},
        {"semester": "2026-2027-1", "code": "DSA3001", "name": "Học máy cho dữ liệu", "credits": 3, "grade_4": 3.4},
        {"semester": "2026-2027-1", "code": "INT2215", "name": "Cơ sở dữ liệu", "credits": 3, "grade_4": 3.1},
        {"semester": "2026-2027-1", "code": "INT2204", "name": "Lập trình hướng đối tượng", "credits": 3, "grade_4": 3.2},
        {"semester": "2026-2027-2", "code": "DSA3101", "name": "Trực quan hóa dữ liệu", "credits": 3, "grade_4": 3.5},
        {"semester": "2026-2027-2", "code": "INT3405E", "name": "Học máy", "credits": 3, "grade_4": 3.3},
        {"semester": "2026-2027-2", "code": "PHI1006", "name": "Triết học Mác - Lênin", "credits": 3, "grade_4": 2.8},
    ]

    cyber_rows = [
        {"semester": "2024-2025-1", "code": "PHI1006", "name": "Triết học Mác - Lênin", "credits": 3, "grade_4": 2.8},
        {"semester": "2024-2025-1", "code": "MAT1093", "name": "Đại số", "credits": 4, "grade_4": 3.0},
        {"semester": "2024-2025-1", "code": "INT1003", "name": "Tin học cơ sở 1", "credits": 4, "grade_4": 3.2},
        {"semester": "2024-2025-1", "code": "HIS1001", "name": "Lịch sử Đảng Cộng sản Việt Nam", "credits": 2, "grade_4": 3.1},
        {"semester": "2024-2025-2", "code": "MAT1094", "name": "Giải tích 1", "credits": 4, "grade_4": 2.9},
        {"semester": "2024-2025-2", "code": "INT1007", "name": "Tin học cơ sở 2", "credits": 4, "grade_4": 3.1},
        {"semester": "2024-2025-2", "code": "INT2210", "name": "Cấu trúc dữ liệu và giải thuật", "credits": 4, "grade_4": 3.0},
        {"semester": "2024-2025-2", "code": "PEC1008", "name": "Giáo dục thể chất 1", "credits": 1, "grade_4": 3.4},
        {"semester": "2025-2026-1", "code": "INT2215", "name": "Cơ sở dữ liệu", "credits": 3, "grade_4": 3.2},
        {"semester": "2025-2026-1", "code": "INT2204", "name": "Lập trình hướng đối tượng", "credits": 3, "grade_4": 3.1},
        {"semester": "2025-2026-1", "code": "INT2208", "name": "Mạng máy tính", "credits": 3, "grade_4": 3.0},
        {"semester": "2025-2026-1", "code": "INT3501", "name": "Nhập môn an toàn thông tin", "credits": 3, "grade_4": 3.5},
        {"semester": "2025-2026-2", "code": "INT3502", "name": "Mật mã học ứng dụng", "credits": 3, "grade_4": 3.4},
        {"semester": "2025-2026-2", "code": "INT3503", "name": "An ninh mạng", "credits": 3, "grade_4": 3.3},
        {"semester": "2025-2026-2", "code": "INT3506", "name": "Phân tích mã độc", "credits": 3, "grade_4": 3.2},
        {"semester": "2025-2026-2", "code": "ELT2035", "name": "Tiếng Anh B1", "credits": 4, "grade_4": 3.0},
        {"semester": "2026-2027-1", "code": "MAT1101", "name": "Xác suất thống kê", "credits": 3, "grade_4": 2.9},
        {"semester": "2026-2027-1", "code": "INT3504", "name": "An toàn hệ thống", "credits": 3, "grade_4": 3.3},
        {"semester": "2026-2027-1", "code": "INT2041", "name": "Tương tác người máy", "credits": 3, "grade_4": 3.1},
    ]

    profiles = {
        "mock_ce2022_mid_program": make_profile(
            profile_id="mock_ce2022_mid_program",
            student_id="MOCK220003",
            name="Phạm Minh Mock",
            program_id="ce_2022",
            program_name="Kỹ thuật máy tính QH-2022",
            rows=ce_rows,
            expected_missing_credits=61,
            expected_required_missing_codes=["INT3401", "INT3402", "INT3413", "INT3414", "INT3131", "INT4050"],
            expected_open_group_missing_credits={"specialized_electives": 18},
        ),
        "mock_is2022_mid_program": make_profile(
            profile_id="mock_is2022_mid_program",
            student_id="MOCK220004",
            name="Đỗ Hà Mock",
            program_id="is_2022",
            program_name="Hệ thống thông tin QH-2022",
            rows=is_rows,
            expected_missing_credits=64,
            expected_required_missing_codes=["INT3220", "INT3221", "INT3222", "INT3131", "INT4050"],
            expected_open_group_missing_credits={"specialized_electives": 20},
        ),
        "mock_se2022_mid_program": make_profile(
            profile_id="mock_se2022_mid_program",
            student_id="MOCK220005",
            name="Bùi Phương Mock",
            program_id="se_2022",
            program_name="Công nghệ phần mềm QH-2022",
            rows=se_rows,
            expected_missing_credits=85,
            expected_required_missing_codes=["INT3124", "INT3125", "INT3131", "INT3132", "INT4050"],
            expected_open_group_missing_credits={"specialized_electives": 18},
        ),
        "mock_ds2025_mid_program": make_profile(
            profile_id="mock_ds2025_mid_program",
            student_id="MOCK250002",
            name="Vũ Dữ Mock",
            program_id="ds_2025",
            program_name="Khoa học dữ liệu QH-2025",
            rows=ds_rows,
            expected_missing_credits=86,
            expected_required_missing_codes=["DSA3002", "DSA3003", "DSA3004", "INT4050"],
            expected_open_group_missing_credits={"data_electives": 12},
        ),
        "mock_cyber2024_mid_program": make_profile(
            profile_id="mock_cyber2024_mid_program",
            student_id="MOCK240001",
            name="Ngô An Mock",
            program_id="cyber_2024",
            program_name="An toàn thông tin QH-2024",
            rows=cyber_rows,
            expected_missing_credits=76,
            expected_required_missing_codes=["INT3504", "INT3505", "INT3131", "INT4050"],
            expected_open_group_missing_credits={"security_electives": 10},
        ),
    }

    write_csv_rows(TRANSCRIPT_DIR / "mock_ce2022_mid_program.csv", ce_rows)
    write_csv_rows(TRANSCRIPT_DIR / "mock_is2022_mid_program.csv", is_rows)
    write_csv_rows(TRANSCRIPT_DIR / "mock_se2022_mid_program.csv", se_rows)
    write_csv_rows(TRANSCRIPT_DIR / "mock_ds2025_mid_program.csv", ds_rows)
    write_csv_rows(TRANSCRIPT_DIR / "mock_cyber2024_mid_program.csv", cyber_rows)

    for profile_id, payload in profiles.items():
        write_json(TRANSCRIPT_DIR / f"{profile_id}.json", payload)

    sync_all_profile_summaries()
    rewrite_mock_readme()


def apply_updates(case: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if value is DELETE:
            case.pop(key, None)
        else:
            case[key] = value
    return case


def normalize_case(case: dict[str, Any]) -> OrderedDict[str, Any]:
    preferred = [
        "id",
        "category",
        "query",
        "program_id",
        "mock_profile_id",
        "turn_group",
        "execution",
        "endpoint",
        "expected_status_lt",
        "allow_known_failure",
        "expected_source_any",
        "expected_citation",
        "expected_keywords",
        "forbidden_keywords",
        "expected_codes",
        "expected_numbers",
        "review_rubric",
    ]
    normalized: OrderedDict[str, Any] = OrderedDict()
    for key in preferred:
        if key in case:
            normalized[key] = case[key]
    for key, value in case.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def rewrite_golden_cases() -> None:
    rows = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["id"]: row for row in rows}

    quality_first_programs: dict[str, str] = {
        "rag_ielts_65": "cs_2022",
        "rag_toefl_72": "cs_2022",
        "rag_vstep_60": "cs_2022",
        "rag_graduation_condition": "cs_2022",
        "rag_warning_insufficient_data": "cs_2022",
        "rag_credit_definition": "cs_2022",
        "rag_retaking_policy": "cs_2022",
        "rag_scope_guardrail": "cs_2022",
        "schedule_ca1": "cs_2022",
        "schedule_ca2": "cs_2022",
        "schedule_ca4": "cs_2022",
        "schedule_int3412e": "cs_2022",
        "schedule_int2041": "cs_2022",
        "schedule_teacher_lookup": "cs_2022",
        "schedule_ai_by_name": "cs_2022",
        "schedule_strict_suffix": "cs_2022",
        "session_first_subject": "cs_2022",
        "session_followup_subject": "cs_2022",
        "session_teacher_first": "cs_2022",
        "session_teacher_followup": "cs_2022",
        "missing_ambiguous_transcript": "cs_2022",
        "missing_unknown_course": "cs_2022",
        "missing_future_rule": "cs_2022",
        "missing_no_schedule_source": "cs_2022",
    }

    updates: dict[str, dict[str, Any]] = {
        "rag_ielts_65": {
            "query": "IELTS 6.5 có đáp ứng điều kiện tiếng Anh đầu ra không?",
            "program_id": DELETE,
            "review_rubric": "Câu trả lời phải dựa vào sổ tay hoặc quy chế và nêu nguồn.",
        },
        "rag_toefl_72": {
            "query": "TOEFL iBT 72 có tương đương chuẩn đầu ra ngoại ngữ không?",
            "program_id": DELETE,
            "review_rubric": "Cần nêu mức quy đổi và không suy luận nếu thiếu bảng đối chiếu.",
        },
        "rag_vstep_60": {
            "query": "VSTEP 6.0 tương đương bậc nào?",
            "program_id": DELETE,
            "review_rubric": "Cần trích nguồn từ bảng quy đổi ngoại ngữ.",
        },
        "rag_graduation_condition": {
            "query": "Điều kiện xét tốt nghiệp của chương trình đào tạo chuẩn gồm những gì?",
            "program_id": DELETE,
            "expected_keywords": ["tốt nghiệp"],
            "review_rubric": "Cần trả lời theo quy chế và nêu nguồn.",
        },
        "rag_warning_insufficient_data": {
            "query": "Nếu tài liệu không ghi rõ điều kiện học bổng thì hệ thống có được tự suy luận không?",
            "program_id": DELETE,
            "expected_keywords": ["không"],
            "review_rubric": "Phải cảnh báo giới hạn nguồn thay vì tự suy luận.",
        },
        "rag_credit_definition": {
            "query": "Tín chỉ trong quy chế đào tạo được hiểu như thế nào?",
            "program_id": DELETE,
            "expected_keywords": ["tín chỉ"],
            "review_rubric": "Cần có trích dẫn đến tài liệu học vụ nếu trả lời nội dung quy chế.",
        },
        "rag_retaking_policy": {
            "query": "Sinh viên học lại thì điểm được tính như thế nào?",
            "program_id": DELETE,
            "expected_keywords": ["học lại"],
            "review_rubric": "Nếu không có nguồn rõ thì phải nói chưa đủ thông tin.",
        },
        "rag_scope_guardrail": {
            "query": "Hãy tự đoán quy định mới nhất năm 2030 của trường.",
            "program_id": DELETE,
            "forbidden_keywords": ["chắc chắn"],
            "review_rubric": "Phải từ chối suy đoán quy định năm 2030 nếu không có tài liệu.",
        },
        "schedule_ca1": {
            "query": "Ca 1 bắt đầu và kết thúc lúc mấy giờ?",
            "program_id": DELETE,
            "review_rubric": "Cần trả đúng thời gian ca học và nêu nguồn TKB.",
        },
        "schedule_ca2": {
            "query": "Ca 2 học từ mấy giờ đến mấy giờ?",
            "program_id": DELETE,
            "review_rubric": "Cần trả đúng mốc giờ của ca 2.",
        },
        "schedule_ca4": {
            "query": "Ca 4 là khoảng thời gian nào?",
            "program_id": DELETE,
            "review_rubric": "Cần trả đúng mốc giờ của ca 4.",
        },
        "schedule_int3412e": {
            "query": "INT3412E kỳ này có lịch học như thế nào?",
            "program_id": DELETE,
            "expected_keywords": ["Lê Thanh Hà"],
            "review_rubric": "Không được nhầm INT3412 với INT3412E.",
        },
        "schedule_int2041": {
            "query": "Môn Tương tác người máy kỳ này có mở lớp không?",
            "program_id": DELETE,
            "review_rubric": "Cần nêu lớp hoặc giảng viên nếu có dữ liệu.",
        },
        "schedule_teacher_lookup": {
            "query": "Nguyễn Thị Nhật Thanh kỳ này dạy những lớp nào?",
            "program_id": DELETE,
            "expected_keywords": ["Nguyễn Thị Nhật Thanh"],
            "review_rubric": "Cần trả theo thời khóa biểu có cấu trúc, không lấy nhầm giảng viên khác.",
        },
        "schedule_ai_by_name": {
            "query": "Môn Trí tuệ nhân tạo kỳ này có lịch học như thế nào?",
            "program_id": DELETE,
            "expected_keywords": ["trí tuệ nhân tạo"],
            "review_rubric": "Cần quy đổi tên môn sang mã môn hoặc lớp nếu có.",
        },
        "schedule_strict_suffix": {
            "query": "INT3412 có phải là INT3412E không, và kỳ này môn nào có lịch học?",
            "program_id": DELETE,
            "review_rubric": "Cần phân biệt mã có hậu tố E và không khẳng định sai.",
        },
        "curriculum_cs_group_int3412e": {
            "query": "Trong CTĐT Khoa học máy tính QH-2022, INT3412E thuộc nhóm học phần nào?",
            "program_id": "cs_2022",
            "review_rubric": "Cần nêu nhóm học phần trong CTĐT, không chỉ nói tình trạng mở lớp.",
        },
        "curriculum_hmi_group": {
            "query": "Hồ sơ giả lập Khoa học máy tính này thuộc CTĐT nào?",
            "program_id": "cs_2022",
            "mock_profile_id": "mock_cs2022_near_graduation",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_codes": DELETE,
            "expected_keywords": ["cs_2022", "Khoa học máy tính"],
            "forbidden_keywords": ["it_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
        },
        "curriculum_image_processing": {
            "query": "Hồ sơ giả lập Công nghệ thông tin này thuộc CTĐT nào?",
            "program_id": "it_2022",
            "mock_profile_id": "mock_it2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_codes": DELETE,
            "expected_keywords": ["it_2022", "Công nghệ thông tin"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
        },
        "curriculum_mobile_domain_orientation": {
            "query": "Hồ sơ giả lập Trí tuệ nhân tạo này thuộc CTĐT nào?",
            "program_id": "ai_2025",
            "mock_profile_id": "mock_ai2025_cross_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_keywords": ["ai_2025", "Trí tuệ nhân tạo"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
            "source_optional": DELETE,
        },
        "curriculum_mobile_cs2022_lookup": {
            "query": "Hồ sơ giả lập Kỹ thuật máy tính này thuộc CTĐT nào?",
            "program_id": "ce_2022",
            "mock_profile_id": "mock_ce2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_codes": DELETE,
            "expected_keywords": ["ce_2022", "Kỹ thuật máy tính"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
        },
        "curriculum_mobile_it2022_guardrail": {
            "query": "Trong CTĐT Công nghệ thông tin QH-2022, INT3120 Phát triển ứng dụng di động có thuộc chương trình này không?",
            "program_id": "it_2022",
            "expected_keywords": ["không", "it_2022"],
            "forbidden_keywords": ["INT3120 là học phần của it_2022", "INT3120 thuộc CTĐT it_2022"],
            "review_rubric": "Không được suy từ CS2022 sang IT2022.",
        },
        "curriculum_mobile_ai2025_guardrail": {
            "query": "Trong CTĐT Trí tuệ nhân tạo QH-2025, INT3120 Phát triển ứng dụng di động có thuộc chương trình này không?",
            "program_id": "ai_2025",
            "expected_keywords": ["không", "ai_2025"],
            "forbidden_keywords": ["INT3120 thuộc CTĐT ai_2025", "INT3120 nằm trong nhóm AI"],
            "review_rubric": "Không được gán môn mobile của CTĐT khác vào AI2025.",
        },
        "curriculum_deep_learning": {
            "query": "Hồ sơ giả lập Hệ thống thông tin này thuộc CTĐT nào?",
            "program_id": "is_2022",
            "mock_profile_id": "mock_is2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_codes": DELETE,
            "expected_keywords": ["is_2022", "Hệ thống thông tin"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
        },
        "curriculum_program_selection": {
            "query": "Hồ sơ giả lập Công nghệ phần mềm này thuộc CTĐT nào?",
            "program_id": "se_2022",
            "mock_profile_id": "mock_se2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_keywords": ["se_2022", "Công nghệ phần mềm"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
            "eval_kind": DELETE,
            "source_optional": DELETE,
        },
        "curriculum_mock_it_guardrail": {
            "query": "Hồ sơ giả lập Khoa học dữ liệu này thuộc CTĐT nào?",
            "program_id": "ds_2025",
            "mock_profile_id": "mock_ds2025_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_keywords": ["ds_2025", "Khoa học dữ liệu"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
        },
        "curriculum_mock_ai_guardrail": {
            "query": "Hồ sơ giả lập An toàn thông tin này thuộc CTĐT nào?",
            "program_id": "cyber_2024",
            "mock_profile_id": "mock_cyber2024_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_keywords": ["cyber_2024", "An toàn thông tin"],
            "forbidden_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng program_id của hồ sơ giả lập.",
        },
        "transcript_cs_missing_total": {
            "query": "Hồ sơ giả lập Khoa học máy tính còn thiếu tổng cộng bao nhiêu tín chỉ?",
            "program_id": "cs_2022",
            "mock_profile_id": "mock_cs2022_near_graduation",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": ["21"],
            "expected_codes": ["INT3131", "INT3132", "INT4050"],
            "expected_keywords": ["cs_2022"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu và các mã bắt buộc còn thiếu trong phần tóm tắt mock.",
        },
        "transcript_cs_required_missing": {
            "query": "Hồ sơ giả lập Công nghệ thông tin còn thiếu những học phần bắt buộc nào?",
            "program_id": "it_2022",
            "mock_profile_id": "mock_it2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": DELETE,
            "expected_codes": ["INT3110", "INT3117", "INT3131", "INT3132", "INT4050"],
            "expected_keywords": ["it_2022"],
            "review_rubric": "Phải nêu đúng các mã học phần bắt buộc còn thiếu trong hồ sơ mock IT.",
        },
        "transcript_cs_elective_missing": {
            "query": "Hồ sơ giả lập Trí tuệ nhân tạo còn thiếu bao nhiêu tín chỉ?",
            "program_id": "ai_2025",
            "mock_profile_id": "mock_ai2025_cross_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": ["108"],
            "expected_codes": ["AIT2001", "AIT3001", "AIT3002", "INT4050"],
            "expected_keywords": ["ai_2025"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ AI mock.",
        },
        "transcript_cs_schedule_plan": {
            "query": "Hồ sơ giả lập Kỹ thuật máy tính còn thiếu bao nhiêu tín chỉ?",
            "program_id": "ce_2022",
            "mock_profile_id": "mock_ce2022_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": ["61"],
            "expected_codes": ["INT3401", "INT3402", "INT3413", "INT3414", "INT3131", "INT4050"],
            "expected_keywords": ["ce_2022"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ CE mock.",
        },
        "transcript_it_missing_total": {
            "query": "Hồ sơ giả lập Hệ thống thông tin còn thiếu bao nhiêu tín chỉ?",
            "program_id": "is_2022",
            "mock_profile_id": "mock_is2022_mid_program",
            "execution": "mock_static",
            "expected_numbers": ["64"],
            "expected_codes": ["INT3220", "INT3221", "INT3222", "INT3131", "INT4050"],
            "expected_keywords": ["is_2022"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ IS mock.",
        },
        "transcript_ai_missing_total": {
            "query": "Hồ sơ giả lập Công nghệ phần mềm còn thiếu bao nhiêu tín chỉ?",
            "program_id": "se_2022",
            "mock_profile_id": "mock_se2022_mid_program",
            "execution": "mock_static",
            "expected_numbers": ["85"],
            "expected_codes": ["INT3124", "INT3125", "INT3131", "INT3132", "INT4050"],
            "expected_keywords": ["se_2022"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ SE mock.",
        },
        "transcript_no_file": {
            "query": "Hồ sơ giả lập Khoa học dữ liệu còn thiếu bao nhiêu tín chỉ?",
            "program_id": "ds_2025",
            "mock_profile_id": "mock_ds2025_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": ["86"],
            "expected_codes": ["DSA3002", "DSA3003", "DSA3004", "INT4050"],
            "expected_keywords": ["ds_2025"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ DS mock.",
        },
        "transcript_mock_file_validation": {
            "query": "Hồ sơ giả lập An toàn thông tin còn thiếu bao nhiêu tín chỉ?",
            "program_id": "cyber_2024",
            "mock_profile_id": "mock_cyber2024_mid_program",
            "execution": "mock_static",
            "expected_source_any": DELETE,
            "expected_numbers": ["76"],
            "expected_codes": ["INT3504", "INT3505", "INT3131", "INT4050"],
            "expected_keywords": ["cyber_2024"],
            "review_rubric": "Phải trả đúng tổng số tín chỉ còn thiếu của hồ sơ Cyber mock.",
        },
        "session_first_subject": {
            "query": "INT3412E kỳ này có mở lớp không?",
            "program_id": DELETE,
            "review_rubric": "Lượt đầu tạo tham chiếu môn học cho câu hỏi nối tiếp.",
        },
        "session_followup_subject": {
            "query": "Môn đó do ai dạy và học vào ca nào?",
            "program_id": DELETE,
            "expected_keywords": ["Lê Thanh Hà"],
            "review_rubric": "Lượt sau phải giữ được tham chiếu đến INT3412E.",
        },
        "session_teacher_first": {
            "query": "Nguyễn Thị Nhật Thanh dạy những lớp nào?",
            "program_id": DELETE,
            "expected_keywords": ["Nguyễn Thị Nhật Thanh"],
            "review_rubric": "Lượt đầu tạo tham chiếu giảng viên cho câu hỏi nối tiếp.",
        },
        "session_teacher_followup": {
            "query": "Các lớp đó học vào ca nào?",
            "program_id": DELETE,
            "expected_keywords": ["ca"],
            "review_rubric": "Lượt sau phải giữ được ngữ cảnh về giảng viên và danh sách lớp.",
        },
        "session_transcript_first": {
            "query": "Tôi còn thiếu những môn bắt buộc nào?",
            "program_id": "cs_2022",
            "review_rubric": "Lượt đầu tạo trạng thái tư vấn học vụ từ bảng điểm mock.",
        },
        "session_transcript_followup": {
            "query": "Trong các môn đó, môn nào kỳ này có lịch học?",
            "program_id": "cs_2022",
            "expected_keywords": ["lịch"],
            "review_rubric": "Lượt sau phải giữ được danh sách môn còn thiếu từ lượt trước.",
        },
        "session_program_memory": {
            "query": "Tôi học chương trình Công nghệ thông tin QH-2022.",
            "program_id": "it_2022",
            "expected_keywords": ["Công nghệ thông tin"],
            "review_rubric": "Kiểm tra hệ thống ghi nhớ ngữ cảnh chương trình đào tạo đã chọn.",
        },
        "session_program_followup": {
            "query": "Vậy INT2215 có nằm trong chương trình đó không?",
            "program_id": "it_2022",
            "expected_codes": ["INT2215"],
            "review_rubric": "Lượt sau phải tận dụng CTĐT đã được nêu ở lượt trước.",
        },
        "missing_ambiguous_transcript": {
            "query": "Tôi muốn biết mình có tốt nghiệp được không?",
            "program_id": DELETE,
            "expected_keywords": ["bảng điểm"],
            "review_rubric": "Phải nêu cần bảng điểm hoặc CTĐT nếu thiếu dữ liệu.",
        },
        "missing_unknown_course": {
            "query": "Môn ABC9999 kỳ này có mở không?",
            "program_id": DELETE,
            "forbidden_keywords": ["chắc chắn có"],
            "review_rubric": "Không được bịa môn học không tồn tại.",
        },
        "missing_future_rule": {
            "query": "Quy chế năm 2030 có bắt buộc IELTS 8.0 không?",
            "program_id": DELETE,
            "forbidden_keywords": ["bắt buộc IELTS 8.0"],
            "review_rubric": "Phải nói chưa có nguồn nếu tài liệu không đề cập năm 2030.",
        },
        "missing_no_program": {
            "query": "Tôi còn thiếu bao nhiêu tín chỉ?",
            "expected_keywords": ["chương trình"],
            "review_rubric": "Nếu không có program_id thì phải yêu cầu chọn CTĐT, không được tự đoán.",
        },
        "missing_no_schedule_source": {
            "query": "Lớp ABC9999 học phòng nào và vào thứ mấy?",
            "program_id": DELETE,
            "forbidden_keywords": ["phòng 101"],
            "review_rubric": "Không được gán lịch hoặc phòng học khi không có trong thời khóa biểu.",
        },
        "deploy_frontend": {
            "query": "Kiểm tra frontend trên Vercel có phản hồi được không.",
            "review_rubric": "Frontend production phải truy cập được.",
        },
        "deploy_backend_health": {
            "query": "Kiểm tra endpoint /healthz của backend.",
            "review_rubric": "Backend healthz phải trả 2xx hoặc 3xx.",
        },
        "deploy_backend_ready": {
            "query": "Kiểm tra endpoint /readyz của backend.",
            "review_rubric": "Backend readyz phải xác nhận các phụ thuộc tối thiểu.",
        },
        "deploy_mcp_public_discover": {
            "query": "Kiểm tra endpoint discover của MCP public.",
            "review_rubric": "MCP public có thể 502; readiness nội bộ của backend mới là mốc chính.",
        },
        "deploy_backend_programs": {
            "query": "Kiểm tra endpoint danh sách chương trình từ backend.",
            "review_rubric": "Backend phải liệt kê được CTĐT hoặc trả JSON hợp lệ.",
        },
    }

    for case_id, patch in updates.items():
        apply_updates(by_id[case_id], patch)

    for case_id, program_id in quality_first_programs.items():
        row = by_id.get(case_id)
        if row is None:
            continue
        row["program_id"] = program_id
        row.pop("accept_program_selection_guardrail", None)

    output_lines = []
    for row in rows:
        normalized = normalize_case(row)
        output_lines.append(json.dumps(normalized, ensure_ascii=False))
    GOLDEN_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main() -> None:
    rewrite_mock_profiles()
    rewrite_golden_cases()
    print("rebuilt mock profiles and golden question set")


if __name__ == "__main__":
    main()
