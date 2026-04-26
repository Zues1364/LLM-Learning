import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.structured_schedule_store as schedule_store  # noqa: E402


def _fake_schedule_docs(_path: str):
    text = "\n".join(
        [
            "K69I-CS1 PEC1008 Kinh tế chính trị Mác – Lênin 2 20 10 9 65 PEC1008 1 CL LT 4 3 503-B Ngô Thái Hà Học 1 ca/10 tuần, thi đợt 1",
            "K69I-CS2 PEC1008 Kinh tế chính trị Mác – Lênin 2 20 10 9 65 PEC1008 2 CL LT 2 4 107-B Nguyễn Thị Hồng Sâm Học 1 ca/10 tuần, thi đợt 1",
            "K68I-IS1 HIS1001 Lịch sử Đảng Cộng sản Việt Nam 2 20 10 9 72 HIS1001 5 CL LT 2 1 103-A Vũ Thị Thu Hà Học 1 ca/10 tuần, thi đợt 1",
            "K68I-IS1 HIS1001 Lịch sử Đảng Cộng sản Việt Nam 2 20 10 9 72 HIS1001 5 CL LT 2 1 103-A Vũ Thị Thu Hà Học 1 ca/10 tuần, thi đợt 1",
        ]
    )
    return [Document(page_content=text, metadata={"file_name": "TKB.pdf", "index": 1, "page": 5})]


def test_structured_schedule_store_ingest_and_lookup(monkeypatch, tmp_path):
    monkeypatch.setattr(schedule_store, "process_pdf", _fake_schedule_docs)
    db_path = tmp_path / "structured_schedule.db"
    pdf_path = tmp_path / "PHU_LUC_TKB.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    store = schedule_store.StructuredScheduleStore(db_path=db_path)
    ingest = store.ingest_schedule_files([pdf_path], force=True)

    assert ingest
    assert ingest[0]["row_count"] >= 3

    alias = store.resolve_course_alias("môn kinh tế chính trị kỳ này")
    assert alias["matched_subject"]["subject_code"] == "PEC1008"
    assert alias["confidence"] >= 0.7

    teachers_payload = store.get_teachers_by_subject("PEC1008")
    assert "Ngô Thái Hà" in teachers_payload["teachers"]
    assert "Nguyễn Thị Hồng Sâm" in teachers_payload["teachers"]
    assert teachers_payload["rows"]
    assert all(row.get("source_page") == 5 for row in teachers_payload["rows"])
    assert all((row.get("source_line") or 0) > 0 for row in teachers_payload["rows"])

    reverse_payload = store.get_classes_by_teacher("Vũ Thị Thu Hà")
    assert reverse_payload["rows"]
    assert any(row["subject_code"] == "HIS1001" for row in reverse_payload["rows"])


def test_structured_schedule_store_checksum_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(schedule_store, "process_pdf", _fake_schedule_docs)
    db_path = tmp_path / "structured_schedule.db"
    pdf_path = tmp_path / "PHU_LUC_TKB.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    store = schedule_store.StructuredScheduleStore(db_path=db_path)
    first = store.ingest_schedule_files([pdf_path], force=False)
    second = store.ingest_schedule_files([pdf_path], force=False)

    assert first and first[0]["skipped"] is False
    assert second and second[0]["skipped"] is True


def test_resolve_course_alias_prefers_higher_score_candidate(monkeypatch, tmp_path):
    def _fake_ambiguous_docs(_path: str):
        text = "\n".join(
            [
                "K68I-CS1 INT2213 Mạng máy tính 3 30 15 8 80 INT2213 1 CL LT 2 4 307-A Đào Minh Thư Học 1 ca/15 tuần",
                "K68I-CS1 INT3412E Thị giác máy 3 30 15 8 80 INT3412E 1 CL LT 3 2 209-T Lê Thanh Hà Học 1 ca/15 tuần",
            ]
        )
        return [Document(page_content=text, metadata={"file_name": "TKB.pdf", "index": 1})]

    monkeypatch.setattr(schedule_store, "process_pdf", _fake_ambiguous_docs)
    db_path = tmp_path / "structured_schedule.db"
    pdf_path = tmp_path / "PHU_LUC_TKB.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    store = schedule_store.StructuredScheduleStore(db_path=db_path)
    store.ingest_schedule_files([pdf_path], force=True)

    alias = store.resolve_course_alias("thị giác máy tính")
    assert alias.get("candidates")
    scores = [float(item.get("score") or 0.0) for item in alias["candidates"]]
    assert scores[0] == max(scores)
    assert alias["matched_subject"]["subject_code"] == "INT3412E"


def test_parse_teacher_list_filters_ocr_noise_and_repeated_tokens():
    teachers = schedule_store._parse_teacher_list(
        "Ngo Thai Ngo Thai Ha Ha Ha Ha, Vu Thi Thu Ha Thi Thu Ha Thi Thu Ha, t t u u a a n n, Nguyen Van Vinh, Nguyen Van Vinh"
    )

    assert "Ngo Thai Ha" in teachers
    assert "Vu Thi Thu Ha" in teachers
    assert "Nguyen Van Vinh" in teachers
    assert all("t t u u" not in t for t in teachers)
    assert len(teachers) == 3


def test_ingest_schedule_files_prunes_removed_sources(monkeypatch, tmp_path):
    def _fake_docs_by_filename(path: str):
        p = Path(path)
        if p.name == "TKB_A.pdf":
            text = "K68I-CS1 INT1001 Subject A 3 30 15 8 80 INT1001 1 CL LT 2 1 101-A Nguyen Van A Hoc 1 ca/15 tuan"
        else:
            text = "K68I-CS1 INT1002 Subject B 3 30 15 8 80 INT1002 1 CL LT 3 2 102-A Nguyen Van B Hoc 1 ca/15 tuan"
        return [Document(page_content=text, metadata={"file_name": p.name, "index": 1})]

    monkeypatch.setattr(schedule_store, "process_pdf", _fake_docs_by_filename)
    db_path = tmp_path / "structured_schedule.db"
    file_a = tmp_path / "TKB_A.pdf"
    file_b = tmp_path / "TKB_B.pdf"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")

    store = schedule_store.StructuredScheduleStore(db_path=db_path)
    store.ingest_schedule_files([file_a, file_b], force=True)

    rows_a = store.get_schedule_rows(subject_code="INT1001").get("rows") or []
    rows_b = store.get_schedule_rows(subject_code="INT1002").get("rows") or []
    assert rows_a
    assert rows_b

    second_summary = store.ingest_schedule_files([file_a], force=False)
    assert any(item.get("removed") is True and item.get("source_file") == "TKB_B.pdf" for item in second_summary)

    rows_b_after = store.get_schedule_rows(subject_code="INT1002").get("rows") or []
    assert rows_b_after == []


def test_semester_code_filter_supports_251_and_252_mapping(monkeypatch, tmp_path):
    def _fake_docs_semesters(path: str):
        p = Path(path)
        if "HKI" in p.name:
            text = (
                "K69I-CS1 INT2041 Tuong tac nguoi may 3 30 15 8 80 "
                "INT2041 1 CL LT 5 1 206-T Ngo Thi Duyen Hoc 1 ca/15 tuan"
            )
        elif "HKII" in p.name:
            text = (
                "K69I-CS1 INT2041 Tuong tac nguoi may 3 30 15 8 80 "
                "INT2041 2 CL LT 5 2 3-G3 Ngo Thi Duyen Hoc 1 ca/15 tuan"
            )
        else:
            text = (
                "K69I-CS1 INT2041 Tuong tac nguoi may 3 30 15 8 80 "
                "INT2041 3 CL LT 6 2 105-G2 Tran Van B Hoc 1 ca/15 tuan"
            )
        return [Document(page_content=text, metadata={"file_name": p.name, "index": 1})]

    monkeypatch.setattr(schedule_store, "process_pdf", _fake_docs_semesters)
    db_path = tmp_path / "structured_schedule.db"
    file_hk1 = tmp_path / "PHU_LUC_TKB_HKI_2025-2026.pdf"
    file_hk2 = tmp_path / "PHU_LUC_TKB_HKII_2025-2026.pdf"
    file_hk3 = tmp_path / "PHU_LUC_TKB_HKIII_2025-2026.pdf"
    file_hk1.write_text("hk1", encoding="utf-8")
    file_hk2.write_text("hk2", encoding="utf-8")
    file_hk3.write_text("hk3", encoding="utf-8")

    store = schedule_store.StructuredScheduleStore(db_path=db_path)
    store.ingest_schedule_files([file_hk1, file_hk2, file_hk3], force=True)

    rows_251 = store.get_schedule_rows(subject_code="INT2041", semester="251").get("rows") or []
    assert rows_251
    assert all("HKI" in str(row.get("semester") or "") for row in rows_251)
    assert all("HKII" not in str(row.get("semester") or "") for row in rows_251)
    assert all("HKIII" not in str(row.get("semester") or "") for row in rows_251)

    rows_252 = store.get_schedule_rows(subject_code="INT2041", semester="252").get("rows") or []
    assert rows_252
    semesters_252 = {str(row.get("semester") or "") for row in rows_252}
    assert any("HKII" in sem for sem in semesters_252)
    assert any("HKIII" in sem for sem in semesters_252)
    assert not any("HKI" in sem and "HKII" not in sem and "HKIII" not in sem for sem in semesters_252)
