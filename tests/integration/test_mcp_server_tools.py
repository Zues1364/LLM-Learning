import json
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402


def _make_docs(pdf_name: str) -> list[Document]:
    return [
        Document(page_content="alpha", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 1}),
        Document(page_content="beta", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 2}),
    ]


def test_ensure_file_loaded_uses_cached_embeddings(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_text("fake pdf", encoding="utf-8")

    monkeypatch.setattr(server, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(server, "_loaded_files", set())
    monkeypatch.setattr(server, "_store", None)
    monkeypatch.setattr(server, "_embedder", None)

    class DummyMemory:
        def get_summary(self, fid):
            return None

        def save_summary(self, fid, summary):
            self.saved = (fid, summary)

    dummy_mem = DummyMemory()
    monkeypatch.setattr(server, "_memory", dummy_mem)

    class DummyServerEmbedder:
        def __init__(self):
            self.model_name = "dummy-server"

    monkeypatch.setattr(server, "VietnameseEmbedder", DummyServerEmbedder)
    monkeypatch.setattr(server, "process_pdf", lambda path: _make_docs(Path(path).name))

    load_calls = {"count": 0}

    def fake_load_embeddings(path, embedder, docs):
        load_calls["count"] += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(server, "load_embeddings_with_cache", fake_load_embeddings)

    resolved = server._ensure_file_loaded(pdf_path.name)
    assert resolved == pdf_path.name
    assert load_calls["count"] == 1
    assert server._store is not None
    assert len(server._store.documents) == 2

    # Second call should not reload
    server._ensure_file_loaded(pdf_path.name)
    assert load_calls["count"] == 1


def test_retrieve_chunks_formats_context(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_text("fake pdf", encoding="utf-8")

    monkeypatch.setattr(server, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(server, "_loaded_files", set())
    monkeypatch.setattr(server, "_store", None)

    class DummyMemory:
        def get_summary(self, fid):
            return None

        def save_summary(self, fid, summary):
            pass

    monkeypatch.setattr(server, "_memory", DummyMemory())

    class DummyEmbedder:
        def __init__(self):
            self.model_name = "dummy"

        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(server, "VietnameseEmbedder", DummyEmbedder)

    docs = _make_docs(pdf_path.name)
    monkeypatch.setattr(server, "process_pdf", lambda path: docs)

    def fake_load_embeddings(path, embedder, docs_arg):
        return np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")

    monkeypatch.setattr(server, "load_embeddings_with_cache", fake_load_embeddings)

    contexts = server.retrieve_chunks("alpha", top_k=1, file_ids=[pdf_path.name])
    assert len(contexts) >= 1
    assert all("Chunk" in ctx for ctx in contexts)
    assert any("alpha" in ctx for ctx in contexts)


def test_consult_advisor_prioritizes_explicit_program_id(monkeypatch):
    captured = {}

    class DummyMemory:
        def get_context(self, query, session_id="default", max_rows=10):
            return ""

    monkeypatch.setattr(server, "_memory", DummyMemory())
    monkeypatch.setattr(server, "_load_session_file_ids", lambda session_id: ["mock_transcript.pdf"])
    monkeypatch.setattr(
        server,
        "_load_schedule_time_slot_map",
        lambda force_refresh=False: ({}, ""),
    )
    monkeypatch.setattr(
        server,
        "analyze_transcript",
        lambda ids: json.dumps(
            {
                "student_info": {"class": "K67-CS", "major": "Khoa hoc may tinh"},
                "semesters": [
                    {
                        "semester_code": "251",
                        "subjects": [
                            {"code": "INT1001", "name": "Nhap mon", "credits": 3, "grade_4": 3.0}
                        ],
                    }
                ],
                "completed_subjects": [
                    {"code": "INT1001", "name": "Nhap mon", "credits": 3, "grade_4": 3.0}
                ],
                "overview": {"total_credits_accumulated": 0},
            }
        ),
    )
    monkeypatch.setattr(
        server,
        "compute_missing_subjects",
        lambda transcript_data, curriculum: {
            "missing": [],
            "completed_map": {},
            "low_grades": [],
            "credit_summary": {
                "transcript_total_credits": 0,
                "total_required_credits": 0,
                "total_completed_applicable_credits": 0,
                "total_missing_credits": 0,
                "external_credits_applied": [],
            },
            "credit_analysis": [],
        },
    )
    monkeypatch.setattr(server, "check_course_schedule", lambda subjects, target_semester=None, class_code=None: [])

    def fake_analyze_curriculum(program_hint=None):
        captured["program_hint"] = program_hint
        return {
            "program_name": program_hint,
            "subjects": [],
            "structure": [],
            "total_credits": None,
            "source_path": None,
            "notes": "stub",
        }

    monkeypatch.setattr(server, "analyze_curriculum", fake_analyze_curriculum)

    def fake_get_electives_with_schedule(check_schedule=True, program_id=None):
        captured["elective_program_id"] = program_id
        return json.dumps({"opened": []})

    monkeypatch.setattr(server, "get_electives_with_schedule", fake_get_electives_with_schedule)

    class DummyAdvisorAgent:
        def run(self, prompt):
            captured["advisor_prompt"] = prompt
            return type("Resp", (), {"content": "advisor-ok"})()

    monkeypatch.setattr(server, "get_academic_advisor_agent", lambda: DummyAdvisorAgent())

    result = server.consult_advisor(
        query="toi nen hoc mon nao",
        file_ids=[],
        session_id="s100",
        program_id="cs_2022",
    )

    assert result == "advisor-ok"
    assert captured["program_hint"] == "cs_2022"
    assert captured["elective_program_id"] == "cs_2022"
    context_blob = captured["advisor_prompt"].split("--- CONTEXT ---\n", 1)[1].split("\n--- END ---", 1)[0]
    advisor_context = json.loads(context_blob)
    assert advisor_context["schedule_table_columns"][1] == "Ca học"
    assert advisor_context["schedule_table_columns"][2] == "Tiết + Thời gian"


def test_extract_time_slot_map_parses_standard_table():
    text = """
| Buoi | Ca | Tiet | Thoi gian hoc | Ghi chu |
| --- | --- | --- | --- | --- |
| Sang | 1 | Tiet 1-3 | 07:00 – 09:40 | break |
| Sang | 2 | Tiet 4-6 | 09:50 – 12:30 | break |
| Chieu | 3 | Tiet 7-9 | 13:30 – 16:10 | break |
| Chieu | 4 | Tiet 10-12 | 16:20 – 19:00 | break |
"""
    slot_map = server._extract_time_slot_map(text)
    assert slot_map["1"]["time_range"] == "07:00 – 09:40"
    assert slot_map["2"]["time_range"] == "09:50 – 12:30"
    assert slot_map["3"]["time_range"] == "13:30 – 16:10"
    assert slot_map["4"]["time_range"] == "16:20 – 19:00"


def test_load_schedule_time_slot_map_prefers_official_cv(monkeypatch, tmp_path):
    cv_path = tmp_path / "Signed.Signed.CV TKB chinh thuc HKII 25-26 gui SV.pdf"
    annex_path = tmp_path / "PHU LUC THOI KHOA BIEU HKII 2025-2026.pdf"
    cv_path.write_text("stub", encoding="utf-8")
    annex_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(server, "_collect_schedule_files", lambda resource_dir: [annex_path, cv_path])
    monkeypatch.setattr(
        server,
        "_SCHEDULE_TIME_SLOT_CACHE",
        {"signature": None, "source_file": None, "slot_map": {}, "checksum": None},
    )

    def fake_process_pdf(path: str):
        if "CV TKB" in path:
            content = "\n".join(
                [
                    "| Buoi | Ca | Tiet | Thoi gian hoc |",
                    "| Sang | 1 | Tiet 1-3 | 07:00 – 09:40 |",
                    "| Sang | 2 | Tiet 4-6 | 09:50 – 12:30 |",
                    "| Chieu | 3 | Tiet 7-9 | 13:30 – 16:10 |",
                    "| Chieu | 4 | Tiet 10-12 | 16:20 – 19:00 |",
                ]
            )
        else:
            content = "| Chieu | 4 | Tiet 10-12 | 16:20 – 19:00 |"
        return [Document(page_content=content, metadata={"file_id": Path(path).name, "index": 1})]

    monkeypatch.setattr(server, "process_pdf", fake_process_pdf)

    slot_map, source_file = server._load_schedule_time_slot_map(force_refresh=True)
    assert len(slot_map) == 4
    assert source_file == cv_path.name


def test_get_schedule_includes_time_slot_map_fields(monkeypatch, tmp_path):
    tkb_path = tmp_path / "PHU LUC THOI KHOA BIEU HKII 2025-2026.pdf"
    tkb_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(server, "_collect_schedule_files", lambda resource_dir: [tkb_path])

    def fake_process_pdf(path: str):
        content = "INT4050 Lop A Ca 4 Thu 3"
        return [Document(page_content=content, metadata={"file_id": Path(path).name, "index": 1})]

    monkeypatch.setattr(server, "process_pdf", fake_process_pdf)
    slot_map = {
        "1": {"session": "Sang", "period": "Tiet 1-3", "time_range": "07:00 – 09:40"},
        "4": {"session": "Chieu", "period": "Tiet 10-12", "time_range": "16:20 – 19:00"},
    }
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: (slot_map, "cv.pdf"))

    payload = json.loads(server.get_schedule(["INT4050"]))
    assert payload[0]["subject_code"] == "INT4050"
    assert payload[0]["time_slot_map"] == slot_map
    assert payload[0]["time_source_file"] == "cv.pdf"
    assert "time_definitions" in payload[0]


def test_check_course_schedule_includes_resolved_time_range(monkeypatch):
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", object())
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda: None)
    monkeypatch.setattr(server.resource_loader, "loaded_resources", {"dummy"}, raising=False)

    tkb_text = "\n".join(
        [
            "INT4050 Lop A Ca 4 Thu 3",
            "PHI1002 Lop B Ca 2 Thu 2",
        ]
    )
    monkeypatch.setattr(server, "_load_best_schedule_text", lambda force_refresh=False: (tkb_text, "appendix.pdf"))
    slot_map = {
        "2": {"session": "Sang", "period": "Tiet 4-6", "time_range": "09:50 – 12:30"},
        "4": {"session": "Chieu", "period": "Tiet 10-12", "time_range": "16:20 – 19:00"},
    }
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: (slot_map, "cv.pdf"))

    result = server.check_course_schedule([{"code": "INT4050"}], target_semester="252")
    assert result[0]["offered"] is True
    assert result[0]["resolved_slot"] == "4"
    assert result[0]["resolved_time_range"] == "16:20 – 19:00"
    assert result[0]["time_slot_map"] == slot_map
    assert result[0]["time_source_file"] == "cv.pdf"


def test_check_course_schedule_strict_e_suffix_not_offered(monkeypatch):
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", object())
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda: None)
    monkeypatch.setattr(server.resource_loader, "loaded_resources", {"dummy"}, raising=False)
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT3404 Lop A Ca 4 Thu 6", "appendix.pdf"),
    )
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: ({}, ""))

    result = server.check_course_schedule([{"code": "INT3404E"}], target_semester="252")
    assert result[0]["code"] == "INT3404E"
    assert result[0]["offered"] is False


def test_check_course_schedule_strict_e_suffix_exact_offered(monkeypatch):
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", object())
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda: None)
    monkeypatch.setattr(server.resource_loader, "loaded_resources", {"dummy"}, raising=False)
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT3412E Lop A Ca 2 Thu 3", "appendix.pdf"),
    )
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: ({}, ""))

    result = server.check_course_schedule([{"code": "INT3412E"}], target_semester="252")
    assert result[0]["code"] == "INT3412E"
    assert result[0]["offered"] is True


def test_get_schedule_strict_e_suffix(monkeypatch, tmp_path):
    tkb_path = tmp_path / "PHU LUC THOI KHOA BIEU HKII 2025-2026.pdf"
    tkb_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(server, "_collect_schedule_files", lambda resource_dir: [tkb_path])

    def fake_process_pdf(path: str):
        content = "INT3404 Lop A Ca 4 Thu 6"
        return [Document(page_content=content, metadata={"file_id": Path(path).name, "index": 1})]

    monkeypatch.setattr(server, "process_pdf", fake_process_pdf)
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: ({}, ""))

    payload = json.loads(server.get_schedule(["INT3404E", "INT3404"]))
    assert payload[0]["subject_code"] == "INT3404E"
    assert payload[0]["note"] == "Not found in TKB."
    assert payload[1]["subject_code"] == "INT3404"
    assert payload[1]["schedule_lines"]


def test_get_electives_with_schedule_strict_e_suffix(monkeypatch):
    curriculum_payload = {
        "total_credits_required": 136,
        "groups": {
            "V.2.4": {
                "group_name": "Nhom tu chon",
                "subjects": [
                    {"code": "INT3404E", "name": "Xu ly anh", "credits": 3},
                ],
                "credits_required": 0,
            }
        },
    }

    monkeypatch.setattr(
        server,
        "get_curriculum_lookup",
        lambda group_hint=None, program_id=None: json.dumps(curriculum_payload, ensure_ascii=False),
    )
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT3404 Lop A Ca 4 Thu 6", "appendix.pdf"),
    )

    result = json.loads(server.get_electives_with_schedule(check_schedule=True, program_id="cs_2022"))
    assert result["opened_count"] == 0
    assert result["not_opened_count"] == 1
    assert result["not_opened"][0]["code"] == "INT3404E"


def test_compute_missing_subjects_strict_e_suffix(monkeypatch):
    transcript_data = {
        "semesters": [
            {
                "semester_code": "251",
                "subjects": [
                    {"code": "INT3404", "name": "Xu ly anh", "credits": 3, "grade_4": 3.0},
                ],
            }
        ],
        "overview": {"total_credits_accumulated": 3},
    }
    curriculum = {
        "subjects": [
            {"code": "INT3404E", "name": "Xu ly anh E", "credits": 3},
        ],
        "structure": [],
        "total_credits": 3,
    }

    missing_info = server.compute_missing_subjects(transcript_data, curriculum)
    missing_codes = {m.get("code") for m in missing_info.get("missing", [])}
    assert "INT3404E" in missing_codes
