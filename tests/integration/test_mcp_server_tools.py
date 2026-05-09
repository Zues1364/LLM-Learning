import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402
import mcp_server.structured_schedule_store as schedule_store  # noqa: E402
from persistent_memory import PersistentMemory  # noqa: E402


def test_mcp_health_endpoint():
    client = TestClient(server.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mcp_ready_endpoint_checks_dependencies(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setattr(server, "check_postgres_ready", lambda: "disabled")
    monkeypatch.setattr(server, "check_blob_ready", lambda store: "disabled")
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["postgres"] == "disabled"
    assert payload["checks"]["blob_store"] == "disabled"
    assert payload["checks"]["vector_store"] == "lazy"


def test_mcp_startup_defaults_to_lazy_vector_init(monkeypatch):
    monkeypatch.delenv("MCP_STARTUP_VECTOR_INIT", raising=False)
    monkeypatch.setattr(
        server,
        "_init_vector_store",
        lambda: (_ for _ in ()).throw(AssertionError("startup should not block on vector init")),
    )

    server.startup_event()


def test_mcp_startup_can_eager_init_vector_store(monkeypatch):
    calls = []
    monkeypatch.setenv("MCP_STARTUP_VECTOR_INIT", "eager")
    monkeypatch.setattr(server, "_init_vector_store", lambda: calls.append("init"))

    server.startup_event()

    assert calls == ["init"]


def _make_docs(pdf_name: str) -> list[Document]:
    return [
        Document(page_content="alpha", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 1, "page": 3}),
        Document(page_content="beta", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 2, "page": 4}),
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
    monkeypatch.setattr(
        server,
        "build_vector_store",
        lambda documents, embedder: server.FAISSVectorStore(documents, embedder),
    )

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
    monkeypatch.setattr(
        server,
        "build_vector_store",
        lambda documents, embedder: server.FAISSVectorStore(documents, embedder),
    )
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda session_id=None, user_id=None: None)

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
    assert all("Page" in ctx for ctx in contexts)
    assert any("alpha" in ctx for ctx in contexts)


def test_retrieve_chunks_with_explicit_file_ids_does_not_merge_scoped_resources(monkeypatch):
    captured: dict[str, list[str]] = {}

    class DummyStore:
        def retrieve(self, query, top_k=25, file_ids=None):
            captured["file_ids"] = list(file_ids or [])
            return []

    monkeypatch.setattr(server, "_build_teacher_lookup_context", lambda **kwargs: [])
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", DummyStore())
    monkeypatch.setattr(server, "_ensure_file_loaded", lambda fid: fid)
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda session_id=None, user_id=None: None)
    monkeypatch.setattr(
        server.resource_loader,
        "get_loaded_resource_ids",
        lambda session_id=None, user_id=None, include_global=True: {"global_ctdt.html", "session_notes.pdf"},
    )

    server.retrieve_chunks(
        "toi con thieu nhung mon nao",
        top_k=5,
        file_ids=["bang_diem1.pdf"],
        session_id="session_1",
    )

    assert captured.get("file_ids") == ["bang_diem1.pdf"]


def test_retrieve_chunks_uses_current_scope_resource_ids(monkeypatch):
    captured: dict[str, list[str]] = {}

    class DummyStore:
        def retrieve(self, query, top_k=25, file_ids=None):
            captured["file_ids"] = list(file_ids or [])
            return []

    monkeypatch.setattr(server, "_build_teacher_lookup_context", lambda **kwargs: [])
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", DummyStore())
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda session_id=None, user_id=None: None)
    monkeypatch.setattr(
        server.resource_loader,
        "get_loaded_resource_ids",
        lambda session_id=None, user_id=None, include_global=True: (_ for _ in ()).throw(
            AssertionError("retrieve_chunks must not rely on loaded_resource_ids")
        ),
    )

    def fake_list_scope_resource_ids(session_id=None, user_id=None):
        if session_id or user_id:
            return {"session_live.pdf"}
        return {"global_live.pdf"}

    monkeypatch.setattr(server.resource_loader, "list_scope_resource_ids", fake_list_scope_resource_ids)

    server.retrieve_chunks(
        "toi can lich hoc",
        top_k=5,
        file_ids=[],
        session_id="session_1",
    )

    assert set(captured.get("file_ids") or []) == {"global_live.pdf", "session_live.pdf"}


def test_retrieve_chunks_teacher_lookup_shortcuts_before_vector_store(monkeypatch):
    monkeypatch.setattr(
        server,
        "_build_teacher_lookup_context",
        lambda question, top_k=25, session_id=None, user_id=None: ["[SCHEDULE PEC1008] K69I-CS1 PEC1008 ... NgÃ´ ThÃ¡i HÃ "],
    )
    monkeypatch.setattr(server, "_init_vector_store", lambda: (_ for _ in ()).throw(AssertionError("should not init vector store")))

    res = server.retrieve_chunks("mÃ´n kinh táº¿ chÃ­nh trá»‹ kÃ¬ nÃ y cÃ³ nhá»¯ng ai dáº¡y", top_k=25, file_ids=[], session_id="s1")
    assert res == ["[SCHEDULE PEC1008] K69I-CS1 PEC1008 ... NgÃ´ ThÃ¡i HÃ "]



def test_build_teacher_lookup_context_infers_code_and_expands_schedule(monkeypatch):
    sample_schedule_text = (
        "K69I-CS1 PEC1008 Kinh te chinh tri Mac - Lenin ...\n"
        "K69I-CS2 PEC1008 Kinh te chinh tri Mac - Lenin ...\n"
    )
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False, session_id=None, user_id=None: (sample_schedule_text, "PHU_LUC_TKB.pdf"),
    )
    monkeypatch.setattr(
        server,
        "get_schedule",
        lambda subject_codes, session_id=None, user_id=None: (
            '[{"subject_code":"PEC1008","schedule_lines":['
            '"K69I-CS1 PEC1008 ... Ng\\u00f4 Th\\u00e1i H\\u00e0 H\\u1ecdc 1 ca/10 tuan",'
            '"K69I-CS2 PEC1008 ... Nguy\\u1ec5n Th\\u1ecb H\\u1ed3ng S\\u00e2m H\\u1ecdc 1 ca/10 tuan"'
            '],"note":""}]'
        ),
    )

    res = server._build_teacher_lookup_context(
        question="mon kinh te chinh tri ki nay co nhung ai day",
        top_k=25,
        session_id="s1",
        user_id=None,
    )
    joined = "\n".join(res)
    assert len(res) >= 2
    assert "PEC1008" in joined
    assert "Ng\u00f4 Th\u00e1i H\u00e0" in joined
    assert "Nguy\u1ec5n Th\u1ecb H\u1ed3ng S\u00e2m" in joined

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
    assert advisor_context["schedule_table_columns"][1] == "Ca h\u1ecdc"
    assert advisor_context["schedule_table_columns"][2] == "Ti\u1ebft + Th\u1eddi gian"


def test_consult_advisor_does_not_over_suggest_electives_when_only_mandatory_missing(monkeypatch):
    captured = {}

    class DummyMemory:
        def get_context(self, query, session_id="default", max_rows=10):
            return ""

    monkeypatch.setattr(server, "_memory", DummyMemory())
    monkeypatch.setattr(server, "_load_session_file_ids", lambda session_id: ["mock_transcript.pdf"])
    monkeypatch.setattr(
        server,
        "_load_schedule_time_slot_map",
        lambda force_refresh=False, session_id=None, user_id=None: ({}, ""),
    )
    monkeypatch.setattr(
        server,
        "analyze_transcript",
        lambda ids: json.dumps(
            {
                "student_info": {"class": "K67-CS", "major": "Khoa hoc may tinh"},
                "semesters": [
                    {"semester_code": "251", "subjects": [{"code": "INT1001", "name": "Nhap mon", "credits": 3, "grade_4": 3.0}]}
                ],
                "completed_subjects": [{"code": "INT1001", "name": "Nhap mon", "credits": 3, "grade_4": 3.0}],
                "overview": {"total_credits_accumulated": 129},
            }
        ),
    )
    monkeypatch.setattr(
        server,
        "compute_missing_subjects",
        lambda transcript_data, curriculum: {
            "missing": [{"code": "INT4050", "name": "Khoa luan", "credits": 7}],
            "completed_map": {},
            "low_grades": [],
            "credit_summary": {
                "transcript_total_credits": 129,
                "total_required_credits": 136,
                "total_completed_applicable_credits": 129,
                "total_missing_credits": 7,
                "external_credits_applied": [],
            },
            "credit_analysis": [
                {
                    "block_id": "V.1",
                    "block_name": "Khoi kien thuc nganh - bat buoc",
                    "block_type": "required",
                    "required_credits": 7,
                    "completed_credits": 0,
                    "missing_credits": 7,
                    "candidates": [{"code": "INT4050", "name": "Khoa luan", "credits": 7}],
                },
                {
                    "block_id": "V.2",
                    "block_name": "Khoi kien thuc nganh - hoc phan tu chon",
                    "block_type": "elective",
                    "required_credits": 10,
                    "completed_credits": 10,
                    "missing_credits": 0,
                    "candidates": [{"code": "INT3306", "name": "Web", "credits": 3}],
                },
            ],
        },
    )
    monkeypatch.setattr(
        server,
        "check_course_schedule",
        lambda subjects, target_semester=None, class_code=None, session_id=None, user_id=None: [
            {
                "code": "INT4050",
                "offered": True,
                "snippet": "INT4050 1",
                "resolved_day": "Th\u1ee9 5",
                "resolved_slot": "2",
                "resolved_time_range": "09:50 – 12:30",
                "time_slot_map": {"2": {"period": "Tiáº¿t 4-6", "time_range": "09:50 – 12:30"}},
                "schedule_source_file": "PHU_LUC_TKB.pdf",
            }
        ],
    )
    monkeypatch.setattr(
        server,
        "analyze_curriculum",
        lambda program_hint=None: {
            "program_name": "cs_2022",
            "subjects": [],
            "structure": [],
            "total_credits": 136,
            "source_path": None,
            "notes": "stub",
        },
    )
    monkeypatch.setattr(
        server,
        "get_electives_with_schedule",
        lambda check_schedule=True, program_id=None, session_id=None, user_id=None: json.dumps(
            {
                "opened": [
                    {"code": "INT3306", "name": "Web", "credits": 3, "group": "V.2.1"},
                    {"code": "INT3323", "name": "IoT", "credits": 3, "group": "V.2.1"},
                ]
            }
        ),
    )

    class DummyAdvisorAgent:
        def run(self, prompt):
            captured["advisor_prompt"] = prompt
            return type("Resp", (), {"content": "advisor-ok"})()

    monkeypatch.setattr(server, "get_academic_advisor_agent", lambda: DummyAdvisorAgent())

    result = server.consult_advisor(
        query="theo chuong trinh dao tao toi con thieu bao tin chi va nhung mon gi",
        file_ids=[],
        session_id="s200",
        program_id="cs_2022",
    )
    assert result == "advisor-ok"

    context_blob = captured["advisor_prompt"].split("--- CONTEXT ---\n", 1)[1].split("\n--- END ---", 1)[0]
    advisor_context = json.loads(context_blob)
    assert advisor_context["missing_subjects"]["recommended"] == [{"code": "INT4050", "name": "Khoa luan", "credits": 7}]
    assert advisor_context["missing_subjects"]["elective_suggestions"] == []


def test_consult_advisor_retries_broad_schedule_for_unresolved_rows(monkeypatch):
    captured = {}
    call_log = []

    class DummyMemory:
        def get_context(self, query, session_id="default", max_rows=10):
            return ""

    monkeypatch.setattr(server, "_memory", DummyMemory())
    monkeypatch.setattr(server, "_load_session_file_ids", lambda session_id: ["mock_transcript.pdf"])
    monkeypatch.setattr(
        server,
        "_load_schedule_time_slot_map",
        lambda force_refresh=False, session_id=None, user_id=None: (
            {"2": {"period": "Tiáº¿t 4-6", "time_range": "09:50 – 12:30"}},
            "TKB_CV.pdf",
        ),
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
                        "subjects": [{"code": "INT2000", "name": "Mon mau", "credits": 3, "grade_4": 2.5}],
                    }
                ],
                "completed_subjects": [{"code": "INT2000", "name": "Mon mau", "credits": 3}],
                "overview": {"total_credits_accumulated": 129},
            }
        ),
    )
    monkeypatch.setattr(
        server,
        "compute_missing_subjects",
        lambda transcript_data, curriculum: {
            "missing": [{"code": "INT4050", "name": "Khoa luan", "credits": 7}],
            "completed_map": {},
            "low_grades": [],
            "credit_summary": {
                "transcript_total_credits": 129,
                "total_required_credits": 136,
                "total_completed_applicable_credits": 129,
                "total_missing_credits": 7,
                "external_credits_applied": [],
            },
            "credit_analysis": [
                {
                    "block_id": "V.1",
                    "block_name": "Khoi kien thuc nganh - bat buoc",
                    "block_type": "required",
                    "required_credits": 7,
                    "completed_credits": 0,
                    "missing_credits": 7,
                    "candidates": [{"code": "INT4050", "name": "Khoa luan", "credits": 7}],
                }
            ],
        },
    )

    def fake_check_course_schedule(subjects, target_semester=None, class_code=None, session_id=None, user_id=None):
        call_log.append({"target_semester": target_semester, "class_code": class_code, "codes": [s.get("code") for s in subjects]})
        if target_semester is None and class_code is None:
            return [
                {
                    "code": "INT4050",
                    "offered": True,
                    "snippet": "INT4050 1",
                    "resolved_day": "Th\u1ee9 5",
                    "resolved_slot": "2",
                    "resolved_time_range": "09:50 – 12:30",
                    "time_slot_map": {"2": {"period": "Tiáº¿t 4-6", "time_range": "09:50 – 12:30"}},
                    "schedule_source_file": "PHU_LUC_TKB.pdf",
                }
            ]
        return [
            {
                "code": "INT4050",
                "offered": True,
                "snippet": "INT4050 1",
                "resolved_day": None,
                "resolved_slot": None,
                "resolved_time_range": None,
                "time_slot_map": {"2": {"period": "Tiáº¿t 4-6", "time_range": "09:50 – 12:30"}},
                "schedule_source_file": "PHU_LUC_TKB.pdf",
            }
        ]

    monkeypatch.setattr(server, "check_course_schedule", fake_check_course_schedule)
    monkeypatch.setattr(
        server,
        "analyze_curriculum",
        lambda program_hint=None: {
            "program_name": "cs_2022",
            "subjects": [],
            "structure": [],
            "total_credits": 136,
            "source_path": None,
            "notes": "stub",
        },
    )
    monkeypatch.setattr(
        server,
        "get_electives_with_schedule",
        lambda check_schedule=True, program_id=None, session_id=None, user_id=None: json.dumps({"opened": []}),
    )

    class DummyAdvisorAgent:
        def run(self, prompt):
            captured["advisor_prompt"] = prompt
            return type("Resp", (), {"content": "advisor-ok"})()

    monkeypatch.setattr(server, "get_academic_advisor_agent", lambda: DummyAdvisorAgent())

    result = server.consult_advisor(
        query="toi con thieu bao tin chi va mon nao",
        file_ids=[],
        session_id="s201",
        program_id="cs_2022",
    )
    assert result == "advisor-ok"

    assert any(c["target_semester"] is not None for c in call_log)
    assert any(c["target_semester"] is None and c["class_code"] is None for c in call_log)

    context_blob = captured["advisor_prompt"].split("--- CONTEXT ---\n", 1)[1].split("\n--- END ---", 1)[0]
    advisor_context = json.loads(context_blob)
    rows = advisor_context.get("schedule_table_rows") or []
    int4050 = next((r for r in rows if r.get("subject_code") == "INT4050"), None)
    assert int4050 is not None
    assert int4050.get("day") == "Th\u1ee9 5"
    assert int4050.get("ca_hoc") == "Ca 2"


def test_query_needs_schedule_context_for_gpa_and_missing_credits():
    assert server._query_needs_schedule_context("với tình trạng điểm gpa của tôi có khả năng lên bằng giỏi không")
    assert server._query_needs_schedule_context("tôi còn thiếu bao nhiêu tín chỉ")


def test_sync_schedule_scope_from_blob_downloads_only_schedule_pdfs(monkeypatch, tmp_path):
    class Blob:
        def __init__(self, key, size=10):
            self.key = key
            self.size = size

    class DummyStore:
        def __init__(self):
            self.downloaded = []

        def list_objects(self, prefix):
            if prefix == "resources/global/pdf/":
                return [
                    Blob("resources/global/pdf/PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf"),
                    Blob("resources/global/pdf/SO_TAY_HOC_VU.pdf"),
                    Blob("resources/global/html/CTDT.html"),
                ]
            if prefix == "resources/s_schedule/pdf/":
                return [Blob("resources/s_schedule/pdf/Signed_CV_TKB.pdf")]
            return []

        def download_to_path(self, key, target_path):
            self.downloaded.append(key)
            target_path.write_bytes(b"schedule")

    store = DummyStore()
    monkeypatch.setattr(server, "blob_mode_enabled", lambda: True)
    monkeypatch.setattr(server, "get_blob_store", lambda: store)
    monkeypatch.setattr(server, "local_path_from_key", lambda key: tmp_path / key.replace("/", "_"))

    server._sync_schedule_scope_from_blob(session_id="s_schedule", user_id=None)

    assert store.downloaded == [
        "resources/global/pdf/PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf",
        "resources/s_schedule/pdf/Signed_CV_TKB.pdf",
    ]


def test_structured_schedule_pdf_parser_prefers_text_only(monkeypatch, tmp_path):
    pdf_path = tmp_path / "PHU_LUC_THOI_KHOA_BIEU.xlsx_-_Sheet1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% schedule fixture")
    doc = Document(
        page_content=(
            "K68I-CS4 | INT2041 | Tương tác người-máy | 3 | 45 |  |  | 1 | 74 | "
            "INT2041 2 | CL | LT | 5 | 2 | 3-G3 | Ngô Thị Duyên |  | "
            "Học 1 ca/15 tuần, thi đợt 2"
        ),
        metadata={"page": 4},
    )
    monkeypatch.setattr(schedule_store, "process_pdf_text_only", lambda path: [doc])
    monkeypatch.setattr(
        schedule_store,
        "process_pdf",
        lambda path: (_ for _ in ()).throw(AssertionError("schedule parser should not invoke OCR path")),
    )

    store = schedule_store.StructuredScheduleStore(db_path=tmp_path / "structured_schedule.db")
    rows = store._parse_schedule_pdf(pdf_path)

    int2041 = next((row for row in rows if row.get("subject_code") == "INT2041"), None)
    assert int2041 is not None
    assert int2041["class_code"] == "INT2041 2"
    assert int2041["day_of_week"] == "Thứ 5"
    assert int2041["slot"] == "2"
    assert int2041["room"] == "3-G3"


def test_render_gpa_feasibility_text_includes_schedule_rows():
    advisor_context = {
        "credit_summary": {
            "transcript_total_credits": 129,
            "curriculum_applicable_credits": 129,
            "total_missing_credits": 7,
        },
        "missing_subjects": {
            "mandatory_missing": [{"code": "INT4050", "name": "Khoa luan tot nghiep", "credits": 7}],
        },
        "gpa_projection": {
            "target_gpa": 3.2,
            "current_gpa": 2.897,
            "max_gpa_no_retakes": 2.9521,
            "max_possible_gpa": 3.4807,
            "feasible_no_retakes": False,
            "feasible_with_retakes": True,
        },
        "schedule_source_file": "PHU LUC TKB HKII 2025-2026.pdf",
        "schedule_table_rows": [
            {
                "day": "Thứ 5",
                "ca_hoc": "Ca 2",
                "period_time": "Tiết 4-6 (09:50 – 12:30)",
                "subject_code": "INT4050",
                "subject_name": "Khóa luận tốt nghiệp",
                "credits": 7,
                "class_note": "Lớp INT4050 1",
            }
        ],
    }

    answer = server._render_gpa_feasibility_text(
        "với tình trạng điểm gpa của tôi có khả năng lên bằng giỏi không",
        advisor_context,
    )

    assert "Gợi ý lịch" in answer
    assert "INT4050" in answer
    assert "Thứ 5" in answer
    assert "PHU LUC TKB HKII 2025-2026.pdf" in answer



def test_extract_time_slot_map_parses_standard_table():
    text = """
| Buoi | Ca | Tiet | Thoi gian hoc | Ghi chu |
| --- | --- | --- | --- | --- |
| Sang | 1 | Tiet 1-3 | 07:00 - 09:40 | break |
| Sang | 2 | Tiet 4-6 | 09:50 - 12:30 | break |
| Chieu | 3 | Tiet 7-9 | 13:30 - 16:10 | break |
| Chieu | 4 | Tiet 10-12 | 16:20 - 19:00 | break |
"""
    slot_map = server._extract_time_slot_map(text)
    assert slot_map["1"]["time_range"] == "07:00 \u2013 09:40"
    assert slot_map["2"]["time_range"] == "09:50 \u2013 12:30"
    assert slot_map["3"]["time_range"] == "13:30 \u2013 16:10"
    assert slot_map["4"]["time_range"] == "16:20 \u2013 19:00"


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
                    "| Sang | 1 | Tiet 1-3 | 07:00 - 09:40 |",
                    "| Sang | 2 | Tiet 4-6 | 09:50 - 12:30 |",
                    "| Chieu | 3 | Tiet 7-9 | 13:30 - 16:10 |",
                    "| Chieu | 4 | Tiet 10-12 | 16:20 - 19:00 |",
                ]
            )
        else:
            content = "| Chieu | 4 | Tiet 10-12 | 16:20 - 19:00 |"
        return [Document(page_content=content, metadata={"file_id": Path(path).name, "index": 1})]

    monkeypatch.setattr(server, "process_pdf", fake_process_pdf)

    slot_map, source_file = server._load_schedule_time_slot_map(force_refresh=True)
    assert len(slot_map) == 4
    assert source_file == cv_path.name


def test_load_schedule_time_slot_map_uses_default_when_extraction_empty(monkeypatch, tmp_path):
    annex_path = tmp_path / "PHU LUC THOI KHOA BIEU HKII 2025-2026.pdf"
    annex_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(server, "_collect_schedule_files", lambda resource_dir: [annex_path])
    monkeypatch.setattr(server, "_SCHEDULE_TIME_SLOT_CACHE", {})
    monkeypatch.setattr(
        server,
        "process_pdf",
        lambda path: [Document(page_content="INT2041 Lop A Ca 2 Thu 5", metadata={"file_id": Path(path).name, "index": 1})],
    )

    slot_map, source_file = server._load_schedule_time_slot_map(force_refresh=True)
    assert source_file == "DEFAULT_UET_TIME_SLOTS"
    assert slot_map["1"]["time_range"] == "07:00 \u2013 09:40"
    assert slot_map["2"]["time_range"] == "09:50 \u2013 12:30"
    assert slot_map["4"]["period"] == "Tiet 10-12"

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


def test_check_course_schedule_resolves_slot_even_when_time_map_empty(monkeypatch):
    monkeypatch.setattr(server, "_init_vector_store", lambda: None)
    monkeypatch.setattr(server, "_store", object())
    monkeypatch.setattr(server.resource_loader, "set_vector_store", lambda store: None)
    monkeypatch.setattr(server.resource_loader, "load_resources", lambda: None)
    monkeypatch.setattr(server.resource_loader, "loaded_resources", {"dummy"}, raising=False)
    monkeypatch.setattr(
        server,
        "_load_best_schedule_text",
        lambda force_refresh=False: ("INT2041 2 LT 5 1 206-T", "appendix.pdf"),
    )
    monkeypatch.setattr(server, "_load_schedule_time_slot_map", lambda force_refresh=False: ({}, ""))

    result = server.check_course_schedule([{"code": "INT2041"}], target_semester="252")
    assert result[0]["offered"] is True
    assert result[0]["resolved_day"] == "Th\u1ee9 5"
    assert result[0]["resolved_slot"] == "1"
    assert result[0]["resolved_time_range"] is None


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


def test_memory_state_tools_roundtrip(tmp_path, monkeypatch):
    mem = PersistentMemory(db_path=str(tmp_path / "memory.db"), max_history=5)
    monkeypatch.setattr(server, "_memory", mem)

    state = server.memory_state_get("s-state")
    assert state["turn_index"] == 0

    updated = server.memory_state_upsert(
        "s-state",
        {
            "turn_index": 7,
            "entities": {"course_codes": ["INT2041"]},
            "referents": {"last_subject_codes": ["INT2041"]},
        },
    )
    assert updated["turn_index"] == 7
    assert "INT2041" in updated["entities"]["course_codes"]

    cleared = server.memory_state_clear("s-state")
    assert cleared == "ok"
    reset = server.memory_state_get("s-state")
    assert reset["turn_index"] == 0


def test_resolve_course_alias_tool_returns_structured_payload(monkeypatch):
    class DummyStore:
        def resolve_course_alias(self, query):
            return {
                "matched_subject": {"subject_code": "HIS1001", "subject_name_vi": "Lá»‹ch sá»­ Äáº£ng"},
                "confidence": 0.95,
                "candidates": [{"subject_code": "HIS1001", "score": 0.95}],
            }

    monkeypatch.setattr(server, "_ensure_structured_schedule_ingested", lambda session_id=None, user_id=None, force=False: {})
    monkeypatch.setattr(server, "_get_structured_schedule_store", lambda: DummyStore())
    monkeypatch.setattr(server, "_get_program_subject_codes", lambda program_id=None, session_id=None: {"HIS1001"})

    payload = json.loads(server.resolve_course_alias("lá»‹ch sá»­ Ä‘áº£ng", program_id="cs_2022", session_id="s1"))
    assert payload["matched_subject"]["subject_code"] == "HIS1001"
    assert payload["confidence"] >= 0.9
    assert payload["program_id"] == "cs_2022"


def test_resolve_course_alias_tool_filters_candidates_by_program_subject_codes(monkeypatch):
    class DummyStore:
        def resolve_course_alias(self, query):
            return {
                "matched_subject": {"subject_code": "CTE2059", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh"},
                "confidence": 0.95,
                "candidates": [
                    {"subject_code": "CTE2059", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh", "score": 0.95},
                    {"subject_code": "INT3403", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh", "score": 0.95},
                ],
            }

    monkeypatch.setattr(server, "_ensure_structured_schedule_ingested", lambda session_id=None, user_id=None, force=False: {})
    monkeypatch.setattr(server, "_get_structured_schedule_store", lambda: DummyStore())
    monkeypatch.setattr(server, "_get_program_subject_codes", lambda program_id=None, session_id=None: {"INT3403"})

    payload = json.loads(server.resolve_course_alias("Ä‘á»“ há»a mÃ¡y tÃ­nh", program_id="cs_2022", session_id="s1"))
    assert payload["matched_subject"]["subject_code"] == "INT3403"
    assert payload["candidates"] == [{"subject_code": "INT3403", "subject_name_vi": "Äá»“ há»a mÃ¡y tÃ­nh", "score": 0.95}]


def test_get_teachers_by_subject_tool_uses_alias_resolution(monkeypatch):
    class DummyStore:
        def resolve_course_alias(self, query):
            return {
                "matched_subject": {"subject_code": "PEC1008", "subject_name_vi": "Kinh táº¿ chÃ­nh trá»‹"},
                "confidence": 0.88,
            }

        def get_teachers_by_subject(self, subject_code, semester=None):
            return {
                "matched_subject": {"subject_code": subject_code, "subject_name_vi": "Kinh táº¿ chÃ­nh trá»‹"},
                "confidence": 1.0,
                "teachers": ["NgÃ´ ThÃ¡i HÃ "],
                "rows": [{"subject_code": subject_code, "class_code": "PEC1008 1"}],
                "source_files": ["PHU_LUC_TKB.pdf"],
                "coverage_note": "ok",
            }

    monkeypatch.setattr(server, "_ensure_structured_schedule_ingested", lambda session_id=None, user_id=None, force=False: {})
    monkeypatch.setattr(server, "_get_structured_schedule_store", lambda: DummyStore())

    payload = json.loads(server.get_teachers_by_subject("kinh táº¿ chÃ­nh trá»‹", session_id="s1"))
    assert payload["matched_subject"]["subject_code"] == "PEC1008"
    assert payload["teachers"] == ["NgÃ´ ThÃ¡i HÃ "]
    assert payload["rows"]


def test_get_classes_by_teacher_tool_returns_rows(monkeypatch):
    class DummyStore:
        def get_classes_by_teacher(self, teacher_name, semester=None):
            return {
                "matched_teacher": {"query": teacher_name, "canonical_names": [teacher_name]},
                "confidence": 0.9,
                "rows": [{"subject_code": "HIS1001", "class_code": "HIS1001 5"}],
                "source_files": ["PHU_LUC_TKB.pdf"],
                "coverage_note": "ok",
            }

    monkeypatch.setattr(server, "_ensure_structured_schedule_ingested", lambda session_id=None, user_id=None, force=False: {})
    monkeypatch.setattr(server, "_get_structured_schedule_store", lambda: DummyStore())

    payload = json.loads(server.get_classes_by_teacher("VÅ© Thá»‹ Thu HÃ ", session_id="s1"))
    assert payload["rows"][0]["subject_code"] == "HIS1001"
