import importlib
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def test_upload_pdf_rejects_non_pdf():
    app_mod = importlib.reload(importlib.import_module("app"))
    client = TestClient(app_mod.app)

    resp = client.post("/upload_pdf", files={"file": ("note.txt", BytesIO(b"data"), "text/plain")})
    assert resp.status_code == 400


def test_upload_pdf_saves_and_tracks(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "PDF_DIR", tmp_path / "pdfs")
    app_mod.PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Stub process_pdf to avoid heavy work
    monkeypatch.setattr(app_mod, "process_pdf", lambda path: [])

    client = TestClient(app_mod.app)

    resp = client.post("/upload_pdf", files={"file": ("doc.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    fid = body["file_id"]
    assert fid.endswith(".pdf")
    assert fid in app_mod.loaded_file_ids
    assert app_mod.last_uploaded_file_ids == [fid]
    assert (app_mod.PDF_DIR / fid).exists()


def test_files_endpoint_scopes_results_to_session(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "blob_mode_enabled", lambda: False)
    app_mod.PDF_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.loaded_file_ids = set()
    app_mod.file_meta = {}

    (app_mod.PDF_DIR / "global.pdf").write_bytes(b"%PDF-1.4")
    (app_mod.PDF_DIR / "owned.pdf").write_bytes(b"%PDF-1.4")
    app_mod._save_session_files("session-a", ["owned.pdf"])

    client = TestClient(app_mod.app)
    scoped = client.get("/files?session_id=session-a")
    empty_session = client.get("/files?session_id=session-b")
    legacy = client.get("/files")

    assert scoped.status_code == 200
    assert [item["file_id"] for item in scoped.json()] == ["owned.pdf"]
    assert empty_session.status_code == 200
    assert empty_session.json() == []
    assert legacy.status_code == 200
    assert {item["file_id"] for item in legacy.json()} == {"global.pdf", "owned.pdf"}


def test_authenticated_session_files_do_not_fallback_to_legacy_meta(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "blob_mode_enabled", lambda: False)
    app_mod.PDF_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.loaded_file_ids = set()
    app_mod.file_meta = {}

    (app_mod.PDF_DIR / "legacy.pdf").write_bytes(b"%PDF-1.4")
    (app_mod.PDF_DIR / "owner.pdf").write_bytes(b"%PDF-1.4")
    app_mod._save_session_files("shared-session", ["legacy.pdf"])

    assert app_mod._list_transcript_files("shared-session", user_id="student@example.com") == []

    app_mod._save_session_files("shared-session", ["owner.pdf"], user_id="student@example.com")

    owner_files = app_mod._list_transcript_files("shared-session", user_id="student@example.com")
    legacy_files = app_mod._list_transcript_files("shared-session")

    assert [item["file_id"] for item in owner_files] == ["owner.pdf"]
    assert [item["file_id"] for item in legacy_files] == ["legacy.pdf"]


def test_save_session_files_for_authenticated_user_does_not_inherit_legacy(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "blob_mode_enabled", lambda: False)
    app_mod.PDF_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    app_mod._save_session_files("user_session_1", ["legacy-1.pdf", "legacy-2.pdf"])
    app_mod._save_session_files("user_session_1", ["owned.pdf"], user_id="student@example.com")

    owner_ids = app_mod._load_session_files(
        "user_session_1",
        user_id="student@example.com",
        allow_legacy_fallback=False,
    )
    legacy_ids = app_mod._load_session_files("user_session_1")

    assert owner_ids == ["owned.pdf"]
    assert legacy_ids == ["legacy-1.pdf", "legacy-2.pdf"]


def test_delete_uploaded_file_removes_it_from_session(monkeypatch, tmp_path):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(app_mod, "SESSION_CACHE_DIR", tmp_path / "session_cache")
    monkeypatch.setattr(app_mod, "blob_mode_enabled", lambda: False)
    app_mod.PDF_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app_mod.loaded_file_ids = set()
    app_mod.file_meta = {}

    (app_mod.PDF_DIR / "owned.pdf").write_bytes(b"%PDF-1.4")
    app_mod.file_meta["owned.pdf"] = "Owned transcript.pdf"
    app_mod.loaded_file_ids.add("owned.pdf")
    app_mod._save_session_files("session-a", ["owned.pdf"])

    client = TestClient(app_mod.app)
    resp = client.delete("/files/owned.pdf?session_id=session-a")

    assert resp.status_code == 200
    assert resp.json()["selected_file_ids"] == []
    assert app_mod._list_transcript_files("session-a") == []
    assert not (app_mod.PDF_DIR / "owned.pdf").exists()
