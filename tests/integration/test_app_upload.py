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
