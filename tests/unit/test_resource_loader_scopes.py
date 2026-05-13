import importlib
import json
import sys
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def test_get_resources_includes_global_and_session_items(tmp_path, monkeypatch):
    rl_mod = importlib.reload(importlib.import_module("resource_loader"))

    resource_root = tmp_path / "resources"
    pdf_dir = resource_root / "pdfs"
    html_dir = resource_root / "html"
    config_file = resource_root / "config.json"
    session_dir = resource_root / "sessions"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"urls": []}), encoding="utf-8")

    monkeypatch.setattr(rl_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(rl_mod, "HTML_DIR", html_dir)
    monkeypatch.setattr(rl_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(rl_mod, "SESSION_DIR", session_dir)

    (pdf_dir / "global_rules.pdf").write_text("stub", encoding="utf-8")

    loader = rl_mod.ResourceLoader()
    session_pdf_dir, _, session_config = loader._scope_dirs("session-a")
    (session_pdf_dir / "session_mail.pdf").write_text("stub", encoding="utf-8")
    session_config.write_text(json.dumps({"urls": []}), encoding="utf-8")

    resources = loader.get_resources(session_id="session-a")

    names = {(item["name"], item["scope"]) for item in resources}
    ids = {item["id"] for item in resources}

    assert ("global_rules.pdf", "global") in names
    assert ("session_mail.pdf", "session") in names
    assert "global::global_rules.pdf" in ids
    assert "session::session-a::session_mail.pdf" in ids


def test_get_resources_does_not_leak_other_session_items(tmp_path, monkeypatch):
    rl_mod = importlib.reload(importlib.import_module("resource_loader"))

    resource_root = tmp_path / "resources"
    pdf_dir = resource_root / "pdfs"
    html_dir = resource_root / "html"
    config_file = resource_root / "config.json"
    session_dir = resource_root / "sessions"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"urls": []}), encoding="utf-8")

    monkeypatch.setattr(rl_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(rl_mod, "HTML_DIR", html_dir)
    monkeypatch.setattr(rl_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(rl_mod, "SESSION_DIR", session_dir)

    loader = rl_mod.ResourceLoader()
    session_a_pdf_dir, _, session_a_config = loader._scope_dirs("session-a")
    session_b_pdf_dir, _, session_b_config = loader._scope_dirs("session-b")
    (session_a_pdf_dir / "mail_a.pdf").write_text("stub-a", encoding="utf-8")
    (session_b_pdf_dir / "mail_b.pdf").write_text("stub-b", encoding="utf-8")
    session_a_config.write_text(json.dumps({"urls": []}), encoding="utf-8")
    session_b_config.write_text(json.dumps({"urls": []}), encoding="utf-8")

    resources = loader.get_resources(session_id="session-a")
    names = {item["name"] for item in resources}

    assert "mail_a.pdf" in names
    assert "mail_b.pdf" not in names


def test_get_resources_reads_local_scope_without_blob_sync(tmp_path, monkeypatch):
    rl_mod = importlib.reload(importlib.import_module("resource_loader"))

    resource_root = tmp_path / "resources"
    pdf_dir = resource_root / "pdfs"
    html_dir = resource_root / "html"
    config_file = resource_root / "config.json"
    session_dir = resource_root / "sessions"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"urls": [{"url": "https://example.com/global"}]}), encoding="utf-8")

    monkeypatch.setattr(rl_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(rl_mod, "HTML_DIR", html_dir)
    monkeypatch.setattr(rl_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(rl_mod, "SESSION_DIR", session_dir)
    monkeypatch.setattr(rl_mod, "blob_mode_enabled", lambda: True)
    monkeypatch.setattr(
        rl_mod,
        "get_blob_store",
        lambda: (_ for _ in ()).throw(AssertionError("request-time resource reads must not hit blob storage")),
    )

    loader = rl_mod.ResourceLoader()
    monkeypatch.setattr(
        loader,
        "_sync_scope_from_blob",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("scope sync must be explicit, not request-time")),
    )
    session_pdf_dir, _, session_config = loader._scope_dirs("session-a")
    (pdf_dir / "global_rules.pdf").write_text("stub", encoding="utf-8")
    (session_pdf_dir / "session_mail.pdf").write_text("stub", encoding="utf-8")
    session_config.write_text(json.dumps({"urls": [{"url": "https://example.com/session"}]}), encoding="utf-8")

    resources = loader.get_resources(session_id="session-a")
    names = {(item["name"], item["scope"]) for item in resources}

    assert ("global_rules.pdf", "global") in names
    assert ("session_mail.pdf", "session") in names
    assert ("https://example.com/global", "global") in names
    assert ("https://example.com/session", "session") in names


def test_scope_signature_and_resource_ids_change_when_global_resources_change(tmp_path, monkeypatch):
    rl_mod = importlib.reload(importlib.import_module("resource_loader"))

    resource_root = tmp_path / "resources"
    pdf_dir = resource_root / "pdfs"
    html_dir = resource_root / "html"
    config_file = resource_root / "config.json"
    session_dir = resource_root / "sessions"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"urls": [{"url": "https://example.com/a"}]}), encoding="utf-8")

    monkeypatch.setattr(rl_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(rl_mod, "HTML_DIR", html_dir)
    monkeypatch.setattr(rl_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(rl_mod, "SESSION_DIR", session_dir)

    loader = rl_mod.ResourceLoader()

    (pdf_dir / "global_rules.pdf").write_text("stub", encoding="utf-8")
    sig_before = loader.get_scope_signature()
    ids_before = loader.list_scope_resource_ids()
    assert "global_rules.pdf" in ids_before
    assert any(item.startswith("url_") for item in ids_before)

    (html_dir / "ctdt.html").write_text("<html><body>ctdt</body></html>", encoding="utf-8")
    sig_after = loader.get_scope_signature()
    ids_after = loader.list_scope_resource_ids()

    assert sig_after != sig_before
    assert "ctdt.html" in ids_after


def test_load_resources_uses_cached_embeddings_for_html_and_rebuilds_once(tmp_path, monkeypatch):
    rl_mod = importlib.reload(importlib.import_module("resource_loader"))

    resource_root = tmp_path / "resources"
    pdf_dir = resource_root / "pdfs"
    html_dir = resource_root / "html"
    config_file = resource_root / "config.json"
    session_dir = resource_root / "sessions"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"urls": []}), encoding="utf-8")

    monkeypatch.setattr(rl_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(rl_mod, "HTML_DIR", html_dir)
    monkeypatch.setattr(rl_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(rl_mod, "SESSION_DIR", session_dir)

    pdf_file = pdf_dir / "global_rules.pdf"
    html_file = html_dir / "ctdt.html"
    pdf_file.write_text("pdf", encoding="utf-8")
    html_file.write_text("<html><body>ctdt</body></html>", encoding="utf-8")

    monkeypatch.setattr(
        rl_mod,
        "process_pdf",
        lambda _: [Document(page_content="pdf chunk", metadata={})],
    )
    monkeypatch.setattr(
        rl_mod,
        "crawl_url",
        lambda _: [Document(page_content="html chunk", metadata={})],
    )

    emb_calls: list[str] = []

    def _fake_embeddings(path, _embedder, docs):
        emb_calls.append(Path(path).name)
        return np.array([[1.0, 0.0]] * len(docs), dtype="float32")

    monkeypatch.setattr(rl_mod, "load_embeddings_with_cache", _fake_embeddings)

    class DummyStore:
        def __init__(self):
            self.embedder = object()
            self.add_with_emb_calls = []
            self.rebuild_calls = 0

        def add_documents_with_embeddings(self, docs, embeddings, rebuild_index=True):
            self.add_with_emb_calls.append(
                {
                    "docs": len(docs),
                    "shape": tuple(embeddings.shape),
                    "rebuild_index": rebuild_index,
                }
            )

        def add_documents(self, docs, rebuild_index=True):
            raise AssertionError("URL path is not expected in this test.")

        def rebuild_index(self):
            self.rebuild_calls += 1

    store = DummyStore()
    loader = rl_mod.ResourceLoader(store)
    loader.load_resources()

    assert sorted(emb_calls) == sorted([pdf_file.name, html_file.name])
    assert len(store.add_with_emb_calls) == 2
    assert all(call["rebuild_index"] is False for call in store.add_with_emb_calls)
    assert store.rebuild_calls == 1
