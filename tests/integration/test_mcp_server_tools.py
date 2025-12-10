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
    assert len(contexts) == 1
    assert "Chunk" in contexts[0]
    assert "alpha" in contexts[0]
