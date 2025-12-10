import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import utils  # noqa: E402
from utils import FAISSVectorStore  # noqa: E402


class DummyEmbedder:
    def __init__(self, name="dummy-model", vectors=None):
        self.model_name = name
        self._vectors = vectors or []
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self._vectors:
            return self._vectors
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _make_docs(pdf_name: str) -> list[Document]:
    return [
        Document(page_content="doc1", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 1}),
        Document(page_content="doc2", metadata={"file_id": pdf_name, "file_name": pdf_name, "index": 2}),
    ]


def test_load_embeddings_cache_miss_and_write(tmp_path, temp_cache, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("hello pdf", encoding="utf-8")

    embedder = DummyEmbedder()
    docs = _make_docs(pdf_path.name)

    emb_np = utils.load_embeddings_with_cache(str(pdf_path), embedder, docs)

    assert embedder.calls == 1
    assert emb_np.shape[0] == len(docs)
    emb_file = temp_cache / f"{pdf_path.name}_embeddings.npy"
    meta_file = temp_cache / f"{pdf_path.name}_embeddings_meta.json"
    assert emb_file.exists()
    assert meta_file.exists()
    meta = meta_file.read_text(encoding="utf-8")
    assert embedder.model_name in meta


def test_load_embeddings_cache_hit_skips_embedding(tmp_path, temp_cache, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("hello pdf", encoding="utf-8")
    docs = _make_docs(pdf_path.name)

    first_embedder = DummyEmbedder(vectors=[[1.0, 0.0], [0.0, 1.0]])
    first = utils.load_embeddings_with_cache(str(pdf_path), first_embedder, docs)
    assert first_embedder.calls == 1

    class NoCallEmbedder:
        model_name = "dummy-model"

        def embed_documents(self, texts):
            raise AssertionError("embed_documents should not be called on cache hit")

    second = utils.load_embeddings_with_cache(str(pdf_path), NoCallEmbedder(), docs)
    assert np.allclose(first, second)


def test_faiss_add_documents_with_embeddings_normalizes():
    dummy = DummyEmbedder()
    store = FAISSVectorStore([], dummy)
    docs = [
        Document(page_content="a", metadata={"file_id": "f", "index": 1}),
        Document(page_content="b", metadata={"file_id": "f", "index": 2}),
    ]
    embeddings = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")

    store.add_documents_with_embeddings(docs, embeddings)

    assert len(store.documents) == 2
    norms = np.linalg.norm(store.embeddings_np, axis=1)
    assert np.allclose(norms, 1.0)


def test_faiss_retrieve_filters_file_ids(monkeypatch):
    embedder = DummyEmbedder()
    docs = [
        Document(page_content="alpha beta", metadata={"file_id": "A", "index": 1}),
        Document(page_content="gamma delta", metadata={"file_id": "B", "index": 2}),
    ]
    store = FAISSVectorStore([], embedder)
    store.add_documents_with_embeddings(docs, np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))

    # Only allow file_id A
    results = store.retrieve("alpha", top_k=5, threshold=0.0, file_ids=["A"])
    assert len(results) == 1
    assert results[0].metadata["file_id"] == "A"


def test_load_embeddings_cache_invalidated_on_model_change(tmp_path, temp_cache):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("hello pdf", encoding="utf-8")
    docs = _make_docs(pdf_path.name)

    embedder_v1 = DummyEmbedder(name="m1", vectors=[[1.0, 0.0], [0.0, 1.0]])
    utils.load_embeddings_with_cache(str(pdf_path), embedder_v1, docs)
    assert embedder_v1.calls == 1

    embedder_v2 = DummyEmbedder(name="m2", vectors=[[1.0, 1.0], [1.0, 2.0]])
    out = utils.load_embeddings_with_cache(str(pdf_path), embedder_v2, docs)
    assert embedder_v2.calls == 1  # cache miss due to different model name
    # Embeddings differ from v1
    assert not np.allclose(out, [[1.0, 0.0], [0.0, 1.0]])


def test_faiss_retrieve_threshold_fallback():
    embedder = DummyEmbedder()
    docs = [
        Document(page_content="foo", metadata={"file_id": "A", "index": 1}),
        Document(page_content="bar", metadata={"file_id": "A", "index": 2}),
    ]
    store = FAISSVectorStore([], embedder)
    store.add_documents_with_embeddings(docs, np.array([[0.01, 0.0], [0.02, 0.0]], dtype="float32"))

    # High threshold yields none; fallback inside retrieve should still return something
    results = store.retrieve("anything", top_k=2, threshold=0.9, file_ids=["A"])
    assert len(results) == 2
