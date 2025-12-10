import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import utils  # noqa: E402
from utils import FAISSVectorStore  # noqa: E402


class DummyEmbedder:
    def __init__(self):
        self.model_name = "dummy"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_web_search_fallback_on_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(utils.requests, "get", boom)
    res = utils.web_search("anything", num_results=3)
    assert res == ["(Khong co ket qua web)"]


def test_faiss_empty_store_returns_empty():
    store = FAISSVectorStore([], DummyEmbedder())
    res = store.retrieve("q", top_k=3)
    assert res == []


def test_faiss_dimension_mismatch_raises():
    store = FAISSVectorStore([], DummyEmbedder())
    docs = [Document(page_content="x", metadata={"file_id": "f", "index": 1})]
    store.add_documents_with_embeddings(docs, np.array([[1.0, 0.0]], dtype="float32"))

    with pytest.raises(ValueError):
        store.add_documents_with_embeddings(docs, np.array([[1.0, 2.0, 3.0]], dtype="float32"))
