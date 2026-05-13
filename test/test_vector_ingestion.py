import os
import sys

import numpy as np
from langchain_core.documents import Document

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import FAISSVectorStore


class FakeEmbedder:
    def embed_documents(self, texts):
        vectors = []
        for idx, text in enumerate(texts):
            base = float(idx + 1)
            vectors.append([base, float(len(text) % 7 + 1), 1.0])
        return vectors

    def embed_query(self, text):
        return [1.0, float(len(text) % 7 + 1), 1.0]


def test_vector_ingestion(tmp_path):
    docs = [
        Document(
            page_content="Quy che dao tao quy dinh dieu kien tot nghiep va tin chi.",
            metadata={"source": "handbook.pdf", "page": 1},
        ),
        Document(
            page_content="Thoi khoa bieu mon INT3401E hoc ca 2 tai phong 305.",
            metadata={"source": "schedule.pdf", "page": 2},
        ),
    ]

    vector_store = FAISSVectorStore(documents=[], embedder=FakeEmbedder())
    vector_store.add_documents(docs)

    assert len(vector_store.documents) == 2
    assert vector_store.embeddings_np is not None
    assert vector_store.embeddings_np.shape == (2, 3)
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 2
    assert np.allclose(np.linalg.norm(vector_store.embeddings_np, axis=1), 1.0)

    snapshot = tmp_path / "vector_snapshot.pkl"
    assert vector_store.save_snapshot(snapshot, metadata={"test": "vector_ingestion"}) is True
    assert snapshot.exists()
