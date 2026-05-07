from typing import List

from langchain_core.documents import Document

from supabase_support import pgvector_enabled
from utils import FAISSVectorStore
from vector_store_pg import PGVectorStore


def build_vector_store(documents: List[Document], embedder):
    if pgvector_enabled():
        return PGVectorStore(documents, embedder)
    return FAISSVectorStore(documents, embedder)
