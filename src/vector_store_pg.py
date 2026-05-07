import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from supabase_support import get_pg_connection
from utils import normalize_for_match, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def _vector_literal(values: np.ndarray) -> str:
    # pgvector textual format: [1,2,3]
    return "[" + ",".join(f"{float(v):.8f}" for v in values.tolist()) + "]"


class PGVectorStore:
    """
    pgvector-backed vector store with FAISS-like interface compatibility.
    """

    def __init__(self, documents: List[Document], embedder):
        self.embedder = embedder
        self.embedding_dim = int(getattr(embedder, "embedding_dim", 1024) or 1024)
        self.documents: List[Document] = []
        self._init_db()
        if documents:
            self.add_documents(documents)

    def _conn(self):
        return get_pg_connection()

    def _init_db(self):
        ddl = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS vector_documents (
                id BIGSERIAL PRIMARY KEY,
                file_id TEXT,
                file_name TEXT,
                source TEXT,
                page INTEGER,
                chunk_index INTEGER,
                source_line INTEGER,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                page_content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                document_id BIGINT PRIMARY KEY REFERENCES vector_documents(id) ON DELETE CASCADE,
                embedding VECTOR({self.embedding_dim}) NOT NULL,
                normalized BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_vector_documents_file_id ON vector_documents(file_id)",
            "CREATE INDEX IF NOT EXISTS idx_vector_documents_file_name ON vector_documents(file_name)",
            "CREATE INDEX IF NOT EXISTS idx_vector_documents_page ON vector_documents(page)",
            """
            CREATE INDEX IF NOT EXISTS idx_vector_embeddings_ann
            ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """,
        ]
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for statement in ddl:
                    cur.execute(statement)
            logger.info("PGVector store initialized.")
        finally:
            conn.close()

    def _normalize_embeddings(self, emb_np: np.ndarray) -> np.ndarray:
        if emb_np.ndim == 1:
            emb_np = np.expand_dims(emb_np, axis=0)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb_np / norms

    def rebuild_index(self):
        # No-op for pgvector; index is persisted in DB.
        return

    def _upsert_documents_with_vectors(self, docs: List[Document], embeddings: np.ndarray):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for doc, emb in zip(docs, embeddings):
                    metadata = dict(doc.metadata or {})
                    if "_norm_text" not in metadata:
                        metadata["_norm_text"] = normalize_for_match(doc.page_content)
                    file_id = str(metadata.get("file_id") or metadata.get("source") or "").strip() or None
                    file_name = str(metadata.get("file_name") or metadata.get("source") or "").strip() or None
                    source = str(metadata.get("source") or "").strip() or file_name
                    page = metadata.get("page")
                    chunk_index = metadata.get("chunk_index", metadata.get("index"))
                    source_line = metadata.get("source_line")

                    try:
                        page = int(page) if page is not None else None
                    except Exception:
                        page = None
                    try:
                        chunk_index = int(chunk_index) if chunk_index is not None else None
                    except Exception:
                        chunk_index = None
                    try:
                        source_line = int(source_line) if source_line is not None else None
                    except Exception:
                        source_line = None

                    cur.execute(
                        """
                        INSERT INTO vector_documents(file_id, file_name, source, page, chunk_index, source_line, metadata, page_content, created_at)
                        VALUES(%s, %s, %s, %s, %s, %s, %s::jsonb, %s, NOW())
                        RETURNING id
                        """,
                        (
                            file_id,
                            file_name,
                            source,
                            page,
                            chunk_index,
                            source_line,
                            json.dumps(metadata, ensure_ascii=False),
                            doc.page_content,
                        ),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    document_id = int(row[0])
                    emb_literal = _vector_literal(emb)
                    cur.execute(
                        """
                        INSERT INTO vector_embeddings(document_id, embedding, normalized, created_at)
                        VALUES(%s, %s::vector, TRUE, NOW())
                        ON CONFLICT(document_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            normalized = EXCLUDED.normalized,
                            created_at = NOW()
                        """,
                        (document_id, emb_literal),
                    )
        finally:
            conn.close()

    def add_documents(self, documents: List[Document], rebuild_index: bool = True):
        if not documents:
            return
        for doc in documents:
            if "_norm_text" not in doc.metadata:
                doc.metadata["_norm_text"] = normalize_for_match(doc.page_content)
        embeddings = self.embedder.embed_documents([doc.page_content for doc in documents])
        emb_np = np.array(embeddings, dtype="float32")
        emb_np = self._normalize_embeddings(emb_np)
        self._upsert_documents_with_vectors(documents, emb_np)

    def add_documents_with_embeddings(self, documents: List[Document], embeddings: np.ndarray, rebuild_index: bool = True):
        if not documents:
            return
        emb_np = np.array(embeddings, dtype="float32")
        emb_np = self._normalize_embeddings(emb_np)
        self._upsert_documents_with_vectors(documents, emb_np)

    def save_snapshot(self, snapshot_path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        # Not used in pgvector mode.
        return True

    def load_snapshot(self, snapshot_path, expected_signature: Optional[str] = None):
        # Not used in pgvector mode.
        return None

    def _embed_query(self, query: str) -> np.ndarray:
        q_embedding = self.embedder.embed_query(query)
        q_embedding = np.array(q_embedding, dtype="float32")
        if q_embedding.ndim == 1:
            q_embedding = np.expand_dims(q_embedding, axis=0)
        q_norm = np.linalg.norm(q_embedding, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        return q_embedding / q_norm

    def retrieve(self, query: str, top_k=25, threshold=SIMILARITY_THRESHOLD, file_ids: List[str] | None = None) -> List[Document]:
        logger.info(f"Retrieve for query (pgvector): {query}")
        q_vec = self._embed_query(query)[0]
        q_literal = _vector_literal(q_vec)

        candidate_limit = max(200, top_k * 20)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                if file_ids:
                    cur.execute(
                        """
                        SELECT d.file_id, d.file_name, d.source, d.page, d.chunk_index, d.source_line,
                               d.metadata::text, d.page_content,
                               1 - (e.embedding <=> %s::vector) AS score
                        FROM vector_embeddings e
                        JOIN vector_documents d ON d.id = e.document_id
                        WHERE d.file_id = ANY(%s)
                        ORDER BY e.embedding <=> %s::vector ASC
                        LIMIT %s
                        """,
                        (q_literal, file_ids, q_literal, candidate_limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT d.file_id, d.file_name, d.source, d.page, d.chunk_index, d.source_line,
                               d.metadata::text, d.page_content,
                               1 - (e.embedding <=> %s::vector) AS score
                        FROM vector_embeddings e
                        JOIN vector_documents d ON d.id = e.document_id
                        ORDER BY e.embedding <=> %s::vector ASC
                        LIMIT %s
                        """,
                        (q_literal, q_literal, candidate_limit),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            logger.info("[DEBUG] Returned 0 documents")
            return []

        docs: List[Document] = []
        scores: List[float] = []
        for row in rows:
            file_id, file_name, source, page, chunk_index, source_line, metadata_text, page_content, score = row
            try:
                metadata = json.loads(metadata_text or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata.setdefault("file_id", file_id)
            metadata.setdefault("file_name", file_name)
            metadata.setdefault("source", source or file_name)
            if page is not None:
                metadata.setdefault("page", int(page))
            if chunk_index is not None:
                metadata.setdefault("chunk_index", int(chunk_index))
                metadata.setdefault("index", int(chunk_index) + 1 if int(chunk_index) >= 0 else int(chunk_index))
            if source_line is not None:
                metadata.setdefault("source_line", int(source_line))
            if "_norm_text" not in metadata:
                metadata["_norm_text"] = normalize_for_match(page_content or "")
            docs.append(Document(page_content=str(page_content or ""), metadata=metadata))
            try:
                scores.append(float(score))
            except Exception:
                scores.append(0.0)

        # Lightweight lexical rescoring to keep behavior near existing FAISS pipeline.
        norm_query = normalize_for_match(query)
        query_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", norm_query) if len(t) >= 3]
        phrase_candidates: set[str] = set()
        for n in (3, 2):
            for i in range(len(query_tokens) - n + 1):
                phrase = " ".join(query_tokens[i : i + n])
                if len(phrase) >= 8:
                    phrase_candidates.add(phrase)

        combined_scores: Dict[int, float] = {idx: sc for idx, sc in enumerate(scores)}
        if query_tokens:
            for idx, doc in enumerate(docs):
                norm_doc = doc.metadata.get("_norm_text") or normalize_for_match(doc.page_content)
                phrase_hit = any(p in norm_doc for p in phrase_candidates) if phrase_candidates else False
                token_hits = sum(1 for t in query_tokens if t in norm_doc)
                if phrase_hit or token_hits:
                    boost = 0.0
                    if phrase_hit:
                        boost += 2.0
                    if token_hits:
                        boost += min(0.05 * token_hits, 0.35)
                    combined_scores[idx] = combined_scores.get(idx, 0.0) + boost

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        results: List[Document] = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            if score >= threshold:
                results.append(docs[idx])

        if not results and threshold > 0.05:
            for idx, score in ranked:
                if len(results) >= top_k:
                    break
                if score >= 0.05:
                    results.append(docs[idx])

        logger.info(f"[DEBUG] Returned {len(results)} documents")
        return results
