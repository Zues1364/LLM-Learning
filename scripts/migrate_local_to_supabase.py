import argparse
import json
import os
import pickle
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_loader import load_env
from langchain_core.documents import Document
from runtime_paths import CACHE_DIR, DATA_DIR, MEMORY_DB, PDF_DIR, RESOURCE_DIR
from supabase_support import get_pg_connection, pgvector_enabled, supabase_enabled
from supabase_support import _normalize_storage_key
from supabase_support import SupabaseBlobStore
from utils import FAISSVectorStore, VietnameseEmbedder
from vector_store_pg import PGVectorStore


TABLES_MEMORY = [
    "history",
    "conversation_state",
    "file_summaries",
    "chat_sessions",
    "chat_messages",
    "legacy_chat_migrations",
]


@dataclass
class MigrateStats:
    sqlite_rows: Dict[str, int]
    pg_rows: Dict[str, int]
    uploaded_objects: int
    migrated_vector_docs: int
    reembedded_vector_docs: int
    skipped_vector_docs: int


def _table_exists_sqlite(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _sqlite_count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists_sqlite(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0] or 0)


def _pg_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row = cur.fetchone()
        return int(row[0] or 0)


def _truncate_pg_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chat_messages RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE chat_sessions RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE history RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE conversation_state RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE file_summaries RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE legacy_chat_migrations RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE vector_embeddings RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE vector_documents RESTART IDENTITY CASCADE")


def _migrate_memory(sqlite_db: Path, reset_postgres: bool) -> Tuple[Dict[str, int], Dict[str, int]]:
    sqlite_rows = {name: 0 for name in TABLES_MEMORY}
    pg_rows = {name: 0 for name in TABLES_MEMORY}

    if not sqlite_db.exists():
        return sqlite_rows, pg_rows

    sqlite_conn = sqlite3.connect(str(sqlite_db))
    sqlite_conn.row_factory = sqlite3.Row
    pg_conn = get_pg_connection()
    try:
        if reset_postgres:
            _truncate_pg_tables(pg_conn)

        for name in TABLES_MEMORY:
            sqlite_rows[name] = _sqlite_count(sqlite_conn, name)

        with pg_conn.cursor() as cur:
            if sqlite_rows["history"] > 0:
                rows = sqlite_conn.execute(
                    "SELECT session_id, query, response, chunk_index, timestamp FROM history ORDER BY id ASC"
                ).fetchall()
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO history(session_id, query, response, chunk_index, timestamp)
                        VALUES(%s, %s, %s, %s, %s)
                        """,
                        (row["session_id"], row["query"], row["response"], row["chunk_index"], row["timestamp"]),
                    )

            if sqlite_rows["conversation_state"] > 0:
                rows = sqlite_conn.execute(
                    "SELECT session_id, state_json, updated_at FROM conversation_state"
                ).fetchall()
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO conversation_state(session_id, state_json, updated_at)
                        VALUES(%s, %s::jsonb, %s)
                        ON CONFLICT(session_id) DO UPDATE SET
                          state_json = EXCLUDED.state_json,
                          updated_at = EXCLUDED.updated_at
                        """,
                        (row["session_id"], row["state_json"], row["updated_at"]),
                    )

            if sqlite_rows["file_summaries"] > 0:
                rows = sqlite_conn.execute(
                    "SELECT file_id, summary, created_at FROM file_summaries"
                ).fetchall()
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO file_summaries(file_id, summary, created_at)
                        VALUES(%s, %s, %s)
                        ON CONFLICT(file_id) DO UPDATE SET
                          summary = EXCLUDED.summary,
                          created_at = EXCLUDED.created_at
                        """,
                        (row["file_id"], row["summary"], row["created_at"]),
                    )

            if sqlite_rows["chat_sessions"] > 0:
                rows = sqlite_conn.execute(
                    """
                    SELECT scoped_session_id, session_id, user_id, title, selected_program_id,
                           selected_file_ids_json, created_at, updated_at, archived_at
                    FROM chat_sessions
                    """
                ).fetchall()
                for row in rows:
                    selected_file_ids_json = row["selected_file_ids_json"] or "[]"
                    cur.execute(
                        """
                        INSERT INTO chat_sessions(
                            scoped_session_id, session_id, user_id, title, selected_program_id,
                            selected_file_ids_json, created_at, updated_at, archived_at
                        )
                        VALUES(%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                        ON CONFLICT(scoped_session_id) DO UPDATE SET
                          title = EXCLUDED.title,
                          selected_program_id = EXCLUDED.selected_program_id,
                          selected_file_ids_json = EXCLUDED.selected_file_ids_json,
                          updated_at = EXCLUDED.updated_at,
                          archived_at = EXCLUDED.archived_at
                        """,
                        (
                            row["scoped_session_id"],
                            row["session_id"],
                            row["user_id"],
                            row["title"],
                            row["selected_program_id"],
                            selected_file_ids_json,
                            row["created_at"],
                            row["updated_at"],
                            row["archived_at"],
                        ),
                    )

            if sqlite_rows["chat_messages"] > 0:
                rows = sqlite_conn.execute(
                    """
                    SELECT scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                    FROM chat_messages
                    ORDER BY id ASC
                    """
                ).fetchall()
                for row in rows:
                    citations_json = row["citations_json"] or "[]"
                    cur.execute(
                        """
                        INSERT INTO chat_messages(
                            scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                        )
                        VALUES(%s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            row["scoped_session_id"],
                            row["session_id"],
                            row["user_id"],
                            row["role"],
                            row["content"],
                            citations_json,
                            row["created_at"],
                        ),
                    )

            if sqlite_rows["legacy_chat_migrations"] > 0:
                rows = sqlite_conn.execute(
                    "SELECT session_id, user_id, migrated_at FROM legacy_chat_migrations"
                ).fetchall()
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO legacy_chat_migrations(session_id, user_id, migrated_at)
                        VALUES(%s, %s, %s)
                        ON CONFLICT(session_id) DO UPDATE SET
                          user_id = EXCLUDED.user_id,
                          migrated_at = EXCLUDED.migrated_at
                        """,
                        (row["session_id"], row["user_id"], row["migrated_at"]),
                    )

        for name in TABLES_MEMORY:
            pg_rows[name] = _pg_count(pg_conn, name)
        return sqlite_rows, pg_rows
    finally:
        sqlite_conn.close()
        pg_conn.close()


def _upload_file(store: SupabaseBlobStore, local_path: Path, object_key: str) -> bool:
    suffix = local_path.suffix.lower()
    if suffix == ".pdf":
        ctype = "application/pdf"
    elif suffix in {".html", ".htm"}:
        ctype = "text/html"
    elif suffix == ".json":
        ctype = "application/json"
    elif suffix == ".npy":
        ctype = "application/octet-stream"
    elif suffix == ".pkl":
        ctype = "application/octet-stream"
    else:
        ctype = "application/octet-stream"
    store.upload_file(object_key, local_path, content_type=ctype)
    return True


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _migrate_storage(data_dir: Path) -> int:
    store = SupabaseBlobStore()
    store.ensure_bucket()
    uploaded = 0
    existing_objects = {obj.key: obj.size for obj in store.list_objects("")}

    def _upload_if_needed(file_path: Path, object_key: str) -> None:
        nonlocal uploaded
        normalized_key = _normalize_storage_key(object_key)
        expected_size = file_path.stat().st_size
        if existing_objects.get(normalized_key) == expected_size:
            return
        _upload_file(store, file_path, object_key)
        existing_objects[normalized_key] = expected_size
        uploaded += 1

    def _upload_tree(local_root: Path, key_prefix: str):
        if not local_root.exists():
            return
        for file_path in _iter_files(local_root):
            rel = file_path.relative_to(local_root).as_posix()
            object_key = f"{key_prefix}/{rel}" if rel else key_prefix
            _upload_if_needed(file_path, object_key)

    # transcripts
    transcripts = data_dir / "pdfs"
    if transcripts.exists():
        for file_path in _iter_files(transcripts):
            object_key = f"sessions/global/pdfs/{file_path.name}"
            _upload_if_needed(file_path, object_key)

    # resources global
    resources = data_dir / "resources"
    if resources.exists():
        cfg = resources / "config.json"
        if cfg.exists():
            _upload_if_needed(cfg, "resources/global/config.json")
        _upload_tree(resources / "pdfs", "resources/global/pdf")
        _upload_tree(resources / "html", "resources/global/html")

        sessions_root = resources / "sessions"
        if sessions_root.exists():
            for scope_dir in sessions_root.iterdir():
                if not scope_dir.is_dir():
                    continue
                sid = scope_dir.name
                cfg_file = scope_dir / "config.json"
                if cfg_file.exists():
                    _upload_if_needed(cfg_file, f"resources/{sid}/config.json")
                _upload_tree(scope_dir / "pdfs", f"resources/{sid}/pdf")
                _upload_tree(scope_dir / "html", f"resources/{sid}/html")

        users_root = resources / "users"
        if users_root.exists():
            for scope_dir in users_root.iterdir():
                if not scope_dir.is_dir():
                    continue
                uid = scope_dir.name
                cfg_file = scope_dir / "config.json"
                if cfg_file.exists():
                    _upload_if_needed(cfg_file, f"resources/users/{uid}/config.json")
                _upload_tree(scope_dir / "pdfs", f"resources/users/{uid}/pdf")
                _upload_tree(scope_dir / "html", f"resources/users/{uid}/html")

    # cache artifacts
    _upload_tree(data_dir / "cache", "cache")
    return uploaded


def _load_cache_pair(cache_file: Path) -> Optional[Tuple[List[Document], np.ndarray]]:
    if cache_file.suffix.lower() != ".pkl":
        return None
    if cache_file.name.endswith(".metadata.pkl"):
        return None
    emb_file = cache_file.with_name(f"{cache_file.stem}_embeddings.npy")
    if not emb_file.exists():
        return None
    try:
        with open(cache_file, "rb") as fh:
            docs = pickle.load(fh)
        emb = np.load(emb_file)
    except Exception:
        return None
    if not isinstance(docs, list):
        return None
    if not isinstance(emb, np.ndarray):
        return None
    if emb.ndim == 1:
        emb = np.expand_dims(emb, axis=0)
    if len(docs) != emb.shape[0]:
        return None
    clean_docs: List[Document] = []
    for item in docs:
        if isinstance(item, Document):
            clean_docs.append(item)
        elif isinstance(item, dict):
            clean_docs.append(
                Document(
                    page_content=str(item.get("page_content") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        else:
            return None
    return clean_docs, emb.astype("float32")


def _batched(items: Sequence[Any], size: int = 64) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _migrate_vectors_from_cache(data_dir: Path, reset_vectors: bool) -> Tuple[int, int, int]:
    cache_dir = data_dir / "cache"
    if not cache_dir.exists():
        return 0, 0, 0

    migration_embedder = VietnameseEmbedder()
    dim = int(os.getenv("PGVECTOR_EMBEDDING_DIM", str(migration_embedder.embedding_dim)) or str(migration_embedder.embedding_dim))
    if int(getattr(migration_embedder, "embedding_dim", dim) or dim) != dim:
        raise RuntimeError(
            f"PGVECTOR_EMBEDDING_DIM={dim} does not match runtime embedder dimension "
            f"{migration_embedder.embedding_dim}."
        )

    store = PGVectorStore([], migration_embedder)
    if reset_vectors:
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vector_embeddings RESTART IDENTITY CASCADE")
                cur.execute("TRUNCATE TABLE vector_documents RESTART IDENTITY CASCADE")
        finally:
            conn.close()

    migrated = 0
    reembedded = 0
    skipped = 0
    for cache_file in cache_dir.glob("*.pkl"):
        pair = _load_cache_pair(cache_file)
        if pair is None:
            continue
        docs, emb = pair
        try:
            if emb.ndim == 2 and emb.shape[1] == dim:
                store.add_documents_with_embeddings(docs, emb)
                migrated += len(docs)
                continue
            for batch in _batched(docs, 64):
                store.add_documents(list(batch))
            migrated += len(docs)
            reembedded += len(docs)
        except Exception:
            skipped += len(docs)
    return migrated, reembedded, skipped


def _verify_retrieval_from_cache(data_dir: Path, top_k: int = 5) -> Dict[str, Any]:
    cache_dir = data_dir / "cache"
    embedder = VietnameseEmbedder()
    expected_dim = int(getattr(embedder, "embedding_dim", 1024) or 1024)
    local_store = FAISSVectorStore([], embedder)
    pg_store = PGVectorStore([], embedder)

    docs_all: List[Document] = []
    emb_all: List[np.ndarray] = []
    skipped_by_dim: Dict[str, int] = {}
    for cache_file in cache_dir.glob("*.pkl"):
        pair = _load_cache_pair(cache_file)
        if pair is None:
            continue
        docs, emb = pair
        if emb.ndim != 2 or emb.shape[1] != expected_dim:
            dim_key = str(emb.shape[1] if emb.ndim == 2 else "invalid")
            skipped_by_dim[dim_key] = skipped_by_dim.get(dim_key, 0) + len(docs)
            continue
        docs_all.extend(docs)
        emb_all.append(emb)
    if docs_all and emb_all:
        emb_np = np.vstack(emb_all).astype("float32")
        local_store.add_documents_with_embeddings(docs_all, emb_np)

    queries = [
        "voi 6.5 ielts toi co du dieu kien tieng anh de ra truong khong",
        "ca 1 bat dau tu may gio va ket thuc luc may gio",
        "toi can lich hoc mon thi giac may",
        "thay Le Khanh Trinh ki nay day nhung mon nao",
    ]

    summary: Dict[str, Any] = {
        "queries": [],
        "expected_embedding_dim": expected_dim,
        "skipped_local_docs_by_dim": skipped_by_dim,
    }
    for q in queries:
        local_docs = local_store.retrieve(q, top_k=top_k) if local_store.documents else []
        pg_docs = pg_store.retrieve(q, top_k=top_k)
        local_sources = [str(d.metadata.get("source") or d.metadata.get("file_name") or "") for d in local_docs]
        pg_sources = [str(d.metadata.get("source") or d.metadata.get("file_name") or "") for d in pg_docs]
        overlap = len(set(local_sources) & set(pg_sources))
        summary["queries"].append(
            {
                "query": q,
                "local_top_sources": local_sources,
                "pg_top_sources": pg_sources,
                "overlap": overlap,
            }
        )
    return summary


def _print_counts(sqlite_rows: Dict[str, int], pg_rows: Dict[str, int]) -> None:
    print("\n=== Memory Table Counts ===")
    for name in TABLES_MEMORY:
        print(f"- {name}: sqlite={sqlite_rows.get(name, 0)} | postgres={pg_rows.get(name, 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local runtime data to Supabase (Postgres + Storage + pgvector).")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Local runtime data directory (default: APP_DATA_DIR/data).")
    parser.add_argument("--sqlite-db", default=str(MEMORY_DB), help="SQLite memory db path.")
    parser.add_argument("--skip-memory", action="store_true", help="Skip SQLite -> Postgres memory migration.")
    parser.add_argument("--skip-storage", action="store_true", help="Skip local files -> Supabase Storage upload.")
    parser.add_argument("--skip-vectors", action="store_true", help="Skip cache embeddings -> pgvector migration.")
    parser.add_argument("--reset-postgres", action="store_true", help="Truncate destination Postgres tables before migrate.")
    parser.add_argument("--verify-retrieval", action="store_true", help="Run retrieval comparison checks after migrate.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval verification.")
    parser.add_argument("--report-json", default="", help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    sqlite_db = Path(args.sqlite_db).resolve()

    if not supabase_enabled():
        print("ERROR: Supabase storage env is missing (SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY).")
        return 2
    if not pgvector_enabled():
        print("ERROR: SUPABASE_DB_URL is required for Postgres/pgvector migration.")
        return 2

    sqlite_rows = {name: 0 for name in TABLES_MEMORY}
    pg_rows = {name: 0 for name in TABLES_MEMORY}
    uploaded_objects = 0
    migrated_vector_docs = 0
    reembedded_vector_docs = 0
    skipped_vector_docs = 0
    retrieval_report: Dict[str, Any] = {}

    if not args.skip_memory:
        sqlite_rows, pg_rows = _migrate_memory(sqlite_db=sqlite_db, reset_postgres=args.reset_postgres)
        _print_counts(sqlite_rows, pg_rows)
    else:
        print("Skip memory migration.")

    if not args.skip_storage:
        uploaded_objects = _migrate_storage(data_dir)
        print(f"\nUploaded storage objects: {uploaded_objects}")
    else:
        print("Skip storage migration.")

    if not args.skip_vectors:
        migrated_vector_docs, reembedded_vector_docs, skipped_vector_docs = _migrate_vectors_from_cache(
            data_dir=data_dir,
            reset_vectors=args.reset_postgres,
        )
        print(f"\nMigrated vector docs: {migrated_vector_docs}")
        print(f"Re-embedded vector docs: {reembedded_vector_docs}")
        print(f"Skipped vector docs: {skipped_vector_docs}")
    else:
        print("Skip vector migration.")

    if args.verify_retrieval:
        retrieval_report = _verify_retrieval_from_cache(data_dir=data_dir, top_k=max(1, int(args.top_k)))
        print("\n=== Retrieval Verification ===")
        for item in retrieval_report.get("queries", []):
            print(f"- Query: {item['query']}")
            print(f"  overlap={item['overlap']}")
            print(f"  local={json.dumps(item['local_top_sources'], ensure_ascii=True)}")
            print(f"  pg={json.dumps(item['pg_top_sources'], ensure_ascii=True)}")

    stats = MigrateStats(
        sqlite_rows=sqlite_rows,
        pg_rows=pg_rows,
        uploaded_objects=uploaded_objects,
        migrated_vector_docs=migrated_vector_docs,
        reembedded_vector_docs=reembedded_vector_docs,
        skipped_vector_docs=skipped_vector_docs,
    )
    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "stats": {
                        "sqlite_rows": stats.sqlite_rows,
                        "pg_rows": stats.pg_rows,
                        "uploaded_objects": stats.uploaded_objects,
                        "migrated_vector_docs": stats.migrated_vector_docs,
                        "reembedded_vector_docs": stats.reembedded_vector_docs,
                        "skipped_vector_docs": stats.skipped_vector_docs,
                    },
                    "retrieval_report": retrieval_report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
