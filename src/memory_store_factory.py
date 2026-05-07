import os
from typing import Any

from persistent_memory import PersistentMemory
from persistent_memory_pg import PostgresPersistentMemory
from supabase_support import pgvector_enabled


def build_memory_store(max_history: int = 25, sqlite_db_path: str = "../data/memory.db", embedder: Any = None):
    """
    Build memory store based on env.
    - If SUPABASE_DB_URL is set, use Postgres-backed memory.
    - Else fallback to SQLite-backed PersistentMemory.
    """
    if pgvector_enabled():
        return PostgresPersistentMemory(
            db_url=str(os.getenv("SUPABASE_DB_URL", "")).strip(),
            max_history=max_history,
            embedder=embedder,
        )
    return PersistentMemory(db_path=sqlite_db_path, max_history=max_history, embedder=embedder)
