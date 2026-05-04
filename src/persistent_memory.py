import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from conversation_state import default_conversation_state

logger = logging.getLogger(__name__)


class PersistentMemory:
    def __init__(self, db_path: str = "../data/memory.db", max_history: int = 10, embedder=None):
        self.db_path = db_path
        self.max_history = max_history
        self.embedder = embedder  # kept for compatibility, but not used
        self._init_db()

    @staticmethod
    def _scoped_session_id(session_id: str = "default", user_id: Optional[str] = None) -> str:
        sid = str(session_id or "default").strip() or "default"
        uid = str(user_id or "").strip()
        if not uid:
            return sid
        return f"user:{uid}:session:{sid}"

    def _init_db(self):
        try:
            db_parent = Path(self.db_path).resolve().parent
            db_parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        query TEXT,
                        response TEXT,
                        chunk_index INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS file_summaries (
                        file_id TEXT PRIMARY KEY,
                        summary TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_state (
                        session_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
                logger.info("Khoi tao co so du lieu lich su thanh cong.")
        except sqlite3.Error as e:
            logger.error("Loi khi khoi tao co so du lieu: %s", e)
            raise

    def add_to_history(
        self,
        query: str,
        response: str,
        session_id: str = "default",
        chunk_index: Optional[int] = None,
        user_id: Optional[str] = None,
    ):
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO history (session_id, query, response, chunk_index) VALUES (?, ?, ?, ?)",
                    (scoped_session_id, query, response, chunk_index),
                )
                conn.commit()
                cursor.execute(
                    """
                    DELETE FROM history
                    WHERE session_id = ?
                      AND id NOT IN (
                          SELECT id FROM history
                          WHERE session_id = ?
                          ORDER BY timestamp DESC, id DESC
                          LIMIT ?
                      )
                    """,
                    (scoped_session_id, scoped_session_id, self.max_history),
                )
                conn.commit()
                logger.debug("Da them vao lich su: Query=%s, Session=%s", query, scoped_session_id)
        except sqlite3.Error as e:
            logger.error("Loi khi them vao lich su: %s", e)

    def get_context(
        self,
        query: str,
        session_id: str = "default",
        chunk_index: Optional[int] = None,
        max_rows: int = 10,
        user_id: Optional[str] = None,
    ) -> str:
        _ = query
        _ = chunk_index
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT query, response, timestamp FROM history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC, id DESC LIMIT ?
                    """,
                    (scoped_session_id, max_rows),
                )
                rows = cursor.fetchall()
                logger.info(
                    "[DEBUG] get_context: session_id='%s', found %s rows. DB path: %s",
                    scoped_session_id,
                    len(rows),
                    self.db_path,
                )

                rows_chronological = rows[::-1]
                context = "\n".join(
                    [f"[{row[2]}] Query: {row[0]}\nResponse: {row[1]}" for row in rows_chronological]
                )
                if context:
                    logger.info("Ngu canh lich su duoc truy xuat:\n%s", context)
                return context
        except sqlite3.Error as e:
            logger.error("Loi khi truy xuat lich su: %s", e)
            return ""

    def clear_session(self, session_id: str = "default", user_id: Optional[str] = None):
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history WHERE session_id = ?", (scoped_session_id,))
                cursor.execute("DELETE FROM conversation_state WHERE session_id = ?", (scoped_session_id,))
                conn.commit()
                logger.info("Da xoa lich su cua phien %s", scoped_session_id)
        except sqlite3.Error as e:
            logger.error("Loi khi xoa lich su: %s", e)

    def save_summary(self, file_id: str, summary: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO file_summaries (file_id, summary, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (file_id, summary),
                )
                conn.commit()
                logger.info("Saved summary for file_id=%s", file_id)
        except sqlite3.Error as e:
            logger.error("Loi khi luu summary cho %s: %s", file_id, e)

    def get_summary(self, file_id: str) -> Optional[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT summary FROM file_summaries WHERE file_id = ?", (file_id,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logger.error("Loi khi lay summary cho %s: %s", file_id, e)
            return None

    def get_structured_state(self, session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
        base = default_conversation_state()
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_json FROM conversation_state WHERE session_id = ?",
                    (scoped_session_id,),
                )
                row = cursor.fetchone()
                if not row or not row[0]:
                    return base

                loaded = json.loads(row[0])
                if not isinstance(loaded, dict):
                    return base

                merged = base.copy()
                merged.update(loaded)

                entities = base.get("entities", {}).copy()
                entities.update(loaded.get("entities", {}) if isinstance(loaded.get("entities"), dict) else {})
                merged["entities"] = entities

                referents = base.get("referents", {}).copy()
                referents.update(loaded.get("referents", {}) if isinstance(loaded.get("referents"), dict) else {})
                merged["referents"] = referents
                return merged
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error("Loi khi lay structured state cho session %s: %s", scoped_session_id, e)
            return base

    def save_structured_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        base = default_conversation_state()
        if isinstance(state, dict):
            base.update(state)

        entities = base.get("entities") if isinstance(base.get("entities"), dict) else {}
        referents = base.get("referents") if isinstance(base.get("referents"), dict) else {}

        default_entities = default_conversation_state()["entities"]
        merged_entities = default_entities.copy()
        merged_entities.update(entities)
        base["entities"] = merged_entities

        default_referents = default_conversation_state()["referents"]
        merged_referents = default_referents.copy()
        merged_referents.update(referents)
        base["referents"] = merged_referents

        try:
            payload = json.dumps(base, ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO conversation_state(session_id, state_json, updated_at)
                    VALUES(?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (scoped_session_id, payload),
                )
                conn.commit()
            return base
        except sqlite3.Error as e:
            logger.error("Loi khi luu structured state cho session %s: %s", scoped_session_id, e)
            return self.get_structured_state(session_id=session_id, user_id=user_id)
