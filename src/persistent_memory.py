import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        scoped_session_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_id TEXT,
                        title TEXT NOT NULL DEFAULT 'Phiên mới',
                        selected_program_id TEXT,
                        selected_file_ids_json TEXT NOT NULL DEFAULT '[]',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        archived_at DATETIME
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                    ON chat_sessions(user_id, updated_at DESC)
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scoped_session_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        user_id TEXT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        citations_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(scoped_session_id) REFERENCES chat_sessions(scoped_session_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                    ON chat_messages(scoped_session_id, created_at ASC, id ASC)
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS legacy_chat_migrations (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        migrated_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
                cursor.execute("DELETE FROM chat_messages WHERE scoped_session_id = ?", (scoped_session_id,))
                conn.commit()
                logger.info("Da xoa lich su cua phien %s", scoped_session_id)
        except sqlite3.Error as e:
            logger.error("Loi khi xoa lich su: %s", e)

    @staticmethod
    def _json_list(values: Optional[List[str]]) -> str:
        cleaned = [str(item).strip() for item in (values or []) if str(item or "").strip()]
        return json.dumps(list(dict.fromkeys(cleaned)), ensure_ascii=False)

    @staticmethod
    def _parse_json_list(value: Any) -> List[str]:
        try:
            parsed = json.loads(str(value or "[]"))
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item or "").strip()]

    @staticmethod
    def _serialize_chat_session(row: sqlite3.Row | tuple) -> Dict[str, Any]:
        data = dict(row) if isinstance(row, sqlite3.Row) else {}
        return {
            "id": str(data.get("session_id") or ""),
            "session_id": str(data.get("session_id") or ""),
            "title": str(data.get("title") or "Phiên mới"),
            "selected_program_id": data.get("selected_program_id"),
            "selected_file_ids": PersistentMemory._parse_json_list(data.get("selected_file_ids_json")),
            "created_at": str(data.get("created_at") or ""),
            "updated_at": str(data.get("updated_at") or ""),
            "archived_at": data.get("archived_at"),
        }

    def ensure_chat_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        selected_program_id: Optional[str] = None,
        selected_file_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        sid = str(session_id or "default").strip() or "default"
        title_value = str(title or "").strip() or "Phiên mới"
        selected_program_value = str(selected_program_id or "").strip() or None
        selected_files_json = self._json_list(selected_file_ids)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    "SELECT * FROM chat_sessions WHERE scoped_session_id = ?",
                    (scoped_session_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO chat_sessions(
                            scoped_session_id, session_id, user_id, title,
                            selected_program_id, selected_file_ids_json, created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (
                            scoped_session_id,
                            sid,
                            str(user_id or "").strip() or None,
                            title_value,
                            selected_program_value,
                            selected_files_json,
                        ),
                    )
                else:
                    existing_title = str(existing["title"] or "").strip()
                    update_title = existing_title if existing_title and existing_title != "Phiên mới" else title_value
                    update_program = selected_program_value or existing["selected_program_id"]
                    update_files = selected_files_json if selected_file_ids is not None else existing["selected_file_ids_json"]
                    conn.execute(
                        """
                        UPDATE chat_sessions
                        SET title = ?,
                            selected_program_id = ?,
                            selected_file_ids_json = ?,
                            updated_at = CURRENT_TIMESTAMP,
                            archived_at = NULL
                        WHERE scoped_session_id = ?
                        """,
                        (update_title, update_program, update_files, scoped_session_id),
                    )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM chat_sessions WHERE scoped_session_id = ?",
                    (scoped_session_id,),
                ).fetchone()
                return self._serialize_chat_session(row)
        except sqlite3.Error as e:
            logger.error("Loi khi tao/cap nhat chat session %s: %s", scoped_session_id, e)
            return {
                "id": sid,
                "session_id": sid,
                "title": title_value,
                "selected_program_id": selected_program_value,
                "selected_file_ids": self._parse_json_list(selected_files_json),
                "created_at": "",
                "updated_at": "",
                "archived_at": None,
            }

    def list_chat_sessions(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        if not uid:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                query = """
                    SELECT * FROM chat_sessions
                    WHERE user_id = ?
                """
                params: List[Any] = [uid]
                if not include_archived:
                    query += " AND archived_at IS NULL"
                query += " ORDER BY updated_at DESC, created_at DESC"
                rows = conn.execute(query, params).fetchall()
                return [self._serialize_chat_session(row) for row in rows]
        except sqlite3.Error as e:
            logger.error("Loi khi lay danh sach chat sessions cua user %s: %s", uid, e)
            return []

    @staticmethod
    def _legacy_chat_title(query: Any) -> str:
        text = " ".join(str(query or "").split())
        if not text:
            return "Phiên cũ"
        return text[:77] + "..." if len(text) > 80 else text

    def migrate_legacy_history_to_chat_sessions(
        self,
        user_id: str,
        limit_sessions: int = 50,
        max_pairs_per_session: int = 50,
    ) -> int:
        """Recover pre-account chat history rows into account-scoped chat sessions.

        Older frontend builds stored the sidebar in browser localStorage and only
        persisted Q/A pairs in `history` keyed by raw session_id. Once account
        sessions became the source of truth, those rows need a one-time adoption
        path so logout/login does not make old conversations disappear.
        """
        uid = str(user_id or "").strip()
        if not uid:
            return 0

        try:
            limit = max(1, min(int(limit_sessions or 50), 200))
            max_pairs = max(1, min(int(max_pairs_per_session or 50), 200))
        except (TypeError, ValueError):
            limit = 50
            max_pairs = 50

        migrated = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                legacy_sessions = conn.execute(
                    """
                    SELECT session_id,
                           MIN(timestamp) AS created_at,
                           MAX(timestamp) AS updated_at,
                           COUNT(*) AS row_count
                    FROM history
                    WHERE session_id IS NOT NULL
                      AND TRIM(session_id) != ''
                      AND session_id NOT LIKE 'user:%:session:%'
                    GROUP BY session_id
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

                for legacy in legacy_sessions:
                    sid = str(legacy["session_id"] or "").strip()
                    if not sid:
                        continue
                    claim = conn.execute(
                        "SELECT user_id FROM legacy_chat_migrations WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    if claim is not None and str(claim["user_id"] or "").strip() != uid:
                        continue
                    scoped_session_id = self._scoped_session_id(session_id=sid, user_id=uid)
                    existing = conn.execute(
                        "SELECT 1 FROM chat_sessions WHERE scoped_session_id = ?",
                        (scoped_session_id,),
                    ).fetchone()
                    if existing is not None:
                        continue

                    first_row = conn.execute(
                        """
                        SELECT query FROM history
                        WHERE session_id = ?
                        ORDER BY timestamp ASC, id ASC
                        LIMIT 1
                        """,
                        (sid,),
                    ).fetchone()

                    selected_program_id = None
                    selected_file_ids: List[str] = []
                    state_row = conn.execute(
                        "SELECT state_json FROM conversation_state WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    if state_row is not None:
                        try:
                            state = json.loads(state_row["state_json"] or "{}")
                        except json.JSONDecodeError:
                            state = {}
                        if isinstance(state, dict):
                            selected_program_id = (
                                state.get("selected_program_id")
                                or state.get("current_program_id")
                                or state.get("program_id")
                            )
                            raw_file_ids = state.get("selected_file_ids") or state.get("file_ids") or []
                            if isinstance(raw_file_ids, list):
                                selected_file_ids = [str(item).strip() for item in raw_file_ids if str(item or "").strip()]

                    conn.execute(
                        """
                        INSERT INTO chat_sessions(
                            scoped_session_id, session_id, user_id, title,
                            selected_program_id, selected_file_ids_json,
                            created_at, updated_at, archived_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            scoped_session_id,
                            sid,
                            uid,
                            self._legacy_chat_title(first_row["query"] if first_row else ""),
                            str(selected_program_id or "").strip() or None,
                            self._json_list(selected_file_ids),
                            legacy["created_at"] or None,
                            legacy["updated_at"] or None,
                        ),
                    )

                    rows = conn.execute(
                        """
                        SELECT query, response, timestamp FROM history
                        WHERE session_id = ?
                        ORDER BY timestamp ASC, id ASC
                        LIMIT ?
                        """,
                        (sid, max_pairs),
                    ).fetchall()
                    for row in rows:
                        query_text = str(row["query"] or "").strip()
                        response_text = str(row["response"] or "").strip()
                        created_at = row["timestamp"] or None
                        if query_text:
                            conn.execute(
                                """
                                INSERT INTO chat_messages(
                                    scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                                )
                                VALUES(?, ?, ?, 'user', ?, '[]', ?)
                                """,
                                (scoped_session_id, sid, uid, query_text, created_at),
                            )
                        if response_text:
                            conn.execute(
                                """
                                INSERT INTO chat_messages(
                                    scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                                )
                                VALUES(?, ?, ?, 'assistant', ?, '[]', ?)
                                """,
                                (scoped_session_id, sid, uid, response_text, created_at),
                            )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO legacy_chat_migrations(session_id, user_id, migrated_at)
                        VALUES(?, ?, CURRENT_TIMESTAMP)
                        """,
                        (sid, uid),
                    )
                    migrated += 1
                conn.commit()
                if migrated:
                    logger.info("Migrated %s legacy history sessions into chat_sessions for user %s", migrated, uid)
                return migrated
        except sqlite3.Error as e:
            logger.error("Loi khi migrate legacy history sang chat sessions cho user %s: %s", uid, e)
            return migrated

    def update_chat_session(
        self,
        session_id: str,
        user_id: str,
        title: Optional[str] = None,
        selected_program_id: Optional[str] = None,
        selected_file_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    "SELECT * FROM chat_sessions WHERE scoped_session_id = ? AND user_id = ?",
                    (scoped_session_id, str(user_id or "").strip()),
                ).fetchone()
                if existing is None:
                    return None
                title_value = str(title).strip() if title is not None else existing["title"]
                program_value = (
                    str(selected_program_id).strip()
                    if selected_program_id is not None and str(selected_program_id).strip()
                    else (None if selected_program_id is not None else existing["selected_program_id"])
                )
                files_json = (
                    self._json_list(selected_file_ids)
                    if selected_file_ids is not None
                    else existing["selected_file_ids_json"]
                )
                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?, selected_program_id = ?, selected_file_ids_json = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE scoped_session_id = ? AND user_id = ?
                    """,
                    (title_value or "Phiên mới", program_value, files_json, scoped_session_id, user_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM chat_sessions WHERE scoped_session_id = ? AND user_id = ?",
                    (scoped_session_id, user_id),
                ).fetchone()
                return self._serialize_chat_session(row)
        except sqlite3.Error as e:
            logger.error("Loi khi cap nhat chat session %s: %s", scoped_session_id, e)
            return None

    def archive_chat_session(self, session_id: str, user_id: str) -> bool:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    UPDATE chat_sessions
                    SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE scoped_session_id = ? AND user_id = ? AND archived_at IS NULL
                    """,
                    (scoped_session_id, str(user_id or "").strip()),
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("Loi khi archive chat session %s: %s", scoped_session_id, e)
            return False

    def add_chat_message(
        self,
        session_id: str,
        user_id: Optional[str],
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        sid = str(session_id or "default").strip() or "default"
        role_value = str(role or "").strip().lower()
        if role_value not in {"user", "assistant", "system"}:
            role_value = "assistant"
        content_value = str(content or "").strip()
        if not content_value:
            return None
        try:
            citations_json = json.dumps(citations or [], ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_messages(scoped_session_id, session_id, user_id, role, content, citations_json)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scoped_session_id,
                        sid,
                        str(user_id or "").strip() or None,
                        role_value,
                        content_value,
                        citations_json,
                    ),
                )
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE scoped_session_id = ?",
                    (scoped_session_id,),
                )
                conn.commit()
                return int(cursor.lastrowid)
        except sqlite3.Error as e:
            logger.error("Loi khi luu chat message cho session %s: %s", scoped_session_id, e)
            return None

    def get_chat_messages(self, session_id: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, role, content, citations_json, created_at
                    FROM chat_messages
                    WHERE scoped_session_id = ? AND user_id = ?
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                    """,
                    (scoped_session_id, str(user_id or "").strip(), max(1, min(int(limit or 50), 200))),
                ).fetchall()
                messages: List[Dict[str, Any]] = []
                for row in rows:
                    try:
                        citations = json.loads(row["citations_json"] or "[]")
                    except json.JSONDecodeError:
                        citations = []
                    messages.append(
                        {
                            "id": int(row["id"]),
                            "role": str(row["role"] or ""),
                            "content": str(row["content"] or ""),
                            "citations": citations if isinstance(citations, list) else [],
                            "created_at": str(row["created_at"] or ""),
                        }
                    )
                return messages
        except (sqlite3.Error, ValueError) as e:
            logger.error("Loi khi lay chat messages cho session %s: %s", scoped_session_id, e)
            return []

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
