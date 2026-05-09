import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from conversation_state import default_conversation_state
from supabase_support import get_pg_connection

logger = logging.getLogger(__name__)


class PostgresPersistentMemory:
    """
    Drop-in replacement for PersistentMemory backed by Postgres.
    Keeps method signatures compatible with existing app/mcp usage.
    """

    def __init__(self, db_url: str, max_history: int = 10, embedder=None):
        self.db_url = db_url
        self.db_path = "postgres"
        self.max_history = max_history
        self.embedder = embedder
        self._init_db()

    @staticmethod
    def _scoped_session_id(session_id: str = "default", user_id: Optional[str] = None) -> str:
        sid = str(session_id or "default").strip() or "default"
        uid = str(user_id or "").strip()
        if not uid:
            return sid
        return f"user:{uid}:session:{sid}"

    @staticmethod
    def _normalize_title_key(value: Any) -> str:
        text = unicodedata.normalize("NFD", str(value or "").strip().lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_placeholder_chat_title(value: Any) -> bool:
        raw = str(value or "").strip().lower()
        normalized = PostgresPersistentMemory._normalize_title_key(value)
        return (
            not normalized
            or raw in {"phiãªn má»›i", "phiãªn cå©"}
            or normalized in {"phien moi", "phien cu"}
            or re.fullmatch(r"phien \d+", normalized) is not None
        )

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
    def _legacy_chat_title(query: Any) -> str:
        text = " ".join(str(query or "").split())
        if not text:
            return "Phiên cũ"
        return text[:77] + "..." if len(text) > 80 else text

    @staticmethod
    def _serialize_chat_session(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row.get("session_id") or ""),
            "session_id": str(row.get("session_id") or ""),
            "title": str(row.get("title") or "Phiên mới"),
            "selected_program_id": row.get("selected_program_id"),
            "selected_file_ids": PostgresPersistentMemory._parse_json_list(row.get("selected_file_ids_json")),
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "archived_at": row.get("archived_at"),
        }

    def _conn(self):
        return get_pg_connection()

    def _init_db(self):
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS history (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT,
                query TEXT,
                response TEXT,
                chunk_index INTEGER,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS file_summaries (
                file_id TEXT PRIMARY KEY,
                summary TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_state (
                session_id TEXT PRIMARY KEY,
                state_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                scoped_session_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                title TEXT NOT NULL DEFAULT 'Phiên mới',
                selected_program_id TEXT,
                selected_file_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                archived_at TIMESTAMPTZ
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
            ON chat_sessions(user_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                scoped_session_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                FOREIGN KEY(scoped_session_id) REFERENCES chat_sessions(scoped_session_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
            ON chat_messages(scoped_session_id, created_at ASC, id ASC)
            """,
            """
            CREATE TABLE IF NOT EXISTS legacy_chat_migrations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                migrated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
        ]
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for statement in ddl:
                    cur.execute(statement)
            logger.info("Khoi tao Postgres memory db thanh cong.")
        finally:
            conn.close()

    def add_to_history(
        self,
        query: str,
        response: str,
        session_id: str = "default",
        chunk_index: Optional[int] = None,
        user_id: Optional[str] = None,
    ):
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO history(session_id, query, response, chunk_index) VALUES (%s, %s, %s, %s)",
                    (scoped_session_id, query, response, chunk_index),
                )
                cur.execute(
                    """
                    DELETE FROM history
                    WHERE session_id = %s
                      AND id NOT IN (
                          SELECT id FROM history
                          WHERE session_id = %s
                          ORDER BY timestamp DESC, id DESC
                          LIMIT %s
                      )
                    """,
                    (scoped_session_id, scoped_session_id, self.max_history),
                )
        finally:
            conn.close()

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
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT query, response, timestamp
                    FROM history
                    WHERE session_id = %s
                    ORDER BY timestamp DESC, id DESC
                    LIMIT %s
                    """,
                    (scoped_session_id, max_rows),
                )
                rows = cur.fetchall()
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
        finally:
            conn.close()

    def clear_session(self, session_id: str = "default", user_id: Optional[str] = None):
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM history WHERE session_id = %s", (scoped_session_id,))
                cur.execute("DELETE FROM conversation_state WHERE session_id = %s", (scoped_session_id,))
                cur.execute("DELETE FROM chat_messages WHERE scoped_session_id = %s", (scoped_session_id,))
        finally:
            conn.close()

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

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT scoped_session_id, session_id, user_id, title, selected_program_id, selected_file_ids_json::text, created_at, updated_at, archived_at FROM chat_sessions WHERE scoped_session_id = %s",
                    (scoped_session_id,),
                )
                existing = cur.fetchone()
                if existing is None:
                    cur.execute(
                        """
                        INSERT INTO chat_sessions(
                            scoped_session_id, session_id, user_id, title,
                            selected_program_id, selected_file_ids_json, created_at, updated_at
                        )
                        VALUES(%s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW())
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
                    existing_row = {
                        "scoped_session_id": existing[0],
                        "session_id": existing[1],
                        "user_id": existing[2],
                        "title": existing[3],
                        "selected_program_id": existing[4],
                        "selected_file_ids_json": existing[5],
                        "created_at": existing[6],
                        "updated_at": existing[7],
                        "archived_at": existing[8],
                    }
                    existing_title = str(existing_row["title"] or "").strip()
                    update_title = (
                        existing_title
                        if existing_title and not self._is_placeholder_chat_title(existing_title)
                        else title_value
                    )
                    update_program = selected_program_value or existing_row["selected_program_id"]
                    update_files = selected_files_json if selected_file_ids is not None else existing_row["selected_file_ids_json"]
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET title = %s,
                            selected_program_id = %s,
                            selected_file_ids_json = %s::jsonb,
                            updated_at = NOW(),
                            archived_at = NULL
                        WHERE scoped_session_id = %s
                        """,
                        (update_title, update_program, update_files, scoped_session_id),
                    )

                cur.execute(
                    "SELECT scoped_session_id, session_id, user_id, title, selected_program_id, selected_file_ids_json::text, created_at, updated_at, archived_at FROM chat_sessions WHERE scoped_session_id = %s",
                    (scoped_session_id,),
                )
                row = cur.fetchone()
                data = {
                    "scoped_session_id": row[0],
                    "session_id": row[1],
                    "user_id": row[2],
                    "title": row[3],
                    "selected_program_id": row[4],
                    "selected_file_ids_json": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "archived_at": row[8],
                }
                return self._serialize_chat_session(data)
        finally:
            conn.close()

    def list_chat_sessions(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        if not uid:
            return []
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT scoped_session_id, session_id, user_id, title, selected_program_id, selected_file_ids_json::text, created_at, updated_at, archived_at
                    FROM chat_sessions
                    WHERE user_id = %s
                """
                params: List[Any] = [uid]
                if not include_archived:
                    query += " AND archived_at IS NULL"
                query += " ORDER BY updated_at DESC, created_at DESC"
                cur.execute(query, params)
                rows = cur.fetchall()
                sessions = []
                for row in rows:
                    sessions.append(
                        self._serialize_chat_session(
                            {
                                "scoped_session_id": row[0],
                                "session_id": row[1],
                                "user_id": row[2],
                                "title": row[3],
                                "selected_program_id": row[4],
                                "selected_file_ids_json": row[5],
                                "created_at": row[6],
                                "updated_at": row[7],
                                "archived_at": row[8],
                            }
                        )
                    )
                return sessions
        finally:
            conn.close()

    def update_chat_session(
        self,
        session_id: str,
        user_id: str,
        title: Optional[str] = None,
        selected_program_id: Optional[str] = None,
        selected_file_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        uid = str(user_id or "").strip()
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT scoped_session_id, session_id, user_id, title, selected_program_id, selected_file_ids_json::text, created_at, updated_at, archived_at
                    FROM chat_sessions WHERE scoped_session_id = %s AND user_id = %s
                    """,
                    (scoped_session_id, uid),
                )
                existing = cur.fetchone()
                if existing is None:
                    return None
                existing_row = {
                    "title": existing[3],
                    "selected_program_id": existing[4],
                    "selected_file_ids_json": existing[5],
                }
                title_value = str(title).strip() if title is not None else existing_row["title"]
                program_value = (
                    str(selected_program_id).strip()
                    if selected_program_id is not None and str(selected_program_id).strip()
                    else (None if selected_program_id is not None else existing_row["selected_program_id"])
                )
                files_json = (
                    self._json_list(selected_file_ids)
                    if selected_file_ids is not None
                    else existing_row["selected_file_ids_json"]
                )
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET title = %s, selected_program_id = %s, selected_file_ids_json = %s::jsonb, updated_at = NOW()
                    WHERE scoped_session_id = %s AND user_id = %s
                    """,
                    (title_value or "Phiên mới", program_value, files_json, scoped_session_id, uid),
                )
                cur.execute(
                    """
                    SELECT scoped_session_id, session_id, user_id, title, selected_program_id, selected_file_ids_json::text, created_at, updated_at, archived_at
                    FROM chat_sessions WHERE scoped_session_id = %s AND user_id = %s
                    """,
                    (scoped_session_id, uid),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._serialize_chat_session(
                    {
                        "scoped_session_id": row[0],
                        "session_id": row[1],
                        "user_id": row[2],
                        "title": row[3],
                        "selected_program_id": row[4],
                        "selected_file_ids_json": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                        "archived_at": row[8],
                    }
                )
        finally:
            conn.close()

    def archive_chat_session(self, session_id: str, user_id: str) -> bool:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        uid = str(user_id or "").strip()
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET archived_at = NOW(), updated_at = NOW()
                    WHERE scoped_session_id = %s AND user_id = %s AND archived_at IS NULL
                    """,
                    (scoped_session_id, uid),
                )
                return cur.rowcount > 0
        finally:
            conn.close()

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
        uid = str(user_id or "").strip() or None
        role_value = str(role or "").strip().lower()
        if role_value not in {"user", "assistant", "system"}:
            role_value = "assistant"
        content_value = str(content or "").strip()
        if not content_value:
            return None
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages(scoped_session_id, session_id, user_id, role, content, citations_json)
                    VALUES(%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (scoped_session_id, sid, uid, role_value, content_value, json.dumps(citations or [], ensure_ascii=False)),
                )
                row = cur.fetchone()
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE scoped_session_id = %s",
                    (scoped_session_id,),
                )
                return int(row[0]) if row else None
        finally:
            conn.close()

    def get_chat_messages(self, session_id: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        uid = str(user_id or "").strip()
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role, content, citations_json::text, created_at
                    FROM chat_messages
                    WHERE scoped_session_id = %s AND user_id = %s
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s
                    """,
                    (scoped_session_id, uid, max(1, min(int(limit or 50), 200))),
                )
                rows = cur.fetchall()
                messages: List[Dict[str, Any]] = []
                for row in rows:
                    try:
                        citations = json.loads(row[3] or "[]")
                    except json.JSONDecodeError:
                        citations = []
                    messages.append(
                        {
                            "id": int(row[0]),
                            "role": str(row[1] or ""),
                            "content": str(row[2] or ""),
                            "citations": citations if isinstance(citations, list) else [],
                            "created_at": str(row[4] or ""),
                        }
                    )
                return messages
        finally:
            conn.close()

    def import_chat_session(
        self,
        session_id: str,
        user_id: str,
        title: Optional[str] = None,
        selected_program_id: Optional[str] = None,
        selected_file_ids: Optional[List[str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return {"session_id": sid, "status": "skipped", "imported_messages": 0}

        clean_messages: List[Dict[str, Any]] = []
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("type") or "").strip().lower()
            if role == "bot":
                role = "assistant"
            if role not in {"user", "assistant", "system"}:
                role = "assistant"
            content = str(item.get("content") or item.get("text") or "").strip()
            if not content:
                continue
            citations = item.get("citations") if isinstance(item.get("citations"), list) else []
            clean_messages.append({"role": role, "content": content, "citations": citations})

        session = self.ensure_chat_session(
            session_id=sid,
            user_id=uid,
            title=title or (clean_messages[0]["content"] if clean_messages else "Phiên cũ"),
            selected_program_id=selected_program_id,
            selected_file_ids=selected_file_ids,
        )

        scoped_session_id = self._scoped_session_id(session_id=sid, user_id=uid)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE scoped_session_id = %s AND user_id = %s",
                    (scoped_session_id, uid),
                )
                existing_count = int(cur.fetchone()[0] or 0)
                if existing_count:
                    return {
                        "session_id": sid,
                        "status": "exists",
                        "imported_messages": 0,
                        "session": session,
                    }

                imported = 0
                for item in clean_messages[:500]:
                    cur.execute(
                        """
                        INSERT INTO chat_messages(scoped_session_id, session_id, user_id, role, content, citations_json)
                        VALUES(%s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            scoped_session_id,
                            sid,
                            uid,
                            item["role"],
                            item["content"],
                            json.dumps(item["citations"], ensure_ascii=False),
                        ),
                    )
                    imported += 1
                if imported:
                    cur.execute(
                        "UPDATE chat_sessions SET updated_at = NOW() WHERE scoped_session_id = %s",
                        (scoped_session_id,),
                    )
                return {
                    "session_id": sid,
                    "status": "imported" if imported else "metadata_only",
                    "imported_messages": imported,
                    "session": self.ensure_chat_session(
                        session_id=sid,
                        user_id=uid,
                        title=title,
                        selected_program_id=selected_program_id,
                        selected_file_ids=selected_file_ids,
                    ),
                }
        finally:
            conn.close()

    def migrate_legacy_history_to_chat_sessions(
        self,
        user_id: str,
        limit_sessions: int = 50,
        max_pairs_per_session: int = 50,
    ) -> int:
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
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id,
                           MIN(timestamp) AS created_at,
                           MAX(timestamp) AS updated_at,
                           COUNT(*) AS row_count
                    FROM history
                    WHERE session_id IS NOT NULL
                      AND BTRIM(session_id) != ''
                      AND session_id NOT LIKE 'user:%:session:%'
                    GROUP BY session_id
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                legacy_sessions = cur.fetchall()

                for legacy in legacy_sessions:
                    sid = str(legacy[0] or "").strip()
                    if not sid:
                        continue
                    cur.execute("SELECT user_id FROM legacy_chat_migrations WHERE session_id = %s", (sid,))
                    claim = cur.fetchone()
                    if claim is not None and str(claim[0] or "").strip() != uid:
                        continue
                    scoped_session_id = self._scoped_session_id(session_id=sid, user_id=uid)
                    cur.execute("SELECT 1 FROM chat_sessions WHERE scoped_session_id = %s", (scoped_session_id,))
                    existing = cur.fetchone()
                    if existing is not None:
                        continue

                    cur.execute(
                        """
                        SELECT query FROM history
                        WHERE session_id = %s
                        ORDER BY timestamp ASC, id ASC
                        LIMIT 1
                        """,
                        (sid,),
                    )
                    first_row = cur.fetchone()

                    selected_program_id = None
                    selected_file_ids: List[str] = []
                    cur.execute("SELECT state_json::text FROM conversation_state WHERE session_id = %s", (sid,))
                    state_row = cur.fetchone()
                    if state_row is not None:
                        try:
                            state = json.loads(state_row[0] or "{}")
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

                    cur.execute(
                        """
                        INSERT INTO chat_sessions(
                            scoped_session_id, session_id, user_id, title,
                            selected_program_id, selected_file_ids_json,
                            created_at, updated_at, archived_at
                        )
                        VALUES(%s, %s, %s, %s, %s, %s::jsonb, %s, %s, NULL)
                        """,
                        (
                            scoped_session_id,
                            sid,
                            uid,
                            self._legacy_chat_title(first_row[0] if first_row else ""),
                            str(selected_program_id or "").strip() or None,
                            self._json_list(selected_file_ids),
                            legacy[1] or None,
                            legacy[2] or None,
                        ),
                    )

                    cur.execute(
                        """
                        SELECT query, response, timestamp FROM history
                        WHERE session_id = %s
                        ORDER BY timestamp ASC, id ASC
                        LIMIT %s
                        """,
                        (sid, max_pairs),
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        query_text = str(row[0] or "").strip()
                        response_text = str(row[1] or "").strip()
                        created_at = row[2] or None
                        if query_text:
                            cur.execute(
                                """
                                INSERT INTO chat_messages(
                                    scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                                )
                                VALUES(%s, %s, %s, 'user', %s, '[]'::jsonb, %s)
                                """,
                                (scoped_session_id, sid, uid, query_text, created_at),
                            )
                        if response_text:
                            cur.execute(
                                """
                                INSERT INTO chat_messages(
                                    scoped_session_id, session_id, user_id, role, content, citations_json, created_at
                                )
                                VALUES(%s, %s, %s, 'assistant', %s, '[]'::jsonb, %s)
                                """,
                                (scoped_session_id, sid, uid, response_text, created_at),
                            )
                    cur.execute(
                        """
                        INSERT INTO legacy_chat_migrations(session_id, user_id, migrated_at)
                        VALUES(%s, %s, NOW())
                        ON CONFLICT(session_id) DO NOTHING
                        """,
                        (sid, uid),
                    )
                    migrated += 1
                if migrated:
                    logger.info("Migrated %s legacy history sessions into chat_sessions for user %s", migrated, uid)
                return migrated
        finally:
            conn.close()

    def save_summary(self, file_id: str, summary: str):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO file_summaries(file_id, summary, created_at)
                    VALUES(%s, %s, NOW())
                    ON CONFLICT(file_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        created_at = NOW()
                    """,
                    (file_id, summary),
                )
        finally:
            conn.close()

    def get_summary(self, file_id: str) -> Optional[str]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT summary FROM file_summaries WHERE file_id = %s", (file_id,))
                row = cur.fetchone()
                return str(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()

    def get_structured_state(self, session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
        base = default_conversation_state()
        scoped_session_id = self._scoped_session_id(session_id=session_id, user_id=user_id)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT state_json::text FROM conversation_state WHERE session_id = %s", (scoped_session_id,))
                row = cur.fetchone()
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
        except Exception as e:
            logger.error("Loi khi lay structured state cho session %s: %s", scoped_session_id, e)
            return base
        finally:
            conn.close()

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

        payload = json.dumps(base, ensure_ascii=False)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_state(session_id, state_json, updated_at)
                    VALUES(%s, %s::jsonb, NOW())
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = EXCLUDED.state_json,
                        updated_at = NOW()
                    """,
                    (scoped_session_id, payload),
                )
            return base
        except Exception as e:
            logger.error("Loi khi luu structured state cho session %s: %s", scoped_session_id, e)
            return self.get_structured_state(session_id=session_id, user_id=user_id)
        finally:
            conn.close()
