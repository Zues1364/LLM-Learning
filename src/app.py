import json
import logging
import os
import re
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import Lock, Event, Thread
from typing import Any, Dict, List, Set, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, Body, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import (
    AnswerGeneratorAgent,
    get_elective_interest_agent,
    get_mcp_planner_agent,
    get_rag_agent,
)
from env_loader import load_env

# Initial Env Load
load_env()

from mcp_client.client import MCPClient
from conversation_state import (
    default_conversation_state,
    resolve_query_with_state,
    update_state_after_turn,
)
from persistent_memory import PersistentMemory
from utils import process_pdf, normalize_for_match  # Backward-compatible import for tests that monkeypatch app.process_pdf
# resource_loader import NOT needed here if we delegate to scan_resources via MCP? 
# Use resource_loader for 'get_resources' list only? 
# Or just duplicate the listing logic to avoid dependency issues if separate process?
# They likely share the 'data' dir.
from resource_loader import resource_loader 
from mail_agent import MailOAuthRefreshError, mail_agent_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS origins for credentialed requests (cookies). Do not use wildcard with allow_credentials=True.
_cors_origins_env = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:9000,http://localhost:9000",
)
_cors_origins = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
RESOURCE_PDF_DIR = BASE_DIR / "data" / "resources" / "pdfs"
RESOURCE_HTML_DIR = BASE_DIR / "data" / "resources" / "html"
SESSION_CACHE_DIR = BASE_DIR / "data" / "session_cache"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(RESOURCE_PDF_DIR, exist_ok=True)
os.makedirs(RESOURCE_HTML_DIR, exist_ok=True)
os.makedirs(SESSION_CACHE_DIR, exist_ok=True)

# Globals
memory = PersistentMemory(db_path=str(BASE_DIR / "data" / "memory.db"), max_history=25)
loaded_file_ids: Set[str] = set()
file_meta: Dict[str, str] = {}  # file_id -> original filename
last_uploaded_file_ids: List[str] = []
rag_agent = None
mcp_client = MCPClient()

answer_agent = AnswerGeneratorAgent(get_rag_agent())
_elective_interest_agent = None
_session_locks: Dict[str, Lock] = {}
_session_locks_guard = Lock()
_mail_poller_stop = Event()
_mail_poller_thread: Optional[Thread] = None
STRUCTURED_TKB_ENABLED = os.getenv("STRUCTURED_TKB_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _read_bool_env(env_name: str, default: bool = False) -> bool:
    raw = str(os.getenv(env_name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


APP_COOKIE_SECURE = _read_bool_env("APP_COOKIE_SECURE", False)
APP_COOKIE_SAMESITE = str(os.getenv("APP_COOKIE_SAMESITE", "lax") or "lax").strip().lower()
if APP_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    logger.warning("Invalid APP_COOKIE_SAMESITE=%r, falling back to 'lax'.", APP_COOKIE_SAMESITE)
    APP_COOKIE_SAMESITE = "lax"


def _read_int_env(env_name: str, default: int) -> int:
    raw = str(os.getenv(env_name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return value if value > 0 else default


MAX_TRANSCRIPT_UPLOAD_BYTES = _read_int_env("MAX_TRANSCRIPT_UPLOAD_MB", 50) * 1024 * 1024
MAX_RESOURCE_UPLOAD_BYTES = _read_int_env("MAX_RESOURCE_UPLOAD_MB", 100) * 1024 * 1024
MAX_BATCH_UPLOAD_FILES = _read_int_env("MAX_BATCH_UPLOAD_FILES", 25)


def _read_timeout_env(env_name: str, default_seconds: float) -> Optional[float]:
    raw = str(os.getenv(env_name, str(default_seconds)) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = float(default_seconds)
    # Non-positive values mean "no timeout" for callers that can tolerate long-running work.
    return value if value > 0 else None


MCP_TOOL_TIMEOUTS: Dict[str, Optional[float]] = {
    # consult_advisor can exceed 2 minutes on cold transcript parses; keep default lenient.
    "consult_advisor": _read_timeout_env("ASK_CONSULT_ADVISOR_TIMEOUT_SEC", 300.0),
    "retrieve_chunks": _read_timeout_env("ASK_RETRIEVE_CHUNKS_TIMEOUT_SEC", 45.0),
}
MCP_TOOL_TIMEOUTS_TRANSCRIPT: Dict[str, Optional[float]] = {
    # Transcript-intensive advisory queries may need substantially longer.
    "consult_advisor": _read_timeout_env("ASK_CONSULT_ADVISOR_TIMEOUT_SEC_TRANSCRIPT", 900.0),
}


def _invoke_mcp_tool(tool: str, args: Dict[str, Any], timeout_seconds: Optional[float] = None) -> Any:
    if timeout_seconds is None:
        return mcp_client.invoke(tool, args)
    try:
        return mcp_client.invoke(tool, args, timeout=timeout_seconds)
    except TypeError:
        # Backward-compatible with monkeypatched test doubles / legacy client signature.
        return mcp_client.invoke(tool, args)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Dict[str, Any]:
    checks: Dict[str, str] = {}
    try:
        db_path = Path(memory.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("SELECT 1")
        checks["memory_db"] = "ok"
    except Exception as exc:
        checks["memory_db"] = str(exc)
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})

    for path_name, path in {
        "pdf_dir": PDF_DIR,
        "resource_pdf_dir": RESOURCE_PDF_DIR,
        "resource_html_dir": RESOURCE_HTML_DIR,
        "session_cache_dir": SESSION_CACHE_DIR,
    }.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            checks[path_name] = "ok"
        except Exception as exc:
            checks[path_name] = str(exc)
            raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})

    return {"status": "ready", "checks": checks}


def _session_dir(session_id: str) -> Path:
    return SESSION_CACHE_DIR / _normalize_session_id(session_id)


def _normalize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id or "user_session_1").strip())
    return cleaned or "user_session_1"


def _mail_cookie_name() -> str:
    return str(getattr(mail_agent_service, "app_session_cookie_name", "app_session") or "app_session")


def _invoke_mail_service(fn, *args, owner_ctx: Optional[Dict[str, Any]] = None, **kwargs):
    if owner_ctx is not None:
        try:
            return fn(*args, owner_ctx=owner_ctx, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'owner_ctx'" not in str(exc):
                raise
    return fn(*args, **kwargs)


def _current_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    cookie_name = _mail_cookie_name()
    raw_token = request.cookies.get(cookie_name)
    resolver = getattr(mail_agent_service, "get_authenticated_user", None)
    if not callable(resolver):
        return None
    try:
        return resolver(raw_token)
    except Exception as exc:
        logger.warning("Auth session lookup failed: %s", exc)
        return None


def _current_user_id_from_request(request: Request) -> Optional[str]:
    user = _current_user_from_request(request)
    if not user:
        return None
    user_id = str(user.get("id") or "").strip()
    return user_id or None


def _with_memory_owner(args: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    payload = dict(args or {})
    if user_id:
        payload["user_id"] = user_id
    return payload


def _require_authenticated_user_id(request: Request) -> str:
    user_id = _current_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Bạn cần đăng nhập Google để quản lý phiên chat theo tài khoản.")
    return user_id


def _derive_chat_title(query: str, fallback: str = "Phiên mới") -> str:
    text = " ".join(str(query or "").split())
    if not text:
        return fallback
    if len(text) <= 80:
        return text
    return text[:77].rstrip() + "..."


def _resolve_mail_owner(request: Request, session_id: Optional[str]) -> Dict[str, Any]:
    user = _current_user_from_request(request)
    normalized_session = _normalize_session_id(session_id or "user_session_1")
    resolver = getattr(mail_agent_service, "resolve_owner_context", None)
    if not callable(resolver):
        if user:
            return {
                "owner_type": "user",
                "user_id": str(user.get("id") or ""),
                "session_id": normalized_session,
            }
        return {"owner_type": "session", "session_id": normalized_session}
    if user:
        return resolver(
            session_id=normalized_session,
            user_id=str(user.get("id") or ""),
        )
    return resolver(session_id=normalized_session)


def _scan_resources_with_owner(session_id: Optional[str], user_id: Optional[str] = None):
    payload: Dict[str, Any] = {}
    if user_id:
        payload["user_id"] = user_id
    elif session_id:
        payload["session_id"] = session_id
    try:
        mcp_client.invoke("scan_resources", payload)
    except Exception as first_err:
        if user_id and session_id:
            logger.warning("scan_resources(user) failed, fallback to session scope: %s", first_err)
            mcp_client.invoke("scan_resources", {"session_id": session_id})
        else:
            raise

def _session_meta_path(session_id: str) -> Path:
    return _session_dir(session_id) / "meta.json"

def _normalize_file_ids(file_ids: List[str] | None) -> List[str]:
    return list(dict.fromkeys([f for f in (file_ids or []) if f]))

def _load_session_meta(session_id: str) -> Dict[str, Any]:
    default_meta: Dict[str, Any] = {"file_ids": [], "program_id": None}
    meta_path = _session_meta_path(session_id)
    if not meta_path.exists():
        return default_meta
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default_meta
        ids = data.get("file_ids", [])
        program_id = data.get("program_id")
        normalized_ids = ids if isinstance(ids, list) else []
        normalized_program = str(program_id).strip() if program_id else None
        return {
            "file_ids": _normalize_file_ids(normalized_ids),
            "program_id": normalized_program or None,
        }
    except Exception:
        return default_meta

def _write_session_meta(session_id: str, data: Dict[str, Any]):
    try:
        dir_path = _session_dir(session_id)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta_path = _session_meta_path(session_id)
        payload = {
            "file_ids": _normalize_file_ids(data.get("file_ids", [])),
            "program_id": (str(data.get("program_id")).strip() if data.get("program_id") else None),
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Khong luu duoc session meta cho %s: %s", session_id, e)

def _load_session_files(session_id: str) -> List[str]:
    return _load_session_meta(session_id).get("file_ids", [])

def _save_session_files(session_id: str, file_ids: List[str]):
    meta = _load_session_meta(session_id)
    meta["file_ids"] = _normalize_file_ids(file_ids)
    _write_session_meta(session_id, meta)

def _load_session_program(session_id: str) -> Optional[str]:
    return _load_session_meta(session_id).get("program_id")

def _save_session_program(session_id: str, program_id: Optional[str]):
    meta = _load_session_meta(session_id)
    meta["program_id"] = str(program_id).strip() if program_id else None
    _write_session_meta(session_id, meta)


def _get_session_lock(session_id: str) -> Lock:
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = Lock()
            _session_locks[session_id] = lock
        return lock


def _mail_poller_loop():
    while not _mail_poller_stop.is_set():
        try:
            result = mail_agent_service.poll_all_connected_users()
            polled = int(result.get("polled") or 0)
            if polled > 0:
                logger.info("[mail_poller] polled %s connected user(s)", polled)
        except Exception as e:
            logger.warning("[mail_poller] polling loop error: %s", e)
        _mail_poller_stop.wait(timeout=max(60, mail_agent_service.poll_minutes * 60))


@app.on_event("startup")
def _start_mail_poller():
    global _mail_poller_thread
    _mail_poller_stop.clear()
    _mail_poller_thread = Thread(target=_mail_poller_loop, name="mail-poller", daemon=True)
    _mail_poller_thread.start()
    logger.info("[mail_poller] started with interval=%s minutes", mail_agent_service.poll_minutes)


@app.on_event("shutdown")
def _stop_mail_poller():
    _mail_poller_stop.set()
    thread = _mail_poller_thread
    if thread and thread.is_alive():
        thread.join(timeout=2.0)


def _strip_markdown_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _escape_control_chars_in_json_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escaped = False
    return "".join(out)


def _extract_planner_field_text(text: str, key: str, next_keys: List[str]) -> Optional[str]:
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text, flags=re.IGNORECASE)
    if not key_match:
        return None

    start = key_match.end()
    candidates: List[int] = []
    for next_key in next_keys:
        m = re.search(rf'"\s*,?\s*"{re.escape(next_key)}"\s*:', text[start:], flags=re.IGNORECASE)
        if m:
            candidates.append(start + m.start())
    end = min(candidates) if candidates else len(text)
    value = text[start:end]
    value = re.sub(r'"\s*,?\s*$', "", value).strip()
    return value if value else None


def _heuristic_parse_planner_output(text: str) -> Optional[Dict[str, Any]]:
    source_match = re.search(r'"source"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if not source_match:
        return None

    source = (source_match.group(1) or "").strip()
    if not source:
        return None

    context = _extract_planner_field_text(
        text,
        "context",
        ["memory", "chunk_index", "requires_selection", "source"],
    ) or text
    memory = _extract_planner_field_text(
        text,
        "memory",
        ["chunk_index", "requires_selection", "source", "context"],
    ) or ""

    chunk_index: Any = None
    chunk_match = re.search(
        r'"chunk_index"\s*:\s*(null|-?\d+|"null")',
        text,
        flags=re.IGNORECASE,
    )
    if chunk_match:
        raw_chunk = (chunk_match.group(1) or "").strip().lower().strip('"')
        if raw_chunk not in {"", "null"}:
            try:
                chunk_index = int(raw_chunk)
            except ValueError:
                chunk_index = None

    requires_selection = bool(
        re.search(r'"requires_selection"\s*:\s*true', text, flags=re.IGNORECASE)
    )

    return {
        "source": source,
        "context": context,
        "memory": memory,
        "chunk_index": chunk_index,
        "requires_selection": requires_selection,
    }


def _parse_planner_output(raw_output: Any) -> Optional[Dict[str, Any]]:
    text = _strip_markdown_fences(str(raw_output or ""))
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            repaired = _escape_control_chars_in_json_strings(candidate)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        heuristic = _heuristic_parse_planner_output(candidate)
        if heuristic is not None:
            return heuristic

    heuristic = _heuristic_parse_planner_output(text)
    if heuristic is not None:
        return heuristic

    return None


def _context_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _load_structured_state(session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    state = default_conversation_state()
    try:
        payload = mcp_client.invoke(
            "memory_state_get",
            _with_memory_owner({"session_id": session_id}, user_id),
        )
        if isinstance(payload, dict):
            state.update(payload)
            if isinstance(payload.get("entities"), dict):
                merged_entities = state.get("entities", {}).copy()
                merged_entities.update(payload.get("entities", {}))
                state["entities"] = merged_entities
            if isinstance(payload.get("referents"), dict):
                merged_referents = state.get("referents", {}).copy()
                merged_referents.update(payload.get("referents", {}))
                state["referents"] = merged_referents
    except Exception as e:
        logger.warning("Khong tai duoc structured state cho session %s: %s", session_id, e)
    return state


def _save_structured_state(
    session_id: str,
    prev_state: Dict[str, Any],
    raw_query: str,
    resolved_query: str,
    answer: str,
    planner_source: str,
    planner_context: str,
    selected_program_id: Optional[str],
    user_id: Optional[str] = None,
) -> None:
    try:
        next_state = update_state_after_turn(
            previous_state=prev_state,
            raw_query=raw_query,
            resolved_query=resolved_query,
            answer=answer,
            planner_source=planner_source,
            planner_context=planner_context,
            selected_program_id=selected_program_id,
        )
        mcp_client.invoke(
            "memory_state_upsert",
            _with_memory_owner(
                {
                    "session_id": session_id,
                    "state": next_state,
                },
                user_id,
            ),
        )
    except Exception as e:
        logger.warning("Khong luu duoc structured state cho session %s: %s", session_id, e)


_RETRIEVE_CITATION_LINE_RE = re.compile(
    r"^\[(?P<source>.+?)\s*-\s*Chunk\s*(?P<chunk>\d+)(?:\s*-\s*Page\s*(?P<page>\d+))?(?:\s*-\s*Line\s*(?P<line>\d+))?\]\s*(?P<body>.*)$",
    flags=re.IGNORECASE,
)


_CITATION_FOCUS_STOPWORDS: Set[str] = {
    "la",
    "va",
    "cua",
    "cho",
    "voi",
    "trong",
    "theo",
    "nay",
    "kia",
    "do",
    "den",
    "tu",
    "mot",
    "cac",
    "nhung",
    "duoc",
    "se",
    "da",
    "co",
    "khong",
    "neu",
    "thi",
    "ban",
    "toi",
    "chung",
    "em",
    "anh",
    "chi",
    "rang",
    "nhu",
    "de",
    "o",
    "tai",
    "ve",
    "hay",
    "roi",
    "van",
    "tren",
    "duoi",
    "sau",
    "truoc",
    "giua",
    "cau",
    "hoi",
    "tra",
    "loi",
}

_CITATION_FORCE_KEYWORDS: Tuple[str, ...] = (
    "ielts",
    "toeic",
    "toefl",
    "vstep",
    "aptis",
    "jlpt",
    "nat-test",
    "j-test",
    "ca",
    "tiet",
    "phong",
    "gio",
    "bat dau",
    "ket thuc",
)

_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{3,4}[A-Z]?\b")


def _extract_focus_tokens(query: str, answer: str) -> Set[str]:
    seed = f"{query or ''} {answer or ''}".strip()
    norm = normalize_for_match(seed)
    if not norm:
        return set()
    tokens = re.findall(r"[a-z0-9][a-z0-9.\-]{1,}", norm)
    filtered: Set[str] = set()
    for token in tokens:
        tok = str(token or "").strip()
        if not tok:
            continue
        if tok in _CITATION_FOCUS_STOPWORDS:
            continue
        if tok.isdigit() and len(tok) <= 2:
            continue
        filtered.add(tok)
    return filtered


def _extract_focus_codes(query: str, answer: str) -> Set[str]:
    text = f"{query or ''} {answer or ''}".upper()
    return {str(match).strip() for match in _COURSE_CODE_RE.findall(text) if str(match).strip()}


def _extract_force_keywords(text: str) -> Set[str]:
    norm = normalize_for_match(text or "")
    if not norm:
        return set()
    hits: Set[str] = set()
    for key in _CITATION_FORCE_KEYWORDS:
        if key in norm:
            hits.add(key)
    return hits


def _score_citation_excerpt(
    excerpt: str,
    *,
    focus_tokens: Set[str],
    focus_codes: Set[str],
    force_keywords: Set[str],
) -> float:
    norm_excerpt = normalize_for_match(excerpt or "")
    if not norm_excerpt:
        return 0.0
    score = 0.0

    if focus_tokens:
        token_hits = sum(1 for token in focus_tokens if token and token in norm_excerpt)
        score += min(6.0, token_hits * 0.8)

    if focus_codes:
        excerpt_upper = str(excerpt or "").upper()
        code_hits = sum(1 for code in focus_codes if code and code in excerpt_upper)
        score += code_hits * 3.0

    if force_keywords:
        force_hits = sum(1 for key in force_keywords if key in norm_excerpt)
        if force_hits > 0:
            score += force_hits * 4.0
        else:
            # Penalize generic chunks that miss high-signal query keywords
            score -= 3.0

    return score


def _compact_citation_excerpt(
    excerpt: str,
    *,
    focus_tokens: Set[str],
    focus_codes: Set[str],
    force_keywords: Set[str],
    max_chars: int = 900,
) -> str:
    raw_excerpt = str(excerpt or "").strip()
    if not raw_excerpt:
        return ""

    lines = [ln.strip() for ln in raw_excerpt.splitlines() if ln.strip()]
    if not lines:
        return raw_excerpt[:max_chars]
    if len(lines) <= 4:
        return "\n".join(lines)[:max_chars]

    table_focus_keywords = {
        "ielts",
        "toeic",
        "toefl",
        "aptis",
        "cambridge",
        "vstep",
        "ngoai ngu",
        "knlnvn",
    }
    def _is_tableish_line(value: str) -> bool:
        stripped = str(value or "").strip()
        if not stripped:
            return False
        if stripped.startswith("|"):
            return True
        # Some OCR/markdown conversions may drop leading/trailing pipes
        # but still keep multiple cell separators.
        return stripped.count("|") >= 2

    if force_keywords & table_focus_keywords:
        best_table_excerpt: Optional[str] = None
        best_table_score: Tuple[int, int] = (-1, -1)
        for idx, line in enumerate(lines):
            if not _is_tableish_line(line):
                continue
            table_start = idx
            while table_start - 1 >= 0 and _is_tableish_line(lines[table_start - 1]):
                table_start -= 1
            table_end = idx
            while table_end + 1 < len(lines):
                next_idx = table_end + 1
                next_line = lines[next_idx]
                if _is_tableish_line(next_line):
                    table_end = next_idx
                    continue

                # Some table cells may be OCR'd into a standalone continuation line.
                # Keep a single bridge line if the following line returns to table format.
                if next_idx + 1 < len(lines) and _is_tableish_line(lines[next_idx + 1]):
                    continuation = next_line.strip()
                    if continuation and not continuation.startswith(("#", "[", "Page ")):
                        table_end = next_idx
                        continue
                break

            nearby_start = max(0, table_start - 2)
            nearby_text = "\n".join(lines[nearby_start : table_end + 1])
            nearby_norm = normalize_for_match(nearby_text)
            if not any(keyword in nearby_norm for keyword in table_focus_keywords):
                continue

            selected: List[str] = []
            if table_start - 1 >= 0 and lines[table_start - 1].lstrip("#").strip():
                selected.append(lines[table_start - 1])
            selected.extend(lines[table_start : table_end + 1])
            tableish_count = sum(1 for ln in selected if _is_tableish_line(ln))
            if tableish_count <= 2:
                # Header-only extraction is usually not useful for equivalency tables.
                continue
            candidate = "\n".join(selected).strip()
            candidate_score = (tableish_count, len(candidate))
            if candidate_score > best_table_score:
                best_table_score = candidate_score
                best_table_excerpt = candidate
        if best_table_excerpt:
            return best_table_excerpt[:max_chars]
        # Fallback: keep raw excerpt when table layout is malformed instead of
        # over-compacting to header/separator only.
        return raw_excerpt[:max_chars]

    scored_lines: List[Tuple[float, int]] = []
    for idx, line in enumerate(lines):
        line_score = _score_citation_excerpt(
            line,
            focus_tokens=focus_tokens,
            focus_codes=focus_codes,
            force_keywords=force_keywords,
        )
        scored_lines.append((line_score, idx))

    scored_lines.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_idx = scored_lines[0]
    if best_score <= 0:
        # Keep the opening part if we cannot identify a clearly relevant line
        return "\n".join(lines[:4])[:max_chars]

    selected_idx: Set[int] = {best_idx}
    if best_idx - 1 >= 0:
        selected_idx.add(best_idx - 1)
    if best_idx + 1 < len(lines):
        selected_idx.add(best_idx + 1)

    # Add another high-scoring line if available to enrich context
    for score, idx in scored_lines[1:]:
        if score <= 0:
            break
        selected_idx.add(idx)
        break

    ordered = [lines[i] for i in sorted(selected_idx)]
    compact = "\n".join(ordered).strip()
    return compact[:max_chars]


def _extract_retrieve_citations(
    context: Any,
    max_items: int = 8,
    *,
    query: str = "",
    answer: str = "",
) -> List[Dict[str, Any]]:
    """
    Parse retrieve context lines formatted as:
    [<file_name> - Chunk <index>] <chunk text>
    """
    text = _context_to_text(context)
    if not text.strip():
        return []

    def _to_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(str(value).strip())
        except Exception:
            return None
        if number <= 0:
            return None
        return number

    parsed: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in text.splitlines():
        line = str(raw_line or "").rstrip()
        if not line.strip():
            continue
        match = _RETRIEVE_CITATION_LINE_RE.match(line.strip())
        if match:
            if current:
                parsed.append(current)
            chunk_raw = str(match.group("chunk") or "").strip()
            chunk_index = _to_positive_int(chunk_raw)
            page = _to_positive_int(match.group("page"))
            source_line = _to_positive_int(match.group("line"))
            current = {
                "source_file": str(match.group("source") or "").strip(),
                "chunk_index": chunk_index,
                "page": page,
                "source_line": source_line,
                "excerpt": str(match.group("body") or "").strip(),
            }
            continue
        if current is not None:
            extra = line.strip()
            if extra:
                current["excerpt"] = f"{current.get('excerpt', '')}\n{extra}".strip()

    if current:
        parsed.append(current)

    focus_tokens = _extract_focus_tokens(query=query, answer=answer)
    focus_codes = _extract_focus_codes(query=query, answer=answer)
    force_keywords = _extract_force_keywords(f"{query or ''} {answer or ''}")

    deduped_scored: List[Tuple[float, int, Dict[str, Any]]] = []
    seen: Set[Tuple[str, Optional[int], Optional[int], Optional[int], str]] = set()
    for order, item in enumerate(parsed):
        source_file = str(item.get("source_file") or "").strip()
        chunk_index = item.get("chunk_index")
        page = item.get("page")
        source_line = item.get("source_line")
        raw_excerpt = str(item.get("excerpt") or "").strip()
        if not raw_excerpt:
            continue
        compact_excerpt = _compact_citation_excerpt(
            raw_excerpt,
            focus_tokens=focus_tokens,
            focus_codes=focus_codes,
            force_keywords=force_keywords,
            max_chars=900,
        )
        key = (source_file, chunk_index, page, source_line, compact_excerpt[:160])
        if key in seen:
            continue
        seen.add(key)
        score = _score_citation_excerpt(
            compact_excerpt,
            focus_tokens=focus_tokens,
            focus_codes=focus_codes,
            force_keywords=force_keywords,
        )
        deduped_scored.append(
            (
                score,
                order,
                {
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "page": page,
                    "source_line": source_line,
                    "excerpt": compact_excerpt[:1600],
                },
            )
        )

    if focus_tokens or focus_codes or force_keywords:
        deduped_scored.sort(key=lambda item: (-item[0], item[1]))
    else:
        deduped_scored.sort(key=lambda item: item[1])

    deduped = [item for _, _, item in deduped_scored[:max_items]]
    for idx, item in enumerate(deduped, start=1):
        item["id"] = idx
    return deduped


def _extract_structured_schedule_citations(context: Any, max_items: int = 10) -> List[Dict[str, Any]]:
    payload = _safe_json_loads(context)
    if not isinstance(payload, dict):
        return []

    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    source_files = payload.get("source_files") if isinstance(payload.get("source_files"), list) else []
    fallback_source_file = str(payload.get("source_file") or "").strip()
    coverage_note = str(payload.get("coverage_note") or "").strip()

    citations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Optional[int], Optional[int], str]] = set()

    def _to_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(str(value).strip())
        except Exception:
            return None
        if number <= 0:
            return None
        return number

    def _slot_label(slot_value: Any) -> str:
        slot = str(slot_value or "").strip()
        if not slot:
            return ""
        return f"Ca {slot}" if slot.isdigit() else slot

    for row in rows:
        if not isinstance(row, dict):
            continue
        source_file = str(row.get("source_file") or "").strip()
        if not source_file and len(source_files) == 1:
            source_file = str(source_files[0] or "").strip()
        if not source_file and fallback_source_file:
            source_file = fallback_source_file
        if not source_file:
            source_file = "Unknown schedule source"

        class_code = str(row.get("class_code") or "").strip()
        subject_code = str(row.get("subject_code") or "").strip()
        day_of_week = str(row.get("day_of_week") or "").strip()
        slot = _slot_label(row.get("slot"))
        room = str(row.get("room") or "").strip()
        teacher = str(row.get("teacher_name") or "").strip()
        week_note = str(row.get("week_note") or "").strip()
        source_page = _to_positive_int(row.get("source_page"))
        source_line = _to_positive_int(row.get("source_line"))

        excerpt_parts: List[str] = []
        if class_code:
            excerpt_parts.append(class_code)
        elif subject_code:
            excerpt_parts.append(subject_code)
        time_label = ", ".join([part for part in [day_of_week, slot] if part])
        if time_label:
            excerpt_parts.append(time_label)
        if room:
            excerpt_parts.append(f"phòng {room}")
        if teacher:
            excerpt_parts.append(f"GV {teacher}")
        if week_note:
            excerpt_parts.append(week_note)
        excerpt = " | ".join(excerpt_parts).strip() or str(row)

        dedupe_key = (source_file, source_page, source_line, excerpt[:220])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        citations.append(
            {
                "source_file": source_file,
                "chunk_index": None,
                "page": source_page,
                "source_line": source_line,
                "excerpt": excerpt[:1600],
            }
        )
        if len(citations) >= max_items:
            break

    if not citations:
        fallback_excerpt = coverage_note or "Nguồn dữ liệu lịch học từ structured mapping."
        for src in source_files:
            source_file = str(src or "").strip()
            if not source_file:
                continue
            citations.append(
                {
                    "source_file": source_file,
                    "chunk_index": None,
                    "page": None,
                    "source_line": None,
                    "excerpt": fallback_excerpt,
                }
            )
            if len(citations) >= max_items:
                break

    for idx, item in enumerate(citations, start=1):
        item["id"] = idx
    return citations


def _extract_time_slot_citations(context: Any, max_items: int = 4) -> List[Dict[str, Any]]:
    payload = _safe_json_loads(context)
    if not isinstance(payload, dict):
        return []

    def _to_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(str(value).strip())
        except Exception:
            return None
        if number <= 0:
            return None
        return number

    source_file = str(payload.get("source_file") or "").strip() or str(payload.get("time_source_file") or "").strip()
    if not source_file:
        return []

    slot = str(payload.get("slot") or "").strip()
    period = str(payload.get("period") or "").strip()
    time_range = str(payload.get("time_range") or "").strip()
    excerpt_parts: List[str] = []
    if slot:
        excerpt_parts.append(f"Ca {slot}")
    if period:
        excerpt_parts.append(period)
    if time_range:
        excerpt_parts.append(time_range)
    excerpt = " | ".join(excerpt_parts).strip()
    if not excerpt:
        excerpt = str(payload.get("coverage_note") or "").strip() or "Nguồn dữ liệu thời gian ca học."

    citation = {
        "id": 1,
        "source_file": source_file,
        "chunk_index": None,
        "page": _to_positive_int(payload.get("source_page")),
        "source_line": _to_positive_int(payload.get("source_line")),
        "excerpt": excerpt[:1600],
    }
    return [citation][:max_items]


def _extract_semester_code_citations(context: Any, max_items: int = 4) -> List[Dict[str, Any]]:
    payload = _safe_json_loads(context)
    if not isinstance(payload, dict):
        return []

    code = str(payload.get("semester_code") or "").strip()
    source_files = payload.get("source_files") if isinstance(payload.get("source_files"), list) else []
    normalized_sources = []
    for source in source_files:
        source_name = str(source or "").strip()
        if source_name:
            normalized_sources.append(source_name)
    if not normalized_sources:
        return []

    citations: List[Dict[str, Any]] = []
    for idx, source_name in enumerate(normalized_sources[:max_items], start=1):
        excerpt = f"Mã kỳ học hiện tại: {code}" if code else "Nguồn suy luận mã kỳ học."
        citations.append(
            {
                "id": idx,
                "source_file": source_name,
                "chunk_index": None,
                "page": None,
                "source_line": None,
                "excerpt": excerpt,
            }
        )
    return citations


def _backfill_retrieve_citations_for_answer(
    *,
    query: str,
    session_id: str,
    file_ids: List[str],
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fallback citation collection:
    - Re-run retrieve_chunks for the exact user query.
    - Extract line-level citations for UI badges.

    We intentionally skip transcript-intensive queries to avoid adding
    extra retrieval hops on advisory flows that already prioritize
    consult_advisor determinism and timeout safety.
    """
    if _query_requires_transcript_files(query):
        return []
    # For non-transcript queries, do not constrain retrieval to selected transcript files.
    # Otherwise citations can point to irrelevant grade-sheet chunks instead of the
    # canonical resource (e.g. handbook/timetable).
    retrieve_file_ids: List[str] = []
    try:
        retrieve_result = _invoke_mcp_tool(
            "retrieve_chunks",
            {
                "question": query,
                "top_k": 25,
                "file_ids": retrieve_file_ids,
                "session_id": session_id,
            },
            timeout_seconds=MCP_TOOL_TIMEOUTS.get("retrieve_chunks"),
        )
    except Exception as exc:
        logger.info("[citations] backfill retrieve_chunks failed session=%s: %s", session_id, exc)
        return []

    citations = _extract_retrieve_citations(
        retrieve_result,
        max_items=max_items,
        query=query,
    )
    if citations:
        return citations

    # Defensive fallback in case retrieve payload is structured JSON-like.
    return _extract_structured_schedule_citations(retrieve_result, max_items=max_items)


def _normalize_output_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    # Recover JSON-style escaped unicode/newline payloads (e.g. "Ch\\u00e0o...\\n...")
    # but avoid touching normal UTF-8 text.
    if re.search(r"\\u[0-9a-fA-F]{4}", raw):
        try:
            decoded = raw.encode("utf-8").decode("unicode_escape")
            if decoded and "\\u" not in decoded:
                raw = decoded
        except Exception:
            pass

    raw = raw.replace(
        "AIT3004 - T tạ h o ực hành phát triển hệ thống Trí tuệ nhân",
        "AIT3004 - Thực hành phát triển hệ thống Trí tuệ nhân tạo",
    )

    def _normalize_footer_line(line: str) -> str:
        cleaned = str(line or "").strip()
        cleaned = re.sub(r"^[>\s\-\*\+`#]+", "", cleaned).strip()
        cleaned = cleaned.strip("*_`").strip()
        norm = normalize_for_match(cleaned)
        norm = re.sub(r"[^a-z0-9.\-:\s]+", " ", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def _is_source_header_line(line: str) -> bool:
        norm = _normalize_footer_line(line)
        if not norm:
            return False
        tokens = [tok for tok in norm.split(" ") if tok]
        first = tokens[0] if tokens else ""
        has_nguon_lead = first.startswith("ngu")
        has_tham = any(tok.startswith("tham") for tok in tokens)
        has_chieu = any(tok.startswith("chi") for tok in tokens)
        has_khao = any(tok.startswith("kha") for tok in tokens)
        if has_nguon_lead and ((has_tham and has_chieu) or has_khao):
            return True
        return norm in {
            "nguon",
            "nguon:",
            "nguon tham chieu",
            "nguon tham chieu:",
            "nguon tham khao",
            "nguon tham khao:",
            "nguon tham khảo",
            "nguon tham khảo:",
        } or norm.startswith("nguon tham chieu") or norm.startswith("nguon tham khao")

    def _looks_like_source_payload(line: str) -> bool:
        norm = _normalize_footer_line(line)
        if not norm:
            return False
        return any(
            token in norm
            for token in ("pdf", "html", "sheet", "page", "line", "chunk", ".xlsx", ".doc", ".ppt")
        )

    def _is_source_item_line(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return True
        if re.match(r"^(?:[-*]\s*)?\[\d+\]\s+.+$", stripped):
            return True
        return _looks_like_source_payload(stripped)

    def _trim_right_blank_lines(lines: List[str]) -> List[str]:
        kept = list(lines or [])
        while kept and not str(kept[-1] or "").strip():
            kept.pop()
        return kept

    def _strip_source_footer(value: str) -> str:
        lines = str(value or "").splitlines()
        if not lines:
            return str(value or "")

        end_idx = len(lines) - 1
        while end_idx >= 0 and not lines[end_idx].strip():
            end_idx -= 1
        if end_idx < 0:
            return ""

        # Single-line footer style: "Nguồn: <file/source>"
        tail_norm = _normalize_footer_line(lines[end_idx])
        if tail_norm.startswith("nguon:") and _looks_like_source_payload(lines[end_idx]):
            kept = lines[:end_idx]
            while kept and not kept[-1].strip():
                kept.pop()
            return "\n".join(kept).strip()

        # Fallback: remove trailing [n] source lines even if header text is OCR-broken.
        tail_idx = end_idx
        tail_source_count = 0
        while tail_idx >= 0:
            raw = str(lines[tail_idx] or "").strip()
            if not raw:
                tail_idx -= 1
                continue
            if not _is_source_item_line(raw):
                break
            tail_source_count += 1
            tail_idx -= 1
        if tail_source_count >= 2:
            before_tail_norm = _normalize_footer_line(lines[tail_idx]) if tail_idx >= 0 else ""
            before_tail_is_header = _is_source_header_line(lines[tail_idx]) if tail_idx >= 0 else False
            if (
                not before_tail_norm
                or before_tail_is_header
                or before_tail_norm.startswith("nguon")
                or "tham chieu" in before_tail_norm
                or "tham khao" in before_tail_norm
            ):
                kept = _trim_right_blank_lines(lines[: max(0, tail_idx)])
                return "\n".join(kept).rstrip()

        header_idx: Optional[int] = None
        for i in range(end_idx, -1, -1):
            if _is_source_header_line(lines[i]):
                header_idx = i
                break

        if header_idx is None:
            return str(value or "")

        trailing = lines[header_idx + 1 : end_idx + 1]
        non_empty = [ln for ln in trailing if str(ln or "").strip()]
        if not non_empty:
            return "\n".join(lines[:header_idx]).rstrip()

        citation_like_count = sum(1 for ln in non_empty if _is_source_item_line(ln))
        if citation_like_count >= max(1, int(len(non_empty) * 0.6)):
            return "\n".join(lines[:header_idx]).rstrip()

        return str(value or "")

    return _strip_source_footer(raw)


def _safe_json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": value}
    return {"raw": str(value)}


def _load_memory_context_for_session(
    session_id: str,
    max_rows: int = 10,
    user_id: Optional[str] = None,
) -> str:
    try:
        memory_result = mcp_client.invoke(
            "memory_get",
            _with_memory_owner({"session_id": session_id, "max_rows": max_rows}, user_id),
        )
        return _context_to_text(memory_result)
    except Exception as mem_err:
        logger.warning("memory_get failed for session=%s: %s", session_id, mem_err)
        return ""


def _extract_teacher_name_hint(query: str) -> Optional[str]:
    text = str(query or "").strip()
    if not text:
        return None

    generic_phrases = {
        "ai",
        "ai day",
        "co ai day",
        "nhung ai",
        "nhung ai day",
        "co nhung ai day",
        "giao vien",
        "giang vien",
        "thay nao",
        "co nao",
    }
    generic_tokens = {
        "ai",
        "nhung",
        "nao",
        "co",
        "ky",
        "ki",
        "nay",
        "la",
        "day",
        "giang",
        "vien",
        "giao",
        "mon",
        "hoc",
        "phan",
    }

    def _is_valid_teacher_candidate(candidate: str) -> bool:
        cleaned = re.sub(r"\s+", " ", str(candidate or "")).strip(" .,-")
        if not cleaned:
            return False
        norm = normalize_for_match(cleaned)
        if not norm or norm in generic_phrases:
            return False
        tokens = [tok for tok in norm.split() if tok]
        if len(tokens) < 2 or len(tokens) > 6:
            return False
        if any(tok in {"ai", "nao"} for tok in tokens):
            return False
        if all(tok in generic_tokens for tok in tokens):
            return False
        alpha_tokens = [tok for tok in tokens if re.search(r"[a-zA-ZÀ-ỹà-ỹĐđ]", tok)]
        if len(alpha_tokens) < 2:
            return False
        short_alpha = sum(1 for tok in alpha_tokens if len(tok) <= 1)
        if short_alpha > 0 and short_alpha / max(1, len(alpha_tokens)) >= 0.4:
            return False
        if any(
            token in norm
            for token in (
                "mon ",
                "hoc phan",
                "lich su",
                "kinh te",
                "chu nghia",
                "toi uu",
                "thi giac",
            )
        ):
            return False
        return True

    patterns = [
        re.compile(
            r"(?:c[oô]|th[ầa]y|giảng viên|giang vien)\s+([A-Za-zÀ-ỹà-ỹĐđ'`.\-\s]{2,80}?)(?:\s+(?:dạy|day|dạy lớp|day lop|lớp|lop|kỳ|ky|học|hoc|lịch|lich|ra sao|co nhung ai day|co ai day|nhung ai day|ai day|nao)|[?.,!]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"giảng viên\s+([A-Za-zÀ-ỹà-ỹĐđ'`.\-\s]{2,80}?)(?:\s+(?:dạy|day|lớp|lop|môn|mon)|[?.,!]|$)",
            flags=re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
        if _is_valid_teacher_candidate(candidate):
            return candidate
    return None


def _extract_subject_hint(query: str) -> Optional[str]:
    raw_text = str(query or "").strip()
    if not raw_text:
        return None
    code_match = re.search(r"\b(?:UET\.)?([A-Z]{2,4}\d{3,4}[A-Z]?)\b", raw_text, flags=re.IGNORECASE)
    if code_match:
        return code_match.group(1).upper().replace("UET.", "")
    text = normalize_for_match(raw_text)

    def _clean_subject_phrase(chunk: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(chunk or "")).strip(" .,-;:/")
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"^\s*(?:\d+\s*)?(?:môn|mon|học phần|hoc phan|lớp|lop)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"^\s*(?:là|la)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        tail_patterns = [
            r"\s+(?:có|co)\s+nh(?:ữ|u)ng\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+(?:có|co)\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+(?:kỳ|kì|ki|ky)\s+n(?:à|a)y.*$",
            r"\s+l(?:ị|i)ch.*$",
            r"\s+h(?:ô|o)m\s+nào.*$",
            r"\s+th(?:ứ|u)\s+mấy.*$",
            r"\s+nh(?:ư|u)\s+n(?:à|a)o.*$",
            r"\s+ra\s+sao.*$",
            r"\s+mở\s+lớp.*$",
            r"\s+mo\s+lop.*$",
        ]
        for pattern in tail_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\b(?:mà|ma|nhé|nhe|nhá|nha|đó|do|ạ|a|vậy|vay)\s*$", "", cleaned, flags=re.IGNORECASE).strip(" .,-;:/")
        cleaned = re.sub(r"\b(?:và|va|với|voi)\s*$", "", cleaned, flags=re.IGNORECASE).strip(" .,-;:/")
        return cleaned

    subject_patterns = [
        re.compile(
            r"(?:môn|mon|học phần|hoc phan)\s+(.+?)(?:\s+(?:(?:có|co)\s+nh(?:ữ|u)ng\s+ai\s+dạy|(?:có|co)\s+ai\s+dạy|ai dạy|kỳ này|ki nay|kì này|mở lớp|mo lop|lịch|lich|hôm nào|hom nao|thứ mấy|thu may|ra sao|không|khong)|\?|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:về|ve)\s+môn\s+(.+?)(?:\s+(?:kỳ này|ki nay|kì này|ai dạy|ai day|lịch|lich)|\?|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:lớp|lop)\s+(.+?)(?:\s+(?:(?:có|co)\s+nh(?:ữ|u)ng\s+ai\s+dạy|(?:có|co)\s+ai\s+dạy|ai dạy|kỳ này|ki nay|kì này|mở lớp|mo lop|lịch|lich|hôm nào|hom nao|thứ mấy|thu may|ra sao|không|khong)|\?|$)",
            flags=re.IGNORECASE,
        ),
    ]
    generic_hint_keys = {
        "nay",
        "do",
        "ay",
        "cai nay",
        "lop nay",
        "mon nay",
        "nhung ai",
        "ai day",
        "co nhung ai day",
        "co ai day",
        "mon nao",
        "nhung mon nao",
        "nhung mon",
        "nao",
    }
    for pattern in subject_patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_subject_phrase(match.group(1))
        candidate_norm = normalize_for_match(candidate)
        if candidate_norm in generic_hint_keys:
            continue
        if len(candidate) >= 3:
            return candidate
    return None


def _extract_subject_hints(query: str) -> List[str]:
    raw_text = str(query or "").strip()
    if not raw_text:
        return []
    text = normalize_for_match(raw_text)

    correction_match = re.search(
        r"khong phai\s+.+?\s+ma la\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    is_correction = correction_match is not None
    if correction_match:
        # Keep only the corrected tail after "mà là ..."
        text = correction_match.group(1).strip()
        text = re.sub(r"^\s*(?:\d+\s*)?(?:môn|mon|học phần|hoc phan)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(?:mà|ma|nhé|nhe|nhá|nha|đó|do|ạ|a|vậy|vay)\s*$", "", text, flags=re.IGNORECASE).strip(" .,-;:/")

    def _clean_subject_phrase(chunk: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(chunk or "")).strip(" .,-;:/")
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"^\s*(?:\d+\s*)?(?:môn|mon|học phần|hoc phan|lớp|lop)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"^\s*(?:là|la)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^\s*(?:và|va|với|voi|cùng với|cung voi|&)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        tail_patterns = [
            r"\s+(?:có|co)\s+nh(?:ữ|u)ng\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+(?:có|co)\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+ai\s+d(?:ạ|a)y.*$",
            r"\s+(?:kỳ|kì|ki|ky)\s+n(?:à|a)y.*$",
            r"\s+l(?:ị|i)ch.*$",
            r"\s+h(?:ô|o)m\s+nào.*$",
            r"\s+th(?:ứ|u)\s+mấy.*$",
            r"\s+nh(?:ư|u)\s+n(?:à|a)o.*$",
            r"\s+ra\s+sao.*$",
            r"\s+mở\s+lớp.*$",
            r"\s+mo\s+lop.*$",
        ]
        for pattern in tail_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\b(?:mà|ma|nhé|nhe|nhá|nha|đó|do|ạ|a|vậy|vay)\s*$", "", cleaned, flags=re.IGNORECASE).strip(" .,-;:/")
        cleaned = re.sub(r"\b(?:và|va|với|voi)\s*$", "", cleaned, flags=re.IGNORECASE).strip(" .,-;:/")
        return cleaned

    generic_hint_keys = {
        "nay",
        "do",
        "ay",
        "cai nay",
        "lop nay",
        "mon nay",
        "nhung ai",
        "ai day",
        "co nhung ai day",
        "co ai day",
        "mon nao",
        "nhung mon nao",
        "nhung mon",
        "nao",
    }
    cleaned: List[str] = []
    seen: Set[str] = set()

    def _append_chunk(chunk: str) -> None:
        normalized_chunk = _clean_subject_phrase(chunk)
        if not normalized_chunk:
            return
        key = normalize_for_match(normalized_chunk)
        if not key or key in generic_hint_keys or key in seen:
            return
        broad_non_subject_markers = (
            "khung ctdt",
            "khung chuong trinh",
            "chuong trinh dao tao",
            "co nam trong",
            "cua toi",
            "cua em",
            "co mo khong",
            "mo khong",
            "ky nay co mo",
            "ki nay co mo",
            "nhung mon nao",
            "cac mon nao",
            "tat ca cac mon",
            "liet ke",
        )
        if any(marker in key for marker in broad_non_subject_markers):
            return
        if len(normalized_chunk) < 3:
            return
        seen.add(key)
        cleaned.append(normalized_chunk)

    code_matches = re.findall(r"\b(?:UET\.)?([A-Z]{2,4}\d{3,4}[A-Z]?)\b", text, flags=re.IGNORECASE)
    for code in code_matches:
        _append_chunk(str(code).upper().replace("UET.", ""))
    has_course_cue = bool(
        re.search(r"\b(?:môn|mon|học phần|hoc phan|lớp|lop|mã môn|ma mon)\b", text, flags=re.IGNORECASE)
    ) or bool(code_matches)

    marker_iter = list(re.finditer(r"\b(?:môn|mon|học phần|hoc phan|lớp|lop)\b", text, flags=re.IGNORECASE))
    if marker_iter:
        for idx, marker in enumerate(marker_iter):
            start = marker.start()
            end = marker_iter[idx + 1].start() if idx + 1 < len(marker_iter) else len(text)
            segment = text[start:end]
            segment = re.sub(
                r"^\s*(?:\d+\s*)?(?:môn|mon|học phần|hoc phan|lớp|lop)\s+",
                "",
                segment,
                flags=re.IGNORECASE,
            ).strip()
            parts = re.split(
                r"[,;/]+|\s+(?:với|voi|cùng với|cung voi)\s+",
                segment,
                flags=re.IGNORECASE,
            )
            for part in parts:
                _append_chunk(part)

    if not cleaned:
        fallback = _extract_subject_hint(text)
        if fallback:
            for part in re.split(r"[,;/]+", fallback):
                _append_chunk(part)
        elif text and (is_correction or has_course_cue):
            _append_chunk(text)
    return cleaned


def _subject_match_tokens(text: str) -> Set[str]:
    stop_tokens = {
        "mon",
        "hoc",
        "phan",
        "lop",
        "ky",
        "ki",
        "nay",
        "co",
        "nhung",
        "cac",
        "ai",
        "day",
        "giang",
        "vien",
        "vao",
        "hom",
        "nao",
        "va",
        "voi",
        "cung",
        "ma",
        "la",
        "khong",
        "ctdt",
        "chuong",
        "trinh",
        "dao",
        "tao",
        "khung",
        "toi",
        "em",
    }
    return {
        token
        for token in normalize_for_match(text or "").split()
        if len(token) >= 2 and token not in stop_tokens
    }


def _is_likely_subject_hint(text: str) -> bool:
    norm = normalize_for_match(text or "")
    if not norm:
        return False
    broad_markers = (
        "khung ctdt",
        "khung chuong trinh",
        "chuong trinh dao tao",
        "co nam trong",
        "cua toi",
        "cua em",
        "co mo khong",
        "mo khong",
        "ky nay co mo",
        "ki nay co mo",
        "nhung mon nao",
        "cac mon nao",
        "tat ca cac mon",
        "liet ke",
    )
    if any(marker in norm for marker in broad_markers):
        return False
    tokens = [tok for tok in norm.split() if tok]
    return 1 <= len(tokens) <= 8


def _collect_curriculum_subject_entries(curriculum_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = curriculum_payload.get("groups") if isinstance(curriculum_payload.get("groups"), dict) else {}
    entries: List[Dict[str, Any]] = []
    for group_code, group_data in groups.items():
        if not isinstance(group_data, dict):
            continue
        group_name = str(group_data.get("group_name") or "").strip()
        subjects = group_data.get("subjects") if isinstance(group_data.get("subjects"), list) else []
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            subject_code = str(subject.get("code") or "").strip().upper()
            if not subject_code:
                continue
            subject_name = str(
                subject.get("name")
                or subject.get("name_vi")
                or subject.get("subject_name_vi")
                or ""
            ).strip()
            subject_name_norm = normalize_for_match(subject_name)
            entries.append(
                {
                    "subject_code": subject_code,
                    "subject_name": subject_name,
                    "subject_name_norm": subject_name_norm,
                    "subject_code_norm": normalize_for_match(subject_code),
                    "tokens": _subject_match_tokens(f"{subject_code} {subject_name}"),
                    "group_code": str(group_code or "").strip(),
                    "group_name": group_name,
                }
            )
    return entries


def _pick_curriculum_subject_from_hints(
    subject_hints: List[str],
    curriculum_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not subject_hints or not curriculum_entries:
        return None

    best: Optional[Dict[str, Any]] = None
    for hint in subject_hints:
        hint_norm = normalize_for_match(hint or "")
        if not hint_norm or not _is_likely_subject_hint(hint_norm):
            continue
        hint_tokens = _subject_match_tokens(hint_norm)
        for entry in curriculum_entries:
            subject_code = str(entry.get("subject_code") or "").strip().upper()
            name_norm = str(entry.get("subject_name_norm") or "")
            entry_tokens = entry.get("tokens") if isinstance(entry.get("tokens"), set) else set()

            score = 0.0
            if hint_norm == normalize_for_match(subject_code):
                score = 0.99
            elif hint_norm and name_norm and hint_norm in name_norm:
                coverage = min(1.0, len(hint_norm) / max(1, len(name_norm)))
                score = max(score, 0.80 + 0.15 * coverage)
            elif hint_norm and name_norm and name_norm in hint_norm and len(name_norm) >= 6:
                score = max(score, 0.78)

            overlap = len(hint_tokens & entry_tokens)
            if overlap > 0 and hint_tokens and entry_tokens:
                precision = overlap / max(1, len(entry_tokens))
                recall = overlap / max(1, len(hint_tokens))
                f1 = 0.0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                score = max(score, 0.52 + 0.46 * f1)
                if overlap >= 2:
                    score = max(score, 0.72)
                if overlap >= 3:
                    score = max(score, 0.82)

            score = round(min(score, 0.99), 3)
            if score <= 0:
                continue
            candidate = {
                "subject_code": subject_code,
                "subject_name": str(entry.get("subject_name") or "").strip(),
                "group_code": str(entry.get("group_code") or "").strip(),
                "group_name": str(entry.get("group_name") or "").strip(),
                "score": score,
                "hint": hint,
            }
            if (
                best is None
                or float(candidate["score"]) > float(best["score"])
                or (
                    float(candidate["score"]) == float(best["score"])
                    and str(candidate["subject_code"]) < str(best["subject_code"])
                )
            ):
                best = candidate

    if best and float(best.get("score") or 0.0) >= 0.62:
        return best
    return None


def _dedupe_schedule_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = "|".join(
            [
                str(row.get("subject_code") or "").strip().upper(),
                str(row.get("class_code") or "").strip().upper(),
                str(row.get("teacher_name") or "").strip().upper(),
                str(row.get("day_of_week") or "").strip().lower(),
                str(row.get("slot") or "").strip(),
                str(row.get("room") or "").strip().upper(),
                str(row.get("week_note") or "").strip().lower(),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _collapse_schedule_rows_for_display(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        class_code = str(row.get("class_code") or "").strip()
        day = str(row.get("day_of_week") or "").strip() or "Chưa rõ thứ"
        slot = str(row.get("slot") or "").strip()
        room = str(row.get("room") or "").strip()
        key = (
            class_code.upper(),
            normalize_for_match(day),
            str(slot),
            room.upper(),
        )
        event = grouped.get(key)
        if event is None:
            event = {
                "class_code": class_code,
                "day_of_week": day,
                "slot": slot,
                "room": room,
                "teachers": set(),
            }
            grouped[key] = event
        teacher_name = str(row.get("teacher_name") or "").strip()
        if teacher_name:
            event["teachers"].add(teacher_name)

    def _day_order(day_value: str) -> int:
        norm = normalize_for_match(day_value or "")
        match = re.search(r"\bthu\s*([2-8])\b", norm)
        if match:
            return int(match.group(1))
        if "chu nhat" in norm:
            return 8
        return 99

    def _slot_order(slot_value: str) -> int:
        slot_text = str(slot_value or "").strip()
        return int(slot_text) if slot_text.isdigit() else 99

    ordered_events = sorted(
        grouped.values(),
        key=lambda item: (
            _day_order(str(item.get("day_of_week") or "")),
            _slot_order(str(item.get("slot") or "")),
            str(item.get("class_code") or "").upper(),
            str(item.get("room") or "").upper(),
        ),
    )
    normalized_events: List[Dict[str, Any]] = []
    for event in ordered_events:
        normalized_events.append(
            {
                "class_code": str(event.get("class_code") or "").strip(),
                "day_of_week": str(event.get("day_of_week") or "").strip() or "Chưa rõ thứ",
                "slot": str(event.get("slot") or "").strip(),
                "room": str(event.get("room") or "").strip(),
                "teachers": sorted({str(t).strip() for t in event.get("teachers", set()) if str(t).strip()}),
            }
        )
    return normalized_events


def _query_uses_deictic_subject_reference(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False
    deictic_markers = (
        "mon nay",
        "lop nay",
        "hoc phan nay",
        "mon ay",
        "lop ay",
        "hoc phan ay",
        "cai nay",
        "lop nay",
    )
    return any(marker in norm for marker in deictic_markers)


def _extract_recent_subject_code_from_memory(memory_text: str) -> Optional[str]:
    text = str(memory_text or "").strip()
    if not text:
        return None
    recent_window = text[-3000:]
    code_pattern = re.compile(r"\b(?:UET\.)?([A-Z]{2,4}\d{3,4}[A-Z]?)\b", flags=re.IGNORECASE)
    matches = list(code_pattern.finditer(recent_window))
    if not matches:
        return None
    return matches[-1].group(1).upper().replace("UET.", "")


def _query_targets_time_slot_definition(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False

    has_slot = bool(re.search(r"\bca\s*[1-9]\b", norm))
    if not has_slot:
        return False

    time_markers = (
        "may gio",
        "gio nao",
        "bat dau",
        "ket thuc",
        "tu may gio",
        "den may gio",
        "khung gio",
        "gio hoc",
        "bat dau tu",
        "ket thuc luc",
    )
    if not any(marker in norm for marker in time_markers):
        return False

    has_subject_scope = bool(re.search(r"\b[a-z]{2,4}\d{3,4}[a-z]?\b", norm)) or any(
        marker in norm
        for marker in (
            " mon ",
            " hoc phan ",
            " lop ",
            " giang vien ",
            " gv ",
            " phong ",
            " thu ",
            " thu may",
        )
    )
    return not has_subject_scope


def _extract_time_slot_number_from_query(query: str) -> Optional[str]:
    norm = normalize_for_match(query or "")
    if not norm:
        return None
    match = re.search(r"\bca\s*([1-9])\b", norm)
    if not match:
        return None
    return match.group(1)


def _has_schedule_lookup_signal(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False

    if _query_targets_time_slot_definition(query):
        return True

    direct_markers = (
        "lich hoc",
        "thu may",
        "hom nao",
        "trong tuan",
        "ca nao",
        "tiet nao",
        "phong nao",
        "gio hoc",
    )
    if any(marker in norm for marker in direct_markers):
        return True

    if re.search(r"\blich\s+(?:mon|lop|hoc phan|cac mon|cac lop|cac hoc phan)\b", norm):
        return True

    has_scope = any(
        marker in norm
        for marker in (
            " mon ",
            " lop ",
            " hoc phan ",
            "cac mon",
            "cac lop",
            "cac hoc phan",
        )
    )
    if "lich" in norm and has_scope and any(marker in norm for marker in ("trong tuan", "nhu nao")):
        return True

    return False


def _query_targets_semester_code_lookup(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False

    has_semester_code_phrase = any(
        marker in norm
        for marker in (
            "ma ky hoc",
            "ma hoc ky",
            "ma ki hoc",
            "ma hoc ki",
            "ma ky",
            "ma ki",
        )
    )
    if not has_semester_code_phrase:
        return False

    has_schedule_scope = any(
        marker in norm
        for marker in (
            "thoi khoa bieu",
            "tkb",
            "hoc ky nay",
            "hoc ki nay",
            "hoc ky hien tai",
            "hoc ki hien tai",
            "ky hien tai",
            "ki hien tai",
            "ky nay",
            "ki nay",
        )
    )
    has_term_or_year = bool(_detect_semester_term(norm)) or bool(re.search(r"(20\d{2}|\d{2})\s*[-/]\s*(20\d{2}|\d{2})", norm))
    return has_schedule_scope or has_term_or_year


def _query_asks_schedule_details(query: str) -> bool:
    return _has_schedule_lookup_signal(query)


def _query_requests_elective_recommendation(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False
    recommendation_markers = (
        "lien quan",
        "dinh huong",
        "theo huong",
        "phu hop",
        "nen hoc",
        "goi y",
        "uu tien",
        "tap trung",
    )
    has_recommendation_signal = any(marker in norm for marker in recommendation_markers)
    has_elective_scope = any(
        marker in norm
        for marker in (
            "tu chon",
            "hoc phan tu chon",
            "hoc phan lua chon",
            "mo lop",
            "ky nay",
            "ki nay",
        )
    )
    return has_recommendation_signal and has_elective_scope


def _normalize_opened_elective_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    code = str(item.get("code") or item.get("subject_code") or "").strip().upper()
    if not code:
        return None
    name_vi = str(
        item.get("name_vi")
        or item.get("subject_name_vi")
        or item.get("name")
        or item.get("subject_name")
        or ""
    ).strip()
    name_en = str(item.get("name_en") or item.get("subject_name_en") or "").strip()
    group_code = str(item.get("group_code") or "").strip()
    group_name = str(item.get("group") or item.get("group_name") or "").strip()
    credits = item.get("credits")
    try:
        credits_val: Optional[float] = float(credits) if credits is not None and str(credits).strip() else None
    except Exception:
        credits_val = None
    return {
        "subject_code": code,
        "subject_name_vi": name_vi,
        "subject_name_en": name_en,
        "group_code": group_code,
        "group_name": group_name,
        "credits": credits_val,
        "raw": item,
    }


def _fallback_rank_opened_electives(
    query: str,
    opened_items: List[Dict[str, Any]],
    top_k: int = 6,
) -> Dict[str, Any]:
    query_tokens = {
        tok
        for tok in normalize_for_match(query or "").split()
        if tok
        and tok not in {
            "toi",
            "em",
            "can",
            "hoc",
            "mon",
            "hocphan",
            "hocphan",
            "tu",
            "chon",
            "nhung",
            "cac",
            "nao",
            "ky",
            "ki",
            "nay",
            "mo",
            "lop",
            "trong",
            "tuan",
            "thu",
            "hom",
        }
    }
    scored: List[Tuple[float, str]] = []
    for item in opened_items:
        code = str(item.get("subject_code") or "").strip().upper()
        haystack = normalize_for_match(
            " ".join(
                [
                    str(item.get("subject_code") or ""),
                    str(item.get("subject_name_vi") or ""),
                    str(item.get("subject_name_en") or ""),
                    str(item.get("group_name") or ""),
                    str(item.get("group_code") or ""),
                ]
            )
        )
        hay_tokens = set(haystack.split())
        overlap = len(query_tokens & hay_tokens)
        score = float(overlap)
        scored.append((score, code))
    scored.sort(key=lambda item: (-item[0], item[1]))
    positive_codes = [code for score, code in scored if score > 0][: max(1, min(top_k, 8))]
    selected_codes = list(positive_codes)
    if not selected_codes and opened_items:
        selected_codes = [
            str(item.get("subject_code") or "").strip().upper()
            for item in opened_items[: max(1, min(top_k, 4))]
        ]
    return {
        "focus": "",
        "selected_codes": selected_codes,
        "positive_codes": positive_codes,
        "reason_by_code": {},
        "confidence": 0.0,
        "used_fallback": True,
    }


def _get_elective_interest_agent_instance():
    global _elective_interest_agent
    if _elective_interest_agent is None:
        _elective_interest_agent = get_elective_interest_agent()
    return _elective_interest_agent


def _rank_opened_electives_for_query(
    query: str,
    opened_items: List[Dict[str, Any]],
    top_k: int = 6,
) -> Dict[str, Any]:
    candidates = [item for item in opened_items if str(item.get("subject_code") or "").strip()]
    if not candidates:
        return {
            "focus": "",
            "selected_codes": [],
            "reason_by_code": {},
            "confidence": 0.0,
            "used_fallback": True,
        }
    top_k = max(1, min(int(top_k or 6), 8, len(candidates)))
    fallback = _fallback_rank_opened_electives(query=query, opened_items=candidates, top_k=top_k)
    fallback_codes = [str(code or "").strip().upper() for code in fallback.get("selected_codes") or [] if str(code or "").strip()]
    fallback_positive_codes = [
        str(code or "").strip().upper() for code in fallback.get("positive_codes") or [] if str(code or "").strip()
    ]
    index = {str(item.get("subject_code") or "").strip().upper(): item for item in candidates}
    candidate_payload = [
        {
            "code": str(item.get("subject_code") or "").strip().upper(),
            "name_vi": str(item.get("subject_name_vi") or "").strip(),
            "name_en": str(item.get("subject_name_en") or "").strip(),
            "group_code": str(item.get("group_code") or "").strip(),
            "group_name": str(item.get("group_name") or "").strip(),
            "credits": item.get("credits"),
        }
        for item in candidates
    ]
    prompt = (
        "Câu hỏi người dùng:\n"
        f"{query}\n\n"
        "Danh sách học phần tự chọn đang mở lớp (JSON):\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False)}\n\n"
        f"Hãy chọn tối đa {top_k} mã môn phù hợp nhất với định hướng trong câu hỏi."
    )
    try:
        response = _get_elective_interest_agent_instance().run(prompt)
        raw_content = getattr(response, "content", response)
        parsed = _safe_json_loads(raw_content)
        selected_raw = parsed.get("selected_codes")
        selected_codes: List[str] = []
        if isinstance(selected_raw, list):
            for raw_code in selected_raw:
                code = str(raw_code or "").strip().upper()
                if code and code in index and code not in selected_codes:
                    selected_codes.append(code)
                if len(selected_codes) >= top_k:
                    break
        if not selected_codes:
            selected_codes = list(fallback_codes)
        elif fallback_positive_codes:
            # Guardrail: keep LLM ranking flexible, but prioritize codes that
            # also have lexical support from query/course metadata.
            merged_codes: List[str] = []
            for code in selected_codes:
                if code in fallback_positive_codes and code not in merged_codes:
                    merged_codes.append(code)
            for code in fallback_positive_codes:
                if code not in merged_codes:
                    merged_codes.append(code)
                if len(merged_codes) >= top_k:
                    break
            if merged_codes:
                selected_codes = merged_codes[:top_k]
        reasons_obj = parsed.get("reason_by_code")
        reason_by_code: Dict[str, str] = {}
        if isinstance(reasons_obj, dict):
            for code, reason in reasons_obj.items():
                normalized_code = str(code or "").strip().upper()
                if normalized_code in index and normalized_code in selected_codes:
                    reason_text = str(reason or "").strip()
                    if reason_text:
                        reason_by_code[normalized_code] = reason_text
        confidence_raw = parsed.get("confidence")
        try:
            confidence_val = float(confidence_raw)
        except Exception:
            confidence_val = 0.0
        return {
            "focus": str(parsed.get("focus") or "").strip(),
            "selected_codes": selected_codes,
            "reason_by_code": reason_by_code,
            "confidence": max(0.0, min(confidence_val, 1.0)),
            "used_fallback": False,
        }
    except Exception as exc:
        logger.warning("elective interest ranking failed, fallback to lexical scoring: %s", exc)
        return fallback


def _build_elective_recommendation_payload(
    query: str,
    opened: List[Dict[str, Any]],
    session_id: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    normalized_items = [
        item
        for item in (_normalize_opened_elective_item(raw_item) for raw_item in (opened or []))
        if item is not None
    ]
    ranking = _rank_opened_electives_for_query(query=query, opened_items=normalized_items, top_k=top_k)
    selected_codes = [str(code or "").strip().upper() for code in ranking.get("selected_codes") or [] if str(code or "").strip()]
    index = {str(item.get("subject_code") or "").strip().upper(): item for item in normalized_items}
    selected_subjects = [index[code] for code in selected_codes if code in index]
    if not selected_subjects and normalized_items:
        selected_subjects = normalized_items[: min(max(top_k, 1), len(normalized_items))]

    asks_schedule = _query_asks_schedule_details(query)
    rows: List[Dict[str, Any]] = []
    no_data_subjects: List[Dict[str, Any]] = []
    source_files: Set[str] = set()

    if asks_schedule:
        for subject in selected_subjects[:8]:
            subject_code = str(subject.get("subject_code") or "").strip().upper()
            if not subject_code:
                continue
            schedule_raw = mcp_client.invoke(
                "get_schedule_rows",
                {"subject_code": subject_code, "session_id": session_id},
            )
            schedule_payload = _safe_json_loads(schedule_raw)
            source_vals = schedule_payload.get("source_files")
            if isinstance(source_vals, list):
                for source_val in source_vals:
                    if str(source_val or "").strip():
                        source_files.add(str(source_val).strip())
            schedule_rows = schedule_payload.get("rows") if isinstance(schedule_payload.get("rows"), list) else []
            if not schedule_rows:
                no_data_subjects.append(
                    {
                        "subject_code": subject_code,
                        "subject_name_vi": str(subject.get("subject_name_vi") or ""),
                        "subject_name_en": str(subject.get("subject_name_en") or ""),
                    }
                )
                continue
            for row in schedule_rows:
                if not isinstance(row, dict):
                    continue
                normalized_row = dict(row)
                if not str(normalized_row.get("subject_code") or "").strip():
                    normalized_row["subject_code"] = subject_code
                if not str(normalized_row.get("subject_name_vi") or "").strip():
                    normalized_row["subject_name_vi"] = str(subject.get("subject_name_vi") or "")
                if not str(normalized_row.get("subject_name_en") or "").strip():
                    normalized_row["subject_name_en"] = str(subject.get("subject_name_en") or "")
                rows.append(normalized_row)
        rows = _dedupe_schedule_rows(rows)

    recommendation_rows: List[Dict[str, Any]] = []
    for subject in selected_subjects:
        subject_code = str(subject.get("subject_code") or "").strip().upper()
        recommendation_rows.append(
            {
                "subject_code": subject_code,
                "subject_name_vi": str(subject.get("subject_name_vi") or ""),
                "subject_name_en": str(subject.get("subject_name_en") or ""),
                "group_code": str(subject.get("group_code") or ""),
                "group_name": str(subject.get("group_name") or ""),
                "credits": subject.get("credits"),
            }
        )

    coverage_note = "Danh sách môn gợi ý theo định hướng từ câu hỏi."
    if asks_schedule:
        if rows:
            coverage_note = "Đã kết hợp gợi ý môn và lịch học chi tiết theo TKB structured."
        elif no_data_subjects:
            coverage_note = "Đã gợi ý môn theo định hướng, nhưng chưa thấy dữ liệu lịch cho một số môn."
        else:
            coverage_note = "Đã gợi ý môn theo định hướng, nhưng chưa lấy được lịch học structured."

    return {
        "focus": str(ranking.get("focus") or "").strip(),
        "confidence": float(ranking.get("confidence") or 0.0),
        "used_fallback": bool(ranking.get("used_fallback")),
        "recommended_subjects": recommendation_rows,
        "reason_by_code": ranking.get("reason_by_code") if isinstance(ranking.get("reason_by_code"), dict) else {},
        "rows": rows,
        "no_data_subjects": no_data_subjects,
        "coverage_note": coverage_note,
        "source_files": sorted(source_files),
    }


def _query_targets_specialized_electives(query: str) -> bool:
    norm = normalize_for_match(query or "")
    if not norm:
        return False
    specialized_markers = (
        "tu chon theo chuyen nganh",
        "mon tu chon theo chuyen nganh",
        "chuyen nganh",
        "nhom v.2",
    )
    return any(marker in norm for marker in specialized_markers)


def _query_requires_planner_orchestration(query: str, route_intent: str) -> bool:
    """
    Decide whether this query should be routed through planner orchestration even
    when structured routing has high confidence.
    """
    norm = normalize_for_match(query or "")
    intent = str(route_intent or "").strip()
    if not norm or not intent:
        return False

    has_schedule_detail = _query_asks_schedule_details(query)

    if intent == "electives_overview":
        # elective overview now has deterministic structured handling for
        # recommendation/filter + schedule detail queries.
        if _query_requests_elective_recommendation(query):
            return False

    if intent in {"course_schedule", "teacher_by_subject", "course_offering_status"}:
        if has_schedule_detail and len(_extract_subject_hints(query)) >= 3:
            return True

    return False


def _structured_intent_classifier(query: str) -> Dict[str, Any]:
    norm = normalize_for_match(query or "")
    if not norm:
        return {"intent": None, "confidence": 0.0, "signals": []}

    signals: List[str] = []
    subject_hints = _extract_subject_hints(query or "")
    has_teacher_marker = any(marker in norm for marker in ("giang vien", "ai day", "day mon", "co nao day", "thay nao day"))
    has_schedule_marker = _has_schedule_lookup_signal(query)
    has_class_marker = any(marker in norm for marker in ("lop nao", "day lop nao", "nhung lop nao", "lop nay"))
    asks_teacher_course_list = any(
        marker in norm
        for marker in (
            "day nhung mon nao",
            "day mon nao",
            "giang day nhung mon nao",
            "giang day mon nao",
            "day cac mon nao",
        )
    )
    has_course_marker = any(marker in norm for marker in ("mon ", "hoc phan", "ma mon", "mon nay", "lop nay"))
    has_teacher_name = _extract_teacher_name_hint(query or "") is not None
    has_course_code = bool(re.search(r"\b[a-z]{2,4}\d{3,4}[a-z]?\b", norm))
    has_subject_hint = bool(subject_hints)
    has_correction_marker = "khong phai" in norm and "ma la" in norm
    has_elective_marker = any(
        marker in norm
        for marker in (
            "tu chon",
            "hoc phan tu chon",
            "chuyen nganh",
            "tat ca cac mon",
            "tat ca nhung mon",
            "liet ke",
            "trong cho nay",
        )
    )
    has_opening_marker = any(
        marker in norm
        for marker in (
            "mo lop",
            "co mo",
            "mo khong",
            "chua mo",
            "mo trong ky",
            "ki nay co mo",
            "ky nay co mo",
            "mo hay khong",
        )
    )
    has_curriculum_scope_marker = any(
        marker in norm
        for marker in (
            "ctdt",
            "chuong trinh dao tao",
            "khung ctdt",
            "khung chuong trinh",
        )
    )
    asks_list_marker = any(
        marker in norm
        for marker in ("nhung mon nao", "cac mon nao", "tat ca", "liet ke", "trong cho nay")
    )
    subject_hint_specific = False
    if has_subject_hint:
        first_hint = normalize_for_match(subject_hints[0] if subject_hints else "")
        first_tokens = [tok for tok in first_hint.split() if tok]
        broad_hint_markers = (
            "chuong trinh dao tao",
            "khung ctdt",
            "con thieu",
            "thieu mon",
            "tin chi",
            "tat ca",
            "cac mon",
            "nhung mon",
            "tu chon",
        )
        subject_hint_specific = bool(first_tokens) and len(first_tokens) <= 8 and not any(
            marker in first_hint for marker in broad_hint_markers
        )

    if has_teacher_marker:
        signals.append("teacher_marker")
    if has_schedule_marker:
        signals.append("schedule_marker")
    if has_class_marker:
        signals.append("class_marker")
    if has_course_marker or has_course_code:
        signals.append("course_marker")
    if has_teacher_name:
        signals.append("teacher_name")
    if has_correction_marker:
        signals.append("correction_marker")
    if has_elective_marker:
        signals.append("elective_marker")
    if has_opening_marker:
        signals.append("opening_marker")
    if has_curriculum_scope_marker:
        signals.append("curriculum_scope_marker")
    if _query_targets_semester_code_lookup(query):
        signals.append("semester_code_marker")

    if _query_targets_semester_code_lookup(query):
        return {"intent": "semester_code_lookup", "confidence": 0.9, "signals": signals}

    if has_teacher_name and (has_class_marker or asks_teacher_course_list or ("day" in norm and not has_subject_hint)):
        return {"intent": "classes_by_teacher", "confidence": 0.87, "signals": signals}
    if (
        subject_hint_specific
        and (has_opening_marker or has_curriculum_scope_marker)
        and not has_elective_marker
        and not asks_list_marker
        and not has_teacher_marker
        and not has_schedule_marker
    ):
        return {"intent": "course_offering_status", "confidence": 0.82, "signals": signals}
    if (
        (has_elective_marker or (has_opening_marker and asks_list_marker))
        and not has_teacher_marker
        and not has_teacher_name
        and not has_course_code
        and not subject_hint_specific
        and not has_class_marker
    ):
        # Keep broad elective-opening questions on the electives flow even when users
        # add schedule words like "hôm nào/trong tuần".
        confidence = 0.83 if has_schedule_marker else 0.79
        return {"intent": "electives_overview", "confidence": confidence, "signals": signals}
    if has_schedule_marker and (has_course_marker or has_course_code or has_class_marker or has_subject_hint):
        confidence = 0.84 if has_teacher_marker else 0.78
        return {"intent": "course_schedule", "confidence": confidence, "signals": signals}
    if has_teacher_marker and (has_course_marker or has_course_code or has_subject_hint):
        return {"intent": "teacher_by_subject", "confidence": 0.82, "signals": signals}
    if has_correction_marker and (has_course_marker or has_course_code or has_subject_hint):
        # Follow-up correction (e.g. "không phải môn A mà là môn B") should stay on structured course lookup,
        # not bounce back to consult_advisor.
        return {"intent": "teacher_by_subject", "confidence": 0.76, "signals": signals}
    if has_schedule_marker:
        return {"intent": "course_schedule", "confidence": 0.62, "signals": signals}
    if has_teacher_marker:
        return {"intent": "teacher_by_subject", "confidence": 0.56, "signals": signals}
    return {"intent": None, "confidence": 0.0, "signals": signals}


def _structured_payload_has_rows(payload: Dict[str, Any]) -> bool:
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        return True
    teachers = payload.get("teachers")
    if isinstance(teachers, list) and teachers:
        return True
    return False


def _current_academic_year_start(today: Optional[date] = None) -> int:
    now = today or date.today()
    return now.year if now.month >= 8 else (now.year - 1)


def _detect_semester_term(norm_query: str) -> Optional[str]:
    compact = re.sub(r"[^a-z0-9]", "", str(norm_query or ""))
    if any(token in compact for token in ("hkiii", "hockyiii", "hocky3", "kyhe", "kihe", "hoche", "summer")):
        return "summer"
    if any(token in compact for token in ("hkii", "hockyii", "hocky2", "ky2", "ki2", "semester2", "hockyii")):
        return "semester2"
    if any(token in compact for token in ("hki", "hockyi", "hocky1", "ky1", "ki1", "semester1", "hockyi")):
        return "semester1"
    return None


def _infer_schedule_semester_code(query: str) -> Optional[str]:
    raw = str(query or "")
    if not raw.strip():
        return None
    norm = normalize_for_match(raw)

    explicit_code = re.search(r"\b(\d{3})\b", raw)
    if explicit_code:
        code = explicit_code.group(1)
        if code[-1] in {"1", "2"}:
            return code

    year_match = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})", norm)
    short_year_match = None if year_match else re.search(r"\b(\d{2})\s*[-/]\s*(\d{2})\b", norm)
    term = _detect_semester_term(norm)

    if year_match and int(year_match.group(2)) == int(year_match.group(1)) + 1:
        year_start = int(year_match.group(1))
        if term == "semester1":
            return f"{year_start % 100:02d}1"
        if term in {"semester2", "summer"}:
            # Policy mapping: HKII and summer share x52 code.
            return f"{year_start % 100:02d}2"
    if short_year_match:
        year_start_2d = int(short_year_match.group(1))
        year_end_2d = int(short_year_match.group(2))
        if (year_start_2d + 1) % 100 == year_end_2d:
            if term == "semester1":
                return f"{year_start_2d:02d}1"
            if term in {"semester2", "summer"}:
                return f"{year_start_2d:02d}2"

    if term:
        year_start = _current_academic_year_start()
        term_digit = "1" if term == "semester1" else "2"
        return f"{year_start % 100:02d}{term_digit}"

    deictic_this_semester_markers = (
        "ky nay",
        "ki nay",
        "hoc ky nay",
        "hoc ki nay",
        "ky hien tai",
        "ki hien tai",
        "hoc ky hien tai",
        "hoc ki hien tai",
    )
    if any(marker in norm for marker in deictic_this_semester_markers):
        year_start = _current_academic_year_start()
        term_digit = "1" if date.today().month >= 8 else "2"
        return f"{year_start % 100:02d}{term_digit}"

    return None


def _infer_schedule_semester_code_from_schedule_rows(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(rows, list):
        return None
    code_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        semester_label = str(row.get("semester") or "").strip()
        if not semester_label:
            continue
        inferred = _infer_schedule_semester_code(semester_label)
        if not inferred:
            continue
        code_counts[inferred] = int(code_counts.get(inferred, 0)) + 1
    if not code_counts:
        return None
    return sorted(code_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_schedule_semester_code_from_source_files(source_files: List[str]) -> Optional[str]:
    if not isinstance(source_files, list):
        return None
    code_counts: Dict[str, int] = {}
    for source_file in source_files:
        source_name = str(source_file or "").strip()
        if not source_name:
            continue
        inferred = _infer_schedule_semester_code(source_name)
        if not inferred:
            continue
        code_counts[inferred] = int(code_counts.get(inferred, 0)) + 1
    if not code_counts:
        return None
    return sorted(code_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _build_structured_route_payload(
    query: str,
    session_id: str,
    program_id: Optional[str],
    intent: str,
    confidence: float,
    memory_context: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    memory_text = memory_context if memory_context is not None else _load_memory_context_for_session(session_id=session_id, max_rows=10, user_id=user_id)
    semester_code = _infer_schedule_semester_code(query)

    def _payload(source: str, context_obj: Any, tool_used: str, fallback_stage: str) -> Dict[str, Any]:
        context_text = _context_to_text(context_obj)
        return {
            "source": source,
            "context": context_text,
            "memory": memory_text,
            "chunk_index": None,
            "route_meta": {
                "intent": intent,
                "confidence": round(float(confidence or 0.0), 3),
                "tool_used": tool_used,
                "fallback_stage": fallback_stage,
            },
        }

    try:
        if intent == "semester_code_lookup":
            resolved_code = semester_code
            inference_source = "query"
            source_files: Set[str] = set()
            rows: List[Dict[str, Any]] = []
            if not resolved_code:
                schedule_raw = mcp_client.invoke(
                    "get_schedule_rows",
                    {"session_id": session_id, "semester": None},
                )
                schedule_payload = _safe_json_loads(schedule_raw)
                rows = schedule_payload.get("rows") if isinstance(schedule_payload.get("rows"), list) else []
                rows = [row for row in rows if isinstance(row, dict)]
                source_items = (
                    schedule_payload.get("source_files")
                    if isinstance(schedule_payload.get("source_files"), list)
                    else []
                )
                for item in source_items:
                    src_name = str(item or "").strip()
                    if src_name:
                        source_files.add(src_name)
                for row in rows:
                    src_name = str(row.get("source_file") or "").strip()
                    if src_name:
                        source_files.add(src_name)

                resolved_code = _infer_schedule_semester_code_from_schedule_rows(rows)
                if resolved_code:
                    inference_source = "schedule_rows"
                else:
                    resolved_code = _infer_schedule_semester_code_from_source_files(sorted(source_files))
                    if resolved_code:
                        inference_source = "source_file"

            if not resolved_code:
                fallback = _infer_schedule_semester_code("hoc ky hien tai")
                if fallback:
                    resolved_code = fallback
                    inference_source = "calendar_fallback"

            semester_payload = {
                "semester_code": resolved_code or "",
                "inference_source": inference_source,
                "source_files": sorted(source_files),
                "coverage_note": (
                    "Đã suy luận mã học kỳ theo dữ liệu thời khóa biểu hiện có."
                    if resolved_code
                    else "Chưa suy luận được mã học kỳ từ dữ liệu thời khóa biểu hiện có."
                ),
            }
            return _payload(
                "semester_code_lookup",
                semester_payload,
                "get_schedule_rows",
                "structured",
            )

        if intent == "course_schedule" and _query_targets_time_slot_definition(query):
            slot = _extract_time_slot_number_from_query(query)
            if slot:
                time_slot_raw = mcp_client.invoke(
                    "get_time_slot_info",
                    {"slot": slot, "query": query, "session_id": session_id},
                )
                time_slot_payload = _safe_json_loads(time_slot_raw)
                if isinstance(time_slot_payload, dict) and str(time_slot_payload.get("slot") or "").strip():
                    return _payload(
                        "time_slot_lookup",
                        time_slot_payload,
                        "get_time_slot_info",
                        "structured_time_slot",
                    )

        if intent == "electives_overview":
            electives_raw = mcp_client.invoke(
                "get_electives_with_schedule",
                {"check_schedule": True, "program_id": program_id, "session_id": session_id},
            )
            electives_payload = _safe_json_loads(electives_raw)
            opened = electives_payload.get("opened") if isinstance(electives_payload.get("opened"), list) else []
            if opened:
                if _query_requests_elective_recommendation(query):
                    recommendation_payload = _build_elective_recommendation_payload(
                        query=query,
                        opened=opened,
                        session_id=session_id,
                        top_k=6,
                    )
                    return _payload(
                        "electives_recommendation",
                        recommendation_payload,
                        "get_electives_with_schedule+interest_ranker",
                        "structured",
                    )
                if _query_targets_specialized_electives(query):
                    specialized_opened = [
                        item
                        for item in opened
                        if str((item or {}).get("group_code") or "").strip().upper().startswith("V.2")
                    ]
                    if specialized_opened:
                        electives_payload = dict(electives_payload)
                        electives_payload["opened"] = specialized_opened
                        electives_payload["opened_count"] = len(specialized_opened)
                        electives_payload["coverage_note"] = (
                            "Danh sách học phần tự chọn theo chuyên ngành đang mở lớp (nhóm V.2)."
                        )
                        opened = specialized_opened
                if _query_asks_schedule_details(query):
                    merged_rows: List[Dict[str, Any]] = []
                    no_data_subjects: List[Dict[str, str]] = []
                    seen_codes: Set[str] = set()
                    for item in opened:
                        if not isinstance(item, dict):
                            continue
                        code = str(item.get("code") or "").strip().upper()
                        if not code or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        schedule_raw = mcp_client.invoke(
                            "get_schedule_rows",
                            {"subject_code": code, "session_id": session_id, "semester": semester_code},
                        )
                        schedule_payload = _safe_json_loads(schedule_raw)
                        rows = schedule_payload.get("rows") if isinstance(schedule_payload.get("rows"), list) else []
                        rows = [r for r in rows if isinstance(r, dict)]
                        if rows:
                            merged_rows.extend(rows)
                        else:
                            no_data_subjects.append(
                                {
                                    "subject_code": code,
                                    "subject_name_vi": str(item.get("name") or "").strip(),
                                    "subject_name_en": str(item.get("name_en") or "").strip(),
                                }
                            )
                    electives_payload = dict(electives_payload)
                    electives_payload["rows"] = _dedupe_schedule_rows(merged_rows)
                    electives_payload["no_data_subjects"] = no_data_subjects
                return _payload(
                    "electives_schedule",
                    electives_payload,
                    "get_electives_with_schedule",
                    "structured",
                )

        if intent == "course_offering_status":
            subject_hints = _extract_subject_hints(query)[:4]
            if not subject_hints:
                fallback_subject = _extract_subject_hint(query)
                if fallback_subject:
                    subject_hints = [fallback_subject]
            if not subject_hints:
                subject_hints = [query]
            curriculum_raw = mcp_client.invoke(
                "get_curriculum_lookup",
                {"program_id": program_id, "session_id": session_id},
            )
            curriculum_payload = _safe_json_loads(curriculum_raw)
            curriculum_entries = _collect_curriculum_subject_entries(curriculum_payload)
            curriculum_index = {
                str(entry.get("subject_code") or "").strip().upper(): entry for entry in curriculum_entries
            }
            curriculum_candidate = _pick_curriculum_subject_from_hints(subject_hints, curriculum_entries)

            alias_best_subject: Dict[str, Any] = {}
            alias_best_code = ""
            alias_best_confidence = 0.0
            for hint in subject_hints:
                if not _is_likely_subject_hint(hint):
                    continue
                alias_raw = mcp_client.invoke(
                    "resolve_course_alias",
                    {"query": hint, "program_id": program_id, "session_id": session_id},
                )
                alias_payload = _safe_json_loads(alias_raw)
                candidate = (
                    alias_payload.get("matched_subject")
                    if isinstance(alias_payload.get("matched_subject"), dict)
                    else {}
                )
                candidate_code = str(candidate.get("subject_code") or "").strip().upper()
                candidate_conf = float(alias_payload.get("confidence") or 0.0)
                if candidate_code and candidate_conf > alias_best_confidence:
                    alias_best_subject = candidate
                    alias_best_code = candidate_code
                    alias_best_confidence = candidate_conf

            resolved_code = ""
            matched_subject: Dict[str, Any] = {}
            resolution_confidence = 0.0
            resolution_strategy = "unresolved"

            if alias_best_code and alias_best_confidence >= 0.9:
                resolved_code = alias_best_code
                matched_subject = alias_best_subject
                resolution_confidence = alias_best_confidence
                resolution_strategy = "alias_high_confidence"
            elif curriculum_candidate and (
                not alias_best_code
                or alias_best_confidence < 0.75
                or float(curriculum_candidate.get("score") or 0.0) >= alias_best_confidence + 0.05
            ):
                resolved_code = str(curriculum_candidate.get("subject_code") or "").strip().upper()
                matched_subject = {
                    "subject_code": resolved_code,
                    "subject_name_vi": str(curriculum_candidate.get("subject_name") or "").strip(),
                    "subject_name_en": "",
                }
                resolution_confidence = float(curriculum_candidate.get("score") or 0.0)
                resolution_strategy = "curriculum_fallback"
            elif alias_best_code:
                resolved_code = alias_best_code
                matched_subject = alias_best_subject
                resolution_confidence = alias_best_confidence
                resolution_strategy = "alias"
            elif curriculum_candidate:
                resolved_code = str(curriculum_candidate.get("subject_code") or "").strip().upper()
                matched_subject = {
                    "subject_code": resolved_code,
                    "subject_name_vi": str(curriculum_candidate.get("subject_name") or "").strip(),
                    "subject_name_en": "",
                }
                resolution_confidence = float(curriculum_candidate.get("score") or 0.0)
                resolution_strategy = "curriculum_only"

            rows: List[Dict[str, Any]] = []
            if resolved_code:
                schedule_raw = mcp_client.invoke(
                    "get_schedule_rows",
                    {"subject_code": resolved_code, "session_id": session_id, "semester": semester_code},
                )
                schedule_payload = _safe_json_loads(schedule_raw)
                schedule_rows = schedule_payload.get("rows") if isinstance(schedule_payload.get("rows"), list) else []
                rows = _dedupe_schedule_rows([r for r in schedule_rows if isinstance(r, dict)])

            curriculum_entry = curriculum_index.get(resolved_code) if resolved_code else None
            in_curriculum = curriculum_entry is not None
            curriculum_group_code = str((curriculum_entry or {}).get("group_code") or "").strip()
            curriculum_group_name = str((curriculum_entry or {}).get("group_name") or "").strip()

            if resolved_code:
                status_payload = {
                    "matched_subject": matched_subject,
                    "subject_code": resolved_code,
                    "rows": rows,
                    "is_opened": bool(rows),
                    "in_curriculum": in_curriculum,
                    "curriculum_group_code": curriculum_group_code,
                    "curriculum_group_name": curriculum_group_name,
                    "resolution_confidence": round(float(resolution_confidence or 0.0), 3),
                    "resolution_strategy": resolution_strategy,
                }
                return _payload(
                    "course_offering_status",
                    status_payload,
                    "get_schedule_rows",
                    "structured",
                )

        if intent == "classes_by_teacher":
            teacher_hint = _extract_teacher_name_hint(query) or query
            structured_raw = mcp_client.invoke(
                "get_classes_by_teacher",
                {"teacher_name": teacher_hint, "session_id": session_id, "semester": semester_code},
            )
            structured_payload = _safe_json_loads(structured_raw)
            if _structured_payload_has_rows(structured_payload):
                return _payload("structured_schedule", structured_payload, "get_classes_by_teacher", "structured")

        if intent in {"teacher_by_subject", "course_schedule"}:
            subject_hints = _extract_subject_hints(query)
            subject_hints = subject_hints[:8]
            memory_subject_code = _extract_recent_subject_code_from_memory(memory_text)
            is_deictic = _query_uses_deictic_subject_reference(query)
            has_explicit_course_code = bool(re.search(r"\b[a-z]{2,4}\d{3,4}[a-z]?\b", normalize_for_match(query or "")))

            curriculum_candidate: Optional[Dict[str, Any]] = None
            if intent == "course_schedule" and program_id and subject_hints:
                try:
                    curriculum_raw = mcp_client.invoke(
                        "get_curriculum_lookup",
                        {"program_id": program_id, "session_id": session_id},
                    )
                    curriculum_payload = _safe_json_loads(curriculum_raw)
                    curriculum_entries = _collect_curriculum_subject_entries(curriculum_payload)
                    curriculum_candidate = _pick_curriculum_subject_from_hints(subject_hints, curriculum_entries)
                except Exception as curriculum_err:
                    logger.debug("course_schedule curriculum candidate lookup failed: %s", curriculum_err)

            if is_deictic and memory_subject_code:
                logger.info("[route] Resolved deictic subject from memory: %s", memory_subject_code)
                if not any(
                    normalize_for_match(hint) == normalize_for_match(memory_subject_code)
                    for hint in subject_hints
                ):
                    subject_hints = [memory_subject_code, *subject_hints]
            if not subject_hints:
                if memory_subject_code:
                    subject_hints = [memory_subject_code]
                else:
                    subject_hints = [query]

            alias_payloads: List[Dict[str, Any]] = []
            subject_codes: List[str] = []
            for hint in subject_hints:
                hint_norm = normalize_for_match(hint)
                if not hint_norm or hint_norm in {"ai day", "co ai day", "co nhung ai day", "nhung ai", "nao"}:
                    continue
                alias_attempts: List[Dict[str, Any]] = []
                alias_args: Dict[str, Any] = {"query": hint, "session_id": session_id}
                if program_id:
                    alias_args["program_id"] = program_id
                alias_raw = mcp_client.invoke("resolve_course_alias", alias_args)
                alias_attempts.append(_safe_json_loads(alias_raw))

                # For direct schedule lookup, retry alias resolution without program constraint
                # when the scoped lookup cannot resolve a subject code.
                scoped_match = alias_attempts[0].get("matched_subject") if alias_attempts else {}
                scoped_code = str((scoped_match or {}).get("subject_code") or "").strip()
                if intent == "course_schedule" and program_id and not scoped_code:
                    alias_retry_raw = mcp_client.invoke(
                        "resolve_course_alias",
                        {"query": hint, "session_id": session_id},
                    )
                    alias_attempts.append(_safe_json_loads(alias_retry_raw))

                alias_payload = max(
                    alias_attempts,
                    key=lambda item: float((item or {}).get("confidence") or 0.0),
                )
                alias_payloads.append(alias_payload)
                matched_subject = alias_payload.get("matched_subject") or {}
                subject_code = str(matched_subject.get("subject_code") or "").strip()
                alias_confidence = float(alias_payload.get("confidence") or 0.0)
                min_confidence = 0.55 if len(subject_hints) > 1 else (0.45 if intent == "course_schedule" else 0.5)
                if subject_code and alias_confidence >= min_confidence:
                    if subject_code not in subject_codes:
                        subject_codes.append(subject_code)

            # Only deictic follow-up may retry from memory when explicit alias resolution fails.
            if not subject_codes and is_deictic and memory_subject_code:
                logger.info("[route] Alias retry with deictic memory subject: %s", memory_subject_code)
                alias_raw = mcp_client.invoke(
                    "resolve_course_alias",
                    {"query": memory_subject_code, "program_id": program_id, "session_id": session_id},
                )
                alias_payload = _safe_json_loads(alias_raw)
                alias_payloads.append(alias_payload)
                matched_subject = alias_payload.get("matched_subject") or {}
                subject_code = str(matched_subject.get("subject_code") or "").strip()
                if subject_code:
                    subject_codes = [subject_code]

            if intent == "course_schedule" and curriculum_candidate and len(subject_hints) <= 1 and not has_explicit_course_code:
                curriculum_code = str(curriculum_candidate.get("subject_code") or "").strip().upper()
                curriculum_score = float(curriculum_candidate.get("score") or 0.0)
                if curriculum_code and curriculum_score >= 0.86:
                    if not subject_codes or subject_codes[0] != curriculum_code:
                        logger.info(
                            "[route] course_schedule subject overridden by curriculum fallback. code=%s score=%.3f",
                            curriculum_code,
                            curriculum_score,
                        )
                    subject_codes = [curriculum_code]
                    if not any(
                        str(((item or {}).get("matched_subject") or {}).get("subject_code") or "").strip().upper()
                        == curriculum_code
                        for item in alias_payloads
                        if isinstance(item, dict)
                    ):
                        alias_payloads.insert(
                            0,
                            {
                                "matched_subject": {
                                    "subject_code": curriculum_code,
                                    "subject_name_vi": str(curriculum_candidate.get("subject_name") or "").strip(),
                                    "subject_name_en": "",
                                },
                                "confidence": curriculum_score,
                            },
                        )

            if subject_codes:
                if intent == "teacher_by_subject":
                    merged_rows: List[Dict[str, Any]] = []
                    merged_teachers: Set[str] = set()
                    merged_source_files: Set[str] = set()
                    no_data_subjects: List[Dict[str, str]] = []
                    alias_meta_by_code: Dict[str, Dict[str, str]] = {}
                    for alias_item in alias_payloads:
                        if not isinstance(alias_item, dict):
                            continue
                        matched_item = alias_item.get("matched_subject")
                        if not isinstance(matched_item, dict):
                            continue
                        item_code = str(matched_item.get("subject_code") or "").strip().upper()
                        if not item_code:
                            continue
                        alias_meta_by_code[item_code] = {
                            "subject_code": item_code,
                            "subject_name_vi": str(matched_item.get("subject_name_vi") or "").strip(),
                            "subject_name_en": str(matched_item.get("subject_name_en") or "").strip(),
                        }
                    for code in subject_codes:
                        structured_raw = mcp_client.invoke(
                            "get_teachers_by_subject",
                            {"subject_code": code, "session_id": session_id, "semester": semester_code},
                        )
                        structured_payload = _safe_json_loads(structured_raw)
                        rows = structured_payload.get("rows") if isinstance(structured_payload.get("rows"), list) else []
                        teachers = structured_payload.get("teachers") if isinstance(structured_payload.get("teachers"), list) else []
                        source_files = (
                            structured_payload.get("source_files")
                            if isinstance(structured_payload.get("source_files"), list)
                            else []
                        )
                        merged_rows.extend([r for r in rows if isinstance(r, dict)])
                        merged_teachers.update(str(t).strip() for t in teachers if str(t).strip())
                        merged_source_files.update(str(s).strip() for s in source_files if str(s).strip())
                        if not rows and not teachers:
                            meta = alias_meta_by_code.get(code) or {"subject_code": code, "subject_name_vi": "", "subject_name_en": ""}
                            no_data_subjects.append(meta)

                    merged_rows = _dedupe_schedule_rows(merged_rows)
                    if merged_rows or merged_teachers or no_data_subjects:
                        primary_alias = alias_payloads[0] if alias_payloads else {}
                        if merged_rows or merged_teachers:
                            coverage_note = (
                                f"Tìm thấy dữ liệu cho {len(subject_codes)} môn."
                                if len(subject_codes) > 1
                                else "Tìm thấy dữ liệu giảng viên cho môn học."
                            )
                            if no_data_subjects:
                                coverage_note += f" Chưa có dữ liệu lịch/giảng viên cho {len(no_data_subjects)} môn."
                        else:
                            coverage_note = "Chưa có dữ liệu lịch/giảng viên cho các môn được hỏi."
                        structured_payload = {
                            "rows": merged_rows,
                            "teachers": sorted(merged_teachers),
                            "source_files": sorted(merged_source_files),
                            "matched_subject": (
                                (primary_alias or {}).get("matched_subject") if len(subject_codes) == 1 else {}
                            ),
                            "matched_subjects": [
                                (a.get("matched_subject") or {})
                                for a in alias_payloads
                                if isinstance(a, dict) and isinstance(a.get("matched_subject"), dict)
                            ],
                            "alias": primary_alias,
                            "no_data_subjects": no_data_subjects,
                            "coverage_note": coverage_note,
                        }
                        return _payload(
                            "structured_schedule",
                            structured_payload,
                            "get_teachers_by_subject",
                            "structured",
                        )
                else:
                    merged_rows: List[Dict[str, Any]] = []
                    merged_source_files: Set[str] = set()
                    for code in subject_codes:
                        structured_raw = mcp_client.invoke(
                            "get_schedule_rows",
                            {"subject_code": code, "session_id": session_id, "semester": semester_code},
                        )
                        structured_payload = _safe_json_loads(structured_raw)
                        rows = structured_payload.get("rows") if isinstance(structured_payload.get("rows"), list) else []
                        source_files = (
                            structured_payload.get("source_files")
                            if isinstance(structured_payload.get("source_files"), list)
                            else []
                        )
                        merged_rows.extend([r for r in rows if isinstance(r, dict)])
                        merged_source_files.update(str(s).strip() for s in source_files if str(s).strip())

                    merged_rows = _dedupe_schedule_rows(merged_rows)
                    if merged_rows:
                        primary_alias = alias_payloads[0] if alias_payloads else {}
                        structured_payload = {
                            "rows": merged_rows,
                            "source_files": sorted(merged_source_files),
                            "matched_subject": (
                                (primary_alias or {}).get("matched_subject") if len(subject_codes) == 1 else {}
                            ),
                            "matched_subjects": [
                                (a.get("matched_subject") or {})
                                for a in alias_payloads
                                if isinstance(a, dict) and isinstance(a.get("matched_subject"), dict)
                            ],
                            "alias": primary_alias,
                            "coverage_note": (
                                f"Tìm thấy lịch học cho {len(subject_codes)} môn."
                                if len(subject_codes) > 1
                                else "Tìm thấy lịch học cho môn."
                            ),
                        }
                        return _payload("structured_schedule", structured_payload, "get_schedule_rows", "structured")

                    primary_alias = alias_payloads[0] if alias_payloads else {}
                    no_data_subjects: List[Dict[str, str]] = []
                    for code in subject_codes:
                        code_upper = str(code or "").strip().upper()
                        if not code_upper:
                            continue
                        matched = {}
                        for alias_item in alias_payloads:
                            if not isinstance(alias_item, dict):
                                continue
                            matched_item = alias_item.get("matched_subject")
                            if not isinstance(matched_item, dict):
                                continue
                            matched_code = str(matched_item.get("subject_code") or "").strip().upper()
                            if matched_code == code_upper:
                                matched = matched_item
                                break
                        no_data_subjects.append(
                            {
                                "subject_code": code_upper,
                                "subject_name_vi": str(matched.get("subject_name_vi") or "").strip(),
                                "subject_name_en": str(matched.get("subject_name_en") or "").strip(),
                            }
                        )
                    structured_payload = {
                        "rows": [],
                        "matched_subject": (
                            (primary_alias or {}).get("matched_subject") if len(subject_codes) == 1 else {}
                        ),
                        "matched_subjects": [
                            (a.get("matched_subject") or {})
                            for a in alias_payloads
                            if isinstance(a, dict) and isinstance(a.get("matched_subject"), dict)
                        ],
                        "alias": primary_alias,
                        "no_data_subjects": no_data_subjects,
                        "coverage_note": (
                            "Chưa thấy dữ liệu lịch cho môn được hỏi trong thời khóa biểu hiện có."
                            if len(subject_codes) == 1
                            else "Chưa thấy dữ liệu lịch cho các môn được hỏi trong thời khóa biểu hiện có."
                        ),
                    }
                    return _payload("structured_schedule", structured_payload, "get_schedule_rows", "structured_no_data")

                # Fallback stage 1: deterministic get_schedule
                try:
                    sched_raw = mcp_client.invoke(
                        "get_schedule",
                        {"subject_codes": subject_codes, "session_id": session_id},
                    )
                    sched_text = _context_to_text(sched_raw)
                    if sched_text and "Not found in TKB" not in sched_text:
                        return _payload("schedule_lookup", sched_raw, "get_schedule", "fallback_get_schedule")
                except Exception as schedule_err:
                    logger.warning("structured fallback get_schedule failed: %s", schedule_err)

        # Fallback stage 2: vector retrieve
        retrieved = mcp_client.invoke(
            "retrieve_chunks",
            {"question": query, "top_k": 25, "file_ids": [], "session_id": session_id},
        )
        if retrieved:
            return _payload("vector_store", retrieved, "retrieve_chunks", "fallback_retrieve")
    except Exception as structured_err:
        logger.warning(
            "Structured route failed intent=%s confidence=%.2f session=%s err=%s",
            intent,
            confidence,
            session_id,
            structured_err,
        )
    return None


def _render_course_offering_status_answer(context: str) -> str:
    payload = _safe_json_loads(context)
    matched_subject = payload.get("matched_subject") if isinstance(payload.get("matched_subject"), dict) else {}
    subject_code = str(payload.get("subject_code") or matched_subject.get("subject_code") or "").strip().upper()
    subject_name_vi = str(matched_subject.get("subject_name_vi") or "").strip()
    subject_name_en = str(matched_subject.get("subject_name_en") or "").strip()
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    is_opened = bool(payload.get("is_opened")) or bool(rows)
    in_curriculum = bool(payload.get("in_curriculum"))
    group_code = str(payload.get("curriculum_group_code") or "").strip()
    group_name = str(payload.get("curriculum_group_name") or "").strip()

    subject_title = subject_code
    if subject_name_vi and subject_name_en:
        subject_title = f"{subject_code} - {subject_name_vi} ({subject_name_en})"
    elif subject_name_vi:
        subject_title = f"{subject_code} - {subject_name_vi}" if subject_code else subject_name_vi

    lines: List[str] = [subject_title] if subject_title else ["Môn học cần tra cứu"]
    if is_opened:
        lines.append("- Tình trạng mở lớp kỳ này: Có mở lớp.")
    else:
        lines.append("- Tình trạng mở lớp kỳ này: Không mở lớp.")

    if in_curriculum:
        if group_code and group_name:
            lines.append(f"- Trong khung CTĐT: Có (nhóm {group_code} - {group_name}).")
        elif group_code:
            lines.append(f"- Trong khung CTĐT: Có (nhóm {group_code}).")
        else:
            lines.append("- Trong khung CTĐT: Có.")
    else:
        lines.append("- Trong khung CTĐT: Chưa thấy môn này trong chương trình đã chọn.")

    if is_opened and rows:
        events = _collapse_schedule_rows_for_display(rows)
        lines.append("")
        lines.append("Lịch học tìm thấy:")
        for event in events[:8]:
            class_code = str(event.get("class_code") or "").strip()
            day = str(event.get("day_of_week") or "").strip() or "Chưa rõ thứ"
            slot = str(event.get("slot") or "").strip()
            room = str(event.get("room") or "").strip()
            details = [
                part
                for part in [day, f"Ca {slot}" if slot else "", f"phòng {room}" if room else ""]
                if part
            ]
            if class_code:
                lines.append(f"- {class_code}: {', '.join(details)}")
            else:
                lines.append(f"- {', '.join(details)}")

    return "\n".join(lines).strip()


def _render_time_slot_lookup_answer(query: str, context: str) -> str:
    payload = _safe_json_loads(context)
    if not isinstance(payload, dict):
        return "Không tìm thấy thông tin khung giờ ca học trong dữ liệu hiện có."

    slot = str(payload.get("slot") or "").strip() or (_extract_time_slot_number_from_query(query) or "")
    period = str(payload.get("period") or "").strip()
    time_range = str(payload.get("time_range") or "").strip()
    source_file = str(payload.get("source_file") or "").strip()
    coverage_note = str(payload.get("coverage_note") or "").strip()

    start_time = ""
    end_time = ""
    time_match = re.search(r"(\d{1,2}:\d{2})\s*[–-]\s*(\d{1,2}:\d{2})", time_range)
    if time_match:
        start_time = time_match.group(1)
        end_time = time_match.group(2)

    if slot and start_time and end_time:
        period_suffix = f" ({period})" if period else ""
        answer = f"Ca {slot} bắt đầu từ {start_time} và kết thúc lúc {end_time}{period_suffix}."
    elif slot and time_range:
        period_suffix = f" ({period})" if period else ""
        answer = f"Khung giờ của Ca {slot} là {time_range}{period_suffix}."
    else:
        return coverage_note or "Không tìm thấy thông tin chi tiết khung giờ ca học trong dữ liệu hiện có."

    if source_file == "DEFAULT_UET_TIME_SLOTS":
        answer += " (Đang dùng mốc giờ mặc định do chưa đọc được bảng giờ chi tiết trong tài liệu TKB hiện có.)"
    return answer


def _render_semester_code_lookup_answer(context: str) -> str:
    payload = _safe_json_loads(context)
    if not isinstance(payload, dict):
        return "Chưa xác định được mã học kỳ từ dữ liệu hiện có."

    semester_code = str(payload.get("semester_code") or "").strip()
    inference_source = str(payload.get("inference_source") or "").strip()
    coverage_note = str(payload.get("coverage_note") or "").strip()

    if not semester_code:
        return coverage_note or "Chưa xác định được mã học kỳ từ dữ liệu hiện có."

    source_note = ""
    if inference_source == "schedule_rows":
        source_note = " (suy luận từ dữ liệu thời khóa biểu đã ingest)"
    elif inference_source == "source_file":
        source_note = " (suy luận từ tên file thời khóa biểu)"
    elif inference_source == "calendar_fallback":
        source_note = " (suy luận theo học kỳ hiện tại)"
    return f"Mã kỳ học hiện tại theo thời khóa biểu là `{semester_code}`{source_note}."


def _render_structured_schedule_answer(query: str, context: str) -> str:
    payload = _safe_json_loads(context)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    teachers = payload.get("teachers") if isinstance(payload.get("teachers"), list) else []
    matched_subject = payload.get("matched_subject") if isinstance(payload.get("matched_subject"), dict) else {}
    matched_subjects = payload.get("matched_subjects") if isinstance(payload.get("matched_subjects"), list) else []
    matched_teacher = payload.get("matched_teacher") if isinstance(payload.get("matched_teacher"), dict) else {}
    no_data_subjects = payload.get("no_data_subjects") if isinstance(payload.get("no_data_subjects"), list) else []

    def _is_ascii_english_token(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", str(token or "").strip()))

    def _split_bilingual_name(name_vi_or_mix: str, name_en: str = "") -> Tuple[str, str]:
        vi = " ".join(str(name_vi_or_mix or "").replace("\n", " ").split())
        en = " ".join(str(name_en or "").replace("\n", " ").split())
        if en:
            return vi, en
        if not vi:
            return "", ""

        m = re.match(r"^(?P<vi>.+?)\s*\((?P<en>[^()]+)\)\s*$", vi)
        if m:
            return m.group("vi").strip(), m.group("en").strip()

        tokens = vi.split()
        if len(tokens) < 2:
            return vi, ""

        english_tokens = {
            "application", "applications", "development", "internet", "things",
            "iotapplication", "iot", "web", "mobile", "cryptography", "information",
            "security", "computer", "graphics", "vision", "natural", "language",
            "processing", "human", "machine", "interaction", "bigdata", "techniques",
            "technologies", "special", "problems", "science", "data", "learning",
            "deep", "optimization", "numerical", "methods", "entrepreneurship",
            "political", "economy", "marx", "lenin", "socialism", "scientific",
            "revolutionary", "guidelines", "vietnam", "communist", "party", "of", "and",
            "software", "testing", "quality", "assurance",
        }
        vietnamese_ascii_tokens = {
            "phat", "trien", "ung", "dung", "mat", "ma", "an", "toan", "thong", "tin",
            "xu", "ly", "ngon", "ngu", "tu", "nhien", "do", "hoa", "may", "tinh", "cac",
            "chuyen", "de", "va", "ky", "thuat", "cong", "nghe", "du", "lieu", "lon",
            "tuong", "tac", "nguoi", "khoa", "hoc", "kiem", "thu", "dam", "bao",
            "chat", "luong", "phan", "mem", "di", "dong", "toi", "uu", "trong",
        }

        def _token_is_english_like(token_norm: str) -> bool:
            norm = str(token_norm or "").strip()
            if not norm:
                return False
            if norm in english_tokens:
                return True
            parts = [part for part in re.split(r"[^a-z0-9]+", norm) if part]
            return bool(parts) and all(part in english_tokens for part in parts)

        for idx in range(1, len(tokens)):
            left_tokens = tokens[:idx]
            right_tokens = tokens[idx:]
            if not left_tokens or not right_tokens:
                continue
            right_norm = [normalize_for_match(tok) for tok in right_tokens]
            left_norm = [normalize_for_match(tok) for tok in left_tokens]
            english_hits = sum(1 for tok in right_norm if _token_is_english_like(tok))
            vn_hits = sum(1 for tok in right_norm if tok in vietnamese_ascii_tokens)
            if english_hits < 1 or english_hits <= vn_hits:
                continue
            first_norm = normalize_for_match(right_tokens[0] or "")
            first_is_english = _token_is_english_like(first_norm)
            if not first_is_english:
                continue
            left_has_vn_signal = any((not str(tok).isascii()) for tok in left_tokens) or any(
                tok in vietnamese_ascii_tokens for tok in left_norm
            )
            if not left_has_vn_signal:
                continue
            if len(right_tokens) >= 2 and normalize_for_match(right_tokens[0]) == normalize_for_match(right_tokens[1]):
                # Common OCR/bilingual pattern: "... Web Web Application Development".
                # Keep the first token on VI side and start EN from the second duplicate token.
                left_tokens = left_tokens + [right_tokens[0]]
                right_tokens = right_tokens[1:]
            if not right_tokens:
                continue
            return " ".join(left_tokens).strip(), " ".join(right_tokens).strip()
        return vi, ""

    def _repair_subject_name(code: str, name: str) -> str:
        cleaned = " ".join(str(name or "").replace("\n", " ").split())
        if not cleaned:
            return ""
        normalized = normalize_for_match(cleaned)
        # OCR from the timetable sometimes splits "Thực" into noisy single-letter fragments.
        if str(code or "").strip().upper() == "AIT3004" and (
            "t ta h o uc hanh" in normalized
            or "h o uc hanh" in normalized
            or "uc hanh phat trien he thong tri tue nhan" in normalized
        ):
            return "Thực hành phát triển hệ thống Trí tuệ nhân tạo"
        return cleaned

    def _format_subject_title(code: str, name_vi_or_mix: str = "", name_en: str = "") -> str:
        code_clean = str(code or "").strip().upper()
        vi, en = _split_bilingual_name(_repair_subject_name(code_clean, name_vi_or_mix), name_en)
        if vi and en:
            name_text = f"{vi} ({en})"
        else:
            name_text = vi or en
        if code_clean and name_text:
            return f"{code_clean} - {name_text}"
        return code_clean or name_text

    def _clean_teacher_name(name: str) -> str:
        raw = " ".join(str(name or "").split())
        if not raw:
            return ""
        tokens = raw.split()
        for size in range(1, (len(tokens) // 2) + 1):
            if tokens[:size] == tokens[size : 2 * size]:
                tokens = tokens[:size] + tokens[2 * size :]
                break
        cleaned = " ".join(tokens).strip(" ,;")
        if not cleaned:
            return ""
        # Drop OCR-broken names like "t u u ầ ầ n n, t đ h ầ i u v".
        if re.search(r"(?:\b\S\b[\s,;]*){5,}", cleaned):
            return ""
        normalized_tokens = [tok for tok in normalize_for_match(cleaned).split() if tok]
        if len(normalized_tokens) >= 6:
            short_tokens = sum(1 for tok in normalized_tokens if len(tok) <= 1)
            if short_tokens / max(1, len(normalized_tokens)) >= 0.45:
                return ""
        alpha_count = len(re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ]", cleaned))
        if alpha_count < 3:
            return ""
        return cleaned

    def _extract_teacher_candidates(raw_teacher: str) -> List[str]:
        raw = " ".join(str(raw_teacher or "").split())
        if not raw:
            return []
        parts = re.split(r"[;|/]+", raw)
        candidates: List[str] = []
        seen: Set[str] = set()
        for part in parts:
            cleaned = _clean_teacher_name(part)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cleaned)
        return candidates

    def _subject_name_by_code() -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}

        def _upsert(code: str, name_vi: str, name_en: str = ""):
            if not code:
                return
            existing = mapping.get(code) or {"vi": "", "en": ""}
            if name_vi and not existing.get("vi"):
                existing["vi"] = name_vi
            if name_en and not existing.get("en"):
                existing["en"] = name_en
            mapping[code] = existing

        if matched_subject:
            code = str(matched_subject.get("subject_code") or "").strip().upper()
            name_vi = str(matched_subject.get("subject_name_vi") or "").strip()
            name_en = str(matched_subject.get("subject_name_en") or "").strip()
            _upsert(code, name_vi, name_en)
        for item in matched_subjects:
            if not isinstance(item, dict):
                continue
            code = str(item.get("subject_code") or "").strip().upper()
            name_vi = str(item.get("subject_name_vi") or "").strip()
            name_en = str(item.get("subject_name_en") or "").strip()
            _upsert(code, name_vi, name_en)
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("subject_code") or "").strip().upper()
            name_vi = str(row.get("subject_name_vi") or "").strip()
            name_en = str(row.get("subject_name_en") or "").strip()
            _upsert(code, name_vi, name_en)
        return mapping

    if teachers and matched_subject:
        code = str(matched_subject.get("subject_code") or "").strip()
        title = _format_subject_title(
            code=code,
            name_vi_or_mix=str(matched_subject.get("subject_name_vi") or "").strip(),
            name_en=str(matched_subject.get("subject_name_en") or "").strip(),
        )
        normalized_teachers: List[str] = []
        seen_teachers: Set[str] = set()
        for teacher in teachers:
            for candidate in _extract_teacher_candidates(str(teacher or "")):
                key = candidate.lower()
                if key in seen_teachers:
                    continue
                seen_teachers.add(key)
                normalized_teachers.append(candidate)
        if not normalized_teachers:
            return f"Môn {title} kỳ này chưa có tên giảng viên rõ ràng trong dữ liệu lịch."
        lines = [f"Môn {title} kỳ này có {len(normalized_teachers)} giảng viên:"]
        lines.extend([f"- {t}" for t in sorted(normalized_teachers)])
        return "\n".join(lines).strip()

    if matched_teacher and rows:
        teacher_query = str(matched_teacher.get("query") or "").strip()
        code_name_map = _subject_name_by_code()
        grouped: Dict[str, Dict[str, Set[Tuple[str, str, str]]]] = {}

        for row in rows[:200]:
            if not isinstance(row, dict):
                continue
            code = str(row.get("subject_code") or "").strip().upper()
            if not code:
                continue
            class_code = str(row.get("class_code") or "").strip() or code
            day = str(row.get("day_of_week") or "Chưa rõ").strip()
            slot = str(row.get("slot") or "").strip()
            room = str(row.get("room") or "").strip()
            grouped.setdefault(code, {}).setdefault(class_code, set()).add((day, slot, room))

        if not grouped:
            return f"Không tìm thấy lớp phù hợp cho giảng viên {teacher_query}."

        lines = [f"Các lớp của giảng viên {teacher_query}:"]
        for code in sorted(grouped.keys()):
            subject_meta = code_name_map.get(code) or {}
            title = _format_subject_title(
                code=code,
                name_vi_or_mix=str(subject_meta.get("vi") or ""),
                name_en=str(subject_meta.get("en") or ""),
            )
            lines.append("")
            lines.append(title)
            class_map = grouped.get(code) or {}
            for class_code in sorted(class_map.keys()):
                events = class_map[class_code]
                event_parts: List[str] = []
                for day, slot, room in sorted(
                    events,
                    key=lambda item: (
                        str(item[0] or ""),
                        int(item[1]) if str(item[1]).isdigit() else 99,
                        str(item[2] or ""),
                    ),
                ):
                    slot_text = f"Ca {slot}" if slot else "Chưa rõ ca"
                    room_text = f", phòng {room}" if room else ""
                    event_parts.append(f"{day}, {slot_text}{room_text}")
                if not event_parts:
                    event_parts = ["Chưa có dòng lịch chi tiết."]
                lines.append(f"- {class_code}: {'; '.join(event_parts)}")
        return "\n".join(lines).strip()

    if rows:
        code_name_map = _subject_name_by_code()
        grouped: Dict[str, Dict[str, Dict[Tuple[str, str, str], Set[str]]]] = {}
        for row in rows[:200]:
            if not isinstance(row, dict):
                continue
            code = str(row.get("subject_code") or "").strip().upper()
            if not code:
                continue
            class_code = str(row.get("class_code") or "").strip()
            if not class_code:
                class_code = code
            day = str(row.get("day_of_week") or "Chưa rõ").strip()
            slot = str(row.get("slot") or "").strip()
            room = str(row.get("room") or "").strip()
            event_key = (day, slot, room)
            grouped.setdefault(code, {}).setdefault(class_code, {}).setdefault(event_key, set())
            for teacher in _extract_teacher_candidates(str(row.get("teacher_name") or "")):
                grouped[code][class_code][event_key].add(teacher)

        lines = ["Lịch học theo từng môn:"]
        for code in sorted(grouped.keys()):
            subject_meta = code_name_map.get(code) or {}
            title = _format_subject_title(
                code=code,
                name_vi_or_mix=str(subject_meta.get("vi") or ""),
                name_en=str(subject_meta.get("en") or ""),
            )
            lines.append("")
            lines.append(title)
            class_map = grouped.get(code) or {}
            for class_code in sorted(class_map.keys()):
                events = class_map[class_code]
                event_parts: List[str] = []
                for (day, slot, room), event_teachers in sorted(
                    events.items(),
                    key=lambda item: (
                        str(item[0][0] or ""),
                        int(item[0][1]) if str(item[0][1]).isdigit() else 99,
                        str(item[0][2] or ""),
                    ),
                ):
                    slot_text = f"Ca {slot}" if slot else "Chưa rõ ca"
                    room_text = f", phòng {room}" if room else ""
                    teacher_text = ""
                    if event_teachers:
                        teacher_text = f", GV {', '.join(sorted(event_teachers))}"
                    event_parts.append(f"{day}, {slot_text}{room_text}{teacher_text}")
                if not event_parts:
                    event_parts = ["Chưa có dòng lịch chi tiết."]
                lines.append(f"- {class_code}: {'; '.join(event_parts)}")
        if no_data_subjects:
            lines.append("")
            lines.append("Chưa thấy dữ liệu lịch/giảng viên cho:")
            seen_missing: Set[str] = set()
            for item in no_data_subjects:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("subject_code") or "").strip().upper()
                key = code or str(item)
                if not key or key in seen_missing:
                    continue
                seen_missing.add(key)
                lines.append(
                    "- "
                    + _format_subject_title(
                        code=code,
                        name_vi_or_mix=str(item.get("subject_name_vi") or ""),
                        name_en=str(item.get("subject_name_en") or ""),
                    )
                )
        return "\n".join(lines).strip()

    if no_data_subjects:
        lines = ["Chưa thấy dữ liệu lịch/giảng viên cho các môn sau:"]
        seen_missing: Set[str] = set()
        for item in no_data_subjects:
            if not isinstance(item, dict):
                continue
            code = str(item.get("subject_code") or "").strip().upper()
            key = code or str(item)
            if not key or key in seen_missing:
                continue
            seen_missing.add(key)
            lines.append(
                "- "
                + _format_subject_title(
                    code=code,
                    name_vi_or_mix=str(item.get("subject_name_vi") or ""),
                    name_en=str(item.get("subject_name_en") or ""),
                )
            )
        if len(lines) > 1:
            return "\n".join(lines).strip()

    coverage_note = str(payload.get("coverage_note") or "").strip()
    if coverage_note:
        return coverage_note
    return "Không tìm thấy dữ liệu lịch phù hợp trong structured TKB."


def _render_electives_schedule_answer(context: str) -> str:
    payload = _safe_json_loads(context)
    opened = payload.get("opened") if isinstance(payload.get("opened"), list) else []
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    no_data_subjects = payload.get("no_data_subjects") if isinstance(payload.get("no_data_subjects"), list) else []
    if not opened:
        schedule_error = str(payload.get("schedule_error") or "").strip()
        if schedule_error:
            return f"Không lấy được danh sách môn tự chọn mở lớp: {schedule_error}"
        return "Không tìm thấy học phần tự chọn đang mở lớp trong kỳ này."

    def _is_ascii_english_token(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", str(token or "").strip()))

    def _split_bilingual_name(name: str) -> Tuple[str, str]:
        mixed = " ".join(str(name or "").replace("\n", " ").split())
        if not mixed:
            return "", ""
        m = re.match(r"^(?P<vi>.+?)\s*\((?P<en>[^()]+)\)\s*$", mixed)
        if m:
            return m.group("vi").strip(), m.group("en").strip()

        tokens = mixed.split()
        if len(tokens) < 2:
            return mixed, ""

        english_tokens = {
            "application", "applications", "development", "internet", "things",
            "iotapplication", "iot", "web", "mobile", "cryptography", "information",
            "security", "computer", "graphics", "vision", "natural", "language",
            "processing", "human", "machine", "interaction", "bigdata", "techniques",
            "technologies", "special", "problems", "science", "data", "learning",
            "deep", "optimization", "numerical", "methods", "entrepreneurship",
            "political", "economy", "marx", "lenin", "socialism", "scientific",
            "revolutionary", "guidelines", "vietnam", "communist", "party", "of", "and",
            "software", "testing", "quality", "assurance",
        }
        vietnamese_ascii_tokens = {
            "phat", "trien", "ung", "dung", "mat", "ma", "an", "toan", "thong", "tin",
            "xu", "ly", "ngon", "ngu", "tu", "nhien", "do", "hoa", "may", "tinh", "cac",
            "chuyen", "de", "va", "ky", "thuat", "cong", "nghe", "du", "lieu", "lon",
            "tuong", "tac", "nguoi", "khoa", "hoc", "kiem", "thu", "dam", "bao",
            "chat", "luong", "phan", "mem", "di", "dong", "toi", "uu", "trong",
        }

        def _token_is_english_like(token_norm: str) -> bool:
            norm = str(token_norm or "").strip()
            if not norm:
                return False
            if norm in english_tokens:
                return True
            parts = [part for part in re.split(r"[^a-z0-9]+", norm) if part]
            return bool(parts) and all(part in english_tokens for part in parts)

        for idx in range(1, len(tokens)):
            left_tokens = tokens[:idx]
            right_tokens = tokens[idx:]
            if not left_tokens or not right_tokens:
                continue
            right_norm = [normalize_for_match(tok) for tok in right_tokens]
            left_norm = [normalize_for_match(tok) for tok in left_tokens]
            english_hits = sum(1 for tok in right_norm if _token_is_english_like(tok))
            vn_hits = sum(1 for tok in right_norm if tok in vietnamese_ascii_tokens)
            if english_hits < 1 or english_hits <= vn_hits:
                continue
            first_norm = normalize_for_match(right_tokens[0] or "")
            first_is_english = _token_is_english_like(first_norm)
            if not first_is_english:
                continue
            left_has_vn_signal = any((not str(tok).isascii()) for tok in left_tokens) or any(
                tok in vietnamese_ascii_tokens for tok in left_norm
            )
            if not left_has_vn_signal:
                continue
            if len(right_tokens) >= 2 and normalize_for_match(right_tokens[0]) == normalize_for_match(right_tokens[1]):
                # Common OCR/bilingual pattern: "... Web Web Application Development".
                # Keep the first token on VI side and start EN from the second duplicate token.
                left_tokens = left_tokens + [right_tokens[0]]
                right_tokens = right_tokens[1:]
            if not right_tokens:
                continue
            return " ".join(left_tokens).strip(), " ".join(right_tokens).strip()
        return mixed, ""

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for subj in opened:
        if not isinstance(subj, dict):
            continue
        group_code = str(subj.get("group_code") or "").strip()
        group_name = " ".join(str(subj.get("group") or "").split())
        grouped.setdefault((group_code, group_name), []).append(subj)

    lines = [f"Kỳ này có {len(opened)} học phần tự chọn mở lớp theo chương trình đào tạo:"]
    for (group_code, group_name) in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        group_title = ""
        if group_code and group_name:
            group_title = f"{group_code} - {group_name}"
        else:
            group_title = group_code or group_name or "Nhóm khác"
        lines.append("")
        lines.append(f"{group_title}:")
        subjects = grouped[(group_code, group_name)]
        subjects.sort(key=lambda item: str(item.get("code") or ""))
        for subj in subjects:
            code = str(subj.get("code") or "").strip().upper()
            vi, en = _split_bilingual_name(str(subj.get("name") or ""))
            subject_text = f"{vi} ({en})" if en else vi
            credits = subj.get("credits")
            credit_text = f" - {credits} tín chỉ" if credits not in (None, "") else ""
            lines.append(f"- {code}: {subject_text}{credit_text}")

    if rows or no_data_subjects:
        lines.append("")
        lines.append(
            _render_structured_schedule_answer(
                query="electives_overview",
                context=json.dumps(
                    {
                        "rows": rows,
                        "no_data_subjects": no_data_subjects,
                        "coverage_note": str(payload.get("coverage_note") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
            )
        )

    return "\n".join(lines).strip()


def _render_electives_recommendation_answer(context: str) -> str:
    payload = _safe_json_loads(context)
    recommended = (
        payload.get("recommended_subjects")
        if isinstance(payload.get("recommended_subjects"), list)
        else []
    )
    if not recommended:
        coverage_note = str(payload.get("coverage_note") or "").strip()
        if coverage_note:
            return coverage_note
        return "Không tìm thấy môn tự chọn phù hợp với định hướng đã nêu."

    lines: List[str] = []
    focus = str(payload.get("focus") or "").strip()
    if focus:
        lines.append(f"Gợi ý môn tự chọn theo định hướng \"{focus}\":")
    else:
        lines.append("Gợi ý môn tự chọn phù hợp với định hướng của bạn:")

    reason_by_code = payload.get("reason_by_code") if isinstance(payload.get("reason_by_code"), dict) else {}
    for item in recommended:
        if not isinstance(item, dict):
            continue
        code = str(item.get("subject_code") or "").strip().upper()
        name_vi = str(item.get("subject_name_vi") or "").strip()
        name_en = str(item.get("subject_name_en") or "").strip()
        if code and name_vi and name_en:
            title = f"{code} - {name_vi} ({name_en})"
        elif code and name_vi:
            title = f"{code} - {name_vi}"
        elif code and name_en:
            title = f"{code} - {name_en}"
        else:
            title = code or name_vi or name_en or "Mon hoc"
        credits = item.get("credits")
        try:
            credit_text = f" - {int(float(credits))} tín chỉ" if credits is not None else ""
        except Exception:
            credit_text = f" - {credits} tín chỉ" if credits not in (None, "") else ""
        group_code = str(item.get("group_code") or "").strip()
        group_name = str(item.get("group_name") or "").strip()
        group_text = ""
        if group_code and group_name:
            group_text = f" [{group_code} - {group_name}]"
        elif group_code or group_name:
            group_text = f" [{group_code or group_name}]"
        reason_text = ""
        if code and code in reason_by_code and str(reason_by_code.get(code) or "").strip():
            reason_text = f" | Lý do: {str(reason_by_code.get(code) or '').strip()}"
        lines.append(f"- {title}{credit_text}{group_text}{reason_text}")

    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    no_data_subjects = payload.get("no_data_subjects") if isinstance(payload.get("no_data_subjects"), list) else []
    if rows or no_data_subjects:
        lines.append("")
        schedule_context = {
            "rows": rows,
            "no_data_subjects": no_data_subjects,
            "coverage_note": str(payload.get("coverage_note") or "").strip(),
        }
        lines.append(
            _render_structured_schedule_answer(
                query="electives_recommendation",
                context=json.dumps(schedule_context, ensure_ascii=False),
            )
        )
    else:
        coverage_note = str(payload.get("coverage_note") or "").strip()
        if coverage_note:
            lines.append("")
            lines.append(coverage_note)

    return "\n".join(lines).strip()


def _fallback_planner_payload(
    query: str,
    session_id: str,
    selected_files: List[str],
    program_id: Optional[str],
    structured_prefetch: Optional[Dict[str, Any]] = None,
    force_advisor: bool = False,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    memory_context = _load_memory_context_for_session(session_id=session_id, max_rows=10, user_id=user_id)

    if structured_prefetch and _context_to_text(structured_prefetch.get("context")) and not force_advisor:
        merged = dict(structured_prefetch)
        merged["memory"] = _context_to_text(merged.get("memory")) or memory_context
        route_meta = merged.get("route_meta")
        if isinstance(route_meta, dict):
            route_meta["fallback_stage"] = route_meta.get("fallback_stage") or "fallback_structured_prefetch"
            merged["route_meta"] = route_meta
        else:
            merged["route_meta"] = {
                "intent": "structured_prefetch",
                "confidence": 0.5,
                "tool_used": "prefetch",
                "fallback_stage": "fallback_structured_prefetch",
            }
        return merged

    lowered = (query or "").lower()
    academic_markers = (
        "tin chi",
        "tín chỉ",
        "gpa",
        "môn",
        "mon",
        "học kỳ",
        "hoc ky",
        "lịch",
        "lich",
        "đăng ký",
        "dang ky",
        "chương trình",
        "chuong trinh",
    )
    schedule_lookup_markers = (
        "ai dạy",
        "ai day",
        "giảng viên",
        "giang vien",
        "lich hoc",
        "lich mon",
        "lich cac mon",
        "trong tuan",
        "thứ mấy",
        "thu may",
        "hôm nào",
        "hom nao",
        "phòng",
        "phong",
        "ca nào",
        "ca nao",
        "tiết nào",
        "tiet nao",
    )

    is_schedule_lookup = _has_schedule_lookup_signal(query) or any(marker in lowered for marker in schedule_lookup_markers)
    prefer_advisor = any(marker in lowered for marker in academic_markers) and not is_schedule_lookup
    transcript_intensive = bool(selected_files) and _query_requires_transcript_files(query)
    prefer_global_retrieve = _query_prefers_global_resource_retrieval(query)
    retrieve_file_ids: List[str] = [] if prefer_global_retrieve else selected_files

    # Transcript-intensive advisory queries must not silently degrade to vector retrieval,
    # otherwise users receive misleading "khong du du lieu CTDT" answers.
    advisor_args = {"query": query, "file_ids": selected_files, "session_id": session_id, "program_id": program_id}
    if user_id:
        advisor_args["user_id"] = user_id
    if force_advisor or transcript_intensive:
        tool_chain = [
            (
                "consult_advisor",
                advisor_args,
                "academic_advisor",
            )
        ]
    elif prefer_advisor:
        tool_chain = [
            (
                "consult_advisor",
                advisor_args,
                "academic_advisor",
            ),
            (
                "retrieve_chunks",
                {"question": query, "top_k": 25, "file_ids": retrieve_file_ids, "session_id": session_id},
                "vector_store",
            ),
        ]
    else:
        tool_chain = [
            (
                "retrieve_chunks",
                {"question": query, "top_k": 25, "file_ids": retrieve_file_ids, "session_id": session_id},
                "vector_store",
            ),
            (
                "consult_advisor",
                advisor_args,
                "academic_advisor",
            ),
        ]

    for tool_name, args, source in tool_chain:
        tool_timeout = MCP_TOOL_TIMEOUTS.get(tool_name)
        if transcript_intensive and tool_name == "consult_advisor":
            tool_timeout = MCP_TOOL_TIMEOUTS_TRANSCRIPT.get("consult_advisor")
        try:
            tool_result = _invoke_mcp_tool(tool_name, args, timeout_seconds=tool_timeout)
            return {
                "source": source,
                "context": _context_to_text(tool_result),
                "memory": memory_context,
                "chunk_index": None,
                "route_meta": {
                    "intent": "planner_fallback",
                    "confidence": 0.0,
                    "tool_used": tool_name,
                    "fallback_stage": "fallback_tool_chain",
                },
            }
        except Exception as tool_err:
            logger.warning(
                "Planner fallback %s failed for session %s (timeout=%s): %s",
                tool_name,
                session_id,
                tool_timeout,
                tool_err,
            )
            if tool_name == "consult_advisor" and len(tool_chain) == 1:
                return {
                    "source": "error",
                    "context": (
                        "Khong the hoan tat phan tich bang diem/CTDT trong thoi gian cho. "
                        "Vui long thu lai sau, hoac tang ASK_CONSULT_ADVISOR_TIMEOUT_SEC_TRANSCRIPT."
                    ),
                    "memory": memory_context,
                    "chunk_index": None,
                    "route_meta": {
                        "intent": "planner_fallback",
                        "confidence": 0.0,
                        "tool_used": tool_name,
                        "fallback_stage": "fallback_advisor_only_error",
                    },
                }

    return {
        "source": "error",
        "context": "He thong khong tao duoc ke hoach fallback. Vui long thu lai.",
        "memory": memory_context,
        "chunk_index": None,
        "route_meta": {
            "intent": "planner_fallback",
            "confidence": 0.0,
            "tool_used": "none",
            "fallback_stage": "fallback_tool_chain_error",
        },
    }


def _query_requires_transcript_files(query: str) -> bool:
    norm_q = normalize_for_match(query or "")
    if not norm_q:
        return False
    if _has_schedule_lookup_signal(query):
        transcript_intensive_markers = (
            "con thieu",
            "thieu mon",
            "tin chi",
            "gpa",
            "chuong trinh dao tao",
            "ctdt",
            "tot nghiep",
            "lap lich",
        )
        if not any(marker in norm_q for marker in transcript_intensive_markers):
            return False
    markers = (
        "bang diem",
        "tin chi",
        "gpa",
        "lap lich",
        "mon con thieu",
        "con thieu mon",
        "con thieu nhung mon",
        "con thieu",
        "thieu mon",
        "thieu nhung mon",
        "hoc ky sau",
        "chuong trinh dao tao",
        "ctdt",
        "tot nghiep",
    )
    return any(marker in norm_q for marker in markers)


def _query_prefers_global_resource_retrieval(query: str) -> bool:
    """
    Return True for policy/handbook and generic time-slot questions that should
    prioritize global/session resources over selected transcript files.
    """
    norm_q = normalize_for_match(query or "")
    if not norm_q:
        return False
    if _query_requires_transcript_files(query):
        return False
    if _query_targets_time_slot_definition(query):
        return True

    policy_markers = (
        "so tay hoc vu",
        "quy che",
        "quy dinh",
        "ngoai ngu",
        "chung chi",
        "ielts",
        "toeic",
        "toefl",
        "vstep",
        "aptis",
        "cambridge",
        "dieu kien ra truong",
        "chuan dau ra",
        "mien giam",
    )
    return any(marker in norm_q for marker in policy_markers)


def _query_requires_advisor_priority(query: str) -> bool:
    """
    Return True for transcript-intensive advisory questions that should bypass
    high-confidence structured schedule routes and go directly to consult_advisor.
    """
    norm_q = normalize_for_match(query or "")
    if not norm_q:
        return False

    missing_markers = (
        "mon con thieu",
        "con thieu mon",
        "con thieu nhung mon",
        "con thieu",
        "thieu mon",
        "thieu nhung mon",
        "tin chi con thieu",
        "bao nhieu tin chi",
        "bao nhieu tc",
    )
    planning_markers = (
        "gpa",
        "muc tieu",
        "lo trinh",
        "tot nghiep",
        "hoc ky sau",
        "ke hoach hoc",
    )
    has_missing = any(marker in norm_q for marker in missing_markers)
    has_planning = any(marker in norm_q for marker in planning_markers)
    has_credit_by_curriculum = (
        "tin chi" in norm_q
        and any(marker in norm_q for marker in ("chuong trinh dao tao", "ctdt", "khung ctdt"))
    )
    return has_missing or has_planning or has_credit_by_curriculum


def _normalize_program_list(raw: Any) -> List[Dict[str, Any]]:
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    programs = payload.get("programs")
    if not isinstance(programs, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for item in programs:
        if not isinstance(item, dict):
            continue
        program_id = str(item.get("id") or "").strip()
        if not program_id or program_id in seen_ids:
            continue
        seen_ids.add(program_id)
        name = str(item.get("name") or "").strip()
        display_name = str(item.get("display_name") or name or program_id).strip()
        normalized.append(
            {
                "id": program_id,
                "name": name or display_name,
                "year": item.get("year"),
                "year_end": item.get("year_end"),
                "qh_label": item.get("qh_label"),
                "group_name": item.get("group_name") or name or display_name,
                "display_name": display_name,
            }
        )
    return normalized

def _fetch_available_programs(refresh: bool = False) -> List[Dict[str, Any]]:
    result = mcp_client.invoke("get_available_programs", {"refresh": refresh})
    return _normalize_program_list(result)

def _program_selection_response(programs: List[Dict[str, Any]], answer: Optional[str] = None) -> Dict[str, Any]:
    default_answer = (
        "Vui lòng chọn chương trình đào tạo và khóa tuyển sinh (QH) trước khi đặt câu hỏi để hệ thống tư vấn chính xác."
    )
    no_program_answer = (
        "Hiện chưa tìm thấy chương trình đào tạo nào trong hệ thống. Vui lòng cập nhật tài nguyên HTML CTĐT rồi thử lại."
    )
    return {
        "answer": answer or (default_answer if programs else no_program_answer),
        "requires_program_selection": True,
        "programs": programs,
        "selected_program_id": None,
    }

class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False
    session_id: str = "user_session_1"
    file_ids: List[str] | None = None
    program_id: str | None = None

class HistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str

class SessionRequest(BaseModel):
    session_id: str


class ChatSessionCreateRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None
    selected_program_id: Optional[str] = None
    selected_file_ids: List[str] | None = None


class ChatSessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    selected_program_id: Optional[str] = None
    selected_file_ids: List[str] | None = None


class ChatMigrationMessage(BaseModel):
    role: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None
    text: Optional[str] = None
    citations: List[Dict[str, Any]] | None = None


class ChatMigrationSession(BaseModel):
    session_id: str
    title: Optional[str] = None
    selected_program_id: Optional[str] = None
    selected_file_ids: List[str] | None = None
    messages: List[ChatMigrationMessage] | None = None


class ChatMigrationRequest(BaseModel):
    sessions: List[ChatMigrationSession] | None = None


class UrlRequest(BaseModel):
    url: str
    session_id: Optional[str] = None


class GoogleAuthStartRequest(BaseModel):
    session_id: Optional[str] = None
    redirect_uri: Optional[str] = None


class MailConnectStartRequest(BaseModel):
    session_id: str
    redirect_uri: Optional[str] = None


class MailConnectCallbackRequest(BaseModel):
    session_id: str
    state: str
    code: str
    redirect_uri: Optional[str] = None


class MailSessionRequest(BaseModel):
    session_id: str


class MailWhitelistRequest(BaseModel):
    session_id: str
    senders: List[str]


class MailCandidateActionRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None

# --- Resource Endpoints ---

def _is_allowed_extension(filename: Optional[str], allowed_exts: Set[str]) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in allowed_exts


def _http_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


def _validate_batch_size(files: List[UploadFile]) -> None:
    if len(files) > MAX_BATCH_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Chi cho phep upload toi da {MAX_BATCH_UPLOAD_FILES} file moi lan")


def _validate_upload_file(
    upload_file: UploadFile,
    allowed_exts: Set[str],
    allowed_mimes: Set[str],
) -> str:
    original_name = Path(upload_file.filename or "").name or "unnamed"
    if not _is_allowed_extension(original_name, allowed_exts):
        raise HTTPException(status_code=400, detail="invalid extension")

    content_type = str(upload_file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in allowed_mimes and content_type != "application/octet-stream":
        raise HTTPException(status_code=400, detail=f"invalid content type: {content_type}")
    return original_name


def _copy_upload_to_path(upload_file: UploadFile, target_path: Path, max_bytes: int) -> int:
    bytes_written = 0
    try:
        with open(target_path, "wb") as buffer:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File vuot qua gioi han {max_bytes // (1024 * 1024)}MB",
                    )
                buffer.write(chunk)
    except Exception:
        try:
            if target_path.exists():
                target_path.unlink()
        finally:
            raise
    return bytes_written


def _save_resource_batch(files: List[UploadFile], target_dir: Path, allowed_exts: Set[str]) -> Dict[str, Any]:
    uploaded: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    _validate_batch_size(files)
    allowed_mimes = {"application/pdf"} if allowed_exts == {".pdf"} else {"text/html", "application/xhtml+xml"}

    for upload_file in files:
        try:
            original_name = _validate_upload_file(upload_file, allowed_exts, allowed_mimes)
            target_path = target_dir / original_name
            _copy_upload_to_path(upload_file, target_path, MAX_RESOURCE_UPLOAD_BYTES)
            uploaded.append({"name": original_name})
        except Exception as e:
            name = Path(upload_file.filename or "").name or "unnamed"
            errors.append({"name": name, "error": _http_error_message(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "uploaded_count": len(uploaded),
        "error_count": len(errors),
    }


def _save_resource_mixed_batch(
    files: List[UploadFile],
    pdf_dir: Path,
    html_dir: Path,
) -> Dict[str, Any]:
    uploaded: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    _validate_batch_size(files)

    for upload_file in files:
        original_name = Path(upload_file.filename or "").name or "unnamed"
        suffix = Path(original_name).suffix.lower()
        if suffix == ".pdf":
            target_dir = pdf_dir
            file_type = "pdf"
            allowed_exts = {".pdf"}
            allowed_mimes = {"application/pdf"}
        elif suffix in {".html", ".htm"}:
            target_dir = html_dir
            file_type = "html"
            allowed_exts = {".html", ".htm"}
            allowed_mimes = {"text/html", "application/xhtml+xml"}
        else:
            errors.append({"name": original_name, "error": "invalid extension"})
            continue

        try:
            original_name = _validate_upload_file(upload_file, allowed_exts, allowed_mimes)
            target_path = target_dir / original_name
            _copy_upload_to_path(upload_file, target_path, MAX_RESOURCE_UPLOAD_BYTES)
            uploaded.append({"name": original_name, "type": file_type})
        except Exception as e:
            errors.append({"name": original_name, "error": _http_error_message(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "uploaded_count": len(uploaded),
        "error_count": len(errors),
    }

@app.get("/api/resources")
async def get_resources(request: Request, session_id: Optional[str] = Query(default=None)):
    normalized_session = _normalize_session_id(session_id) if session_id else None
    user = _current_user_from_request(request)
    user_id = str(user.get("id") or "") if user else None
    return resource_loader.get_resources(session_id=normalized_session, user_id=user_id)

@app.get("/api/programs")
async def get_programs(refresh: bool = False):
    try:
        programs = _fetch_available_programs(refresh=refresh)
        return {"programs": programs, "count": len(programs)}
    except Exception as e:
        logger.error("Loi khi lay danh sach chuong trinh dao tao: %s", e)
        raise HTTPException(status_code=500, detail="Khong the lay danh sach chuong trinh dao tao.")

@app.post("/api/resources/pdf")
async def upload_resource_pdf(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Form(default=None)):
    try:
        file_name = _validate_upload_file(file, {".pdf"}, {"application/pdf"})
        normalized_session = _normalize_session_id(session_id) if session_id else None
        user = _current_user_from_request(request)
        user_id = str(user.get("id") or "") if user else None
        pdf_dir, _, _ = resource_loader._scope_dirs(session_id=normalized_session, user_id=user_id)
        # Save directly to resource dir
        target_path = pdf_dir / file_name
        _copy_upload_to_path(file, target_path, MAX_RESOURCE_UPLOAD_BYTES)
            
        # Notify MCP Server to scan
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {
            "message": "PDF added to resources successfully",
            "name": file_name,
            "session_id": normalized_session,
            "user_id": user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding PDF resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resources/html")
async def upload_resource_html(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Form(default=None)):
    try:
        file_name = _validate_upload_file(file, {".html", ".htm"}, {"text/html", "application/xhtml+xml"})
        normalized_session = _normalize_session_id(session_id) if session_id else None
        user = _current_user_from_request(request)
        user_id = str(user.get("id") or "") if user else None
        _, html_dir, _ = resource_loader._scope_dirs(session_id=normalized_session, user_id=user_id)
        # Save directly to resource dir
        target_path = html_dir / file_name
        _copy_upload_to_path(file, target_path, MAX_RESOURCE_UPLOAD_BYTES)
            
        # Notify MCP Server to scan
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {
            "message": "HTML added to resources successfully",
            "name": file_name,
            "session_id": normalized_session,
            "user_id": user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding HTML resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resources/pdfs")
async def upload_resource_pdfs(request: Request, files: List[UploadFile] = File(...), session_id: Optional[str] = Form(default=None)):
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file PDF")

    normalized_session = _normalize_session_id(session_id) if session_id else None
    user = _current_user_from_request(request)
    user_id = str(user.get("id") or "") if user else None
    pdf_dir, _, _ = resource_loader._scope_dirs(session_id=normalized_session, user_id=user_id)
    result = _save_resource_batch(files, pdf_dir, {".pdf"})
    if result["uploaded_count"] > 0:
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")

    result["session_id"] = normalized_session
    result["user_id"] = user_id
    return result


@app.post("/api/resources/htmls")
async def upload_resource_htmls(request: Request, files: List[UploadFile] = File(...), session_id: Optional[str] = Form(default=None)):
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file HTML")

    normalized_session = _normalize_session_id(session_id) if session_id else None
    user = _current_user_from_request(request)
    user_id = str(user.get("id") or "") if user else None
    _, html_dir, _ = resource_loader._scope_dirs(session_id=normalized_session, user_id=user_id)
    result = _save_resource_batch(files, html_dir, {".html", ".htm"})
    if result["uploaded_count"] > 0:
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")

    result["session_id"] = normalized_session
    result["user_id"] = user_id
    return result


@app.post("/api/resources/upload")
async def upload_resource_files(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(default=None),
):
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file")

    normalized_session = _normalize_session_id(session_id) if session_id else None
    user = _current_user_from_request(request)
    user_id = str(user.get("id") or "") if user else None
    pdf_dir, html_dir, _ = resource_loader._scope_dirs(session_id=normalized_session, user_id=user_id)

    result = _save_resource_mixed_batch(files, pdf_dir, html_dir)
    if result["uploaded_count"] > 0:
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")

    result["uploaded_pdf_count"] = sum(1 for item in result["uploaded"] if item.get("type") == "pdf")
    result["uploaded_html_count"] = sum(1 for item in result["uploaded"] if item.get("type") == "html")
    result["session_id"] = normalized_session
    result["user_id"] = user_id
    return result

@app.post("/api/resources/url")
async def add_resource_url(req: UrlRequest, request: Request):
    try:
        normalized_session = _normalize_session_id(req.session_id) if req.session_id else None
        user = _current_user_from_request(request)
        user_id = str(user.get("id") or "") if user else None
        resource_loader.add_url(req.url, session_id=normalized_session, user_id=user_id)
        
        try:
            _scan_resources_with_owner(normalized_session, user_id=user_id)
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
             
        return {
            "message": "URL added to resources successfully",
            "url": req.url,
            "session_id": normalized_session,
            "user_id": user_id,
        }
    except Exception as e:
        logger.error(f"Error adding URL resource: {e}")
        # Specific handler for WAF/Crawler errors
        if "WAF Blocked" in str(e):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/resources/{resource_id}")
async def delete_resource(request: Request, resource_id: str, session_id: Optional[str] = Query(default=None)):
    try:
        normalized_session = _normalize_session_id(session_id) if session_id else None
        user = _current_user_from_request(request)
        user_id = str(user.get("id") or "") if user else None
        success = resource_loader.delete_resource(resource_id, session_id=normalized_session, user_id=user_id)
        if not success:
             raise HTTPException(status_code=404, detail="Resource not found")
        
        # Trigger Reset Scan
        try:
            payload: Dict[str, Any] = {"reset": True}
            if user_id:
                payload["user_id"] = user_id
            elif normalized_session:
                payload["session_id"] = normalized_session
            mcp_client.invoke("scan_resources", payload)
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {"message": "Resource deleted successfully", "session_id": normalized_session, "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/google/start")
async def start_google_auth(req: GoogleAuthStartRequest):
    try:
        sid = _normalize_session_id(req.session_id or "user_session_1")
        return mail_agent_service.start_app_auth(session_id=sid, redirect_uri=req.redirect_uri)
    except Exception as e:
        logger.error("Error starting Google auth: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/auth/google/callback")
async def complete_google_auth_callback(
    request: Request,
    state: str,
    code: str,
    redirect_uri: Optional[str] = None,
):
    try:
        result = mail_agent_service.complete_app_auth(state=state, code=code, redirect_uri=redirect_uri)
        response = JSONResponse(
            {
                "message": f"Signed in as {result['user']['email']}. You can close this tab and return to app.",
                "auth": {"authenticated": True, "user": result["user"]},
                "migration": result.get("migration"),
            }
        )
        response.set_cookie(
            key=mail_agent_service.app_session_cookie_name,
            value=result["app_session_token"],
            httponly=True,
            samesite=APP_COOKIE_SAMESITE,
            secure=APP_COOKIE_SECURE,
            max_age=mail_agent_service.app_session_ttl_days * 24 * 60 * 60,
            path="/",
        )
        return response
    except Exception as e:
        logger.error("Error completing Google auth: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/auth/me")
async def get_auth_me(request: Request):
    try:
        return mail_agent_service.get_auth_me(request.cookies.get(_mail_cookie_name()))
    except Exception as e:
        logger.error("Error fetching auth me: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/logout")
async def logout_auth(request: Request):
    try:
        mail_agent_service.logout_app_session(request.cookies.get(_mail_cookie_name()))
        response = JSONResponse({"authenticated": False, "user": None})
        response.delete_cookie(
            key=_mail_cookie_name(),
            path="/",
        )
        return response
    except Exception as e:
        logger.error("Error logging out auth session: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mail/status")
async def get_mail_status(request: Request, session_id: str = Query(...)):
    try:
        sid = _normalize_session_id(session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        return _invoke_mail_service(mail_agent_service.get_status, sid, owner_ctx=owner_ctx)
    except Exception as e:
        logger.error("Error fetching mail status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mail/connect/start")
async def start_mail_connect(req: MailConnectStartRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        return _invoke_mail_service(mail_agent_service.begin_oauth, sid, req.redirect_uri, owner_ctx=owner_ctx)
    except Exception as e:
        logger.error("Error starting mail oauth: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mail/connect/callback")
async def complete_mail_connect(req: MailConnectCallbackRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        return _invoke_mail_service(
            mail_agent_service.complete_oauth,
            sid,
            req.state,
            req.code,
            req.redirect_uri,
            owner_ctx=owner_ctx,
        )
    except Exception as e:
        logger.error("Error completing mail oauth: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mail/connect/callback")
async def complete_mail_connect_get(
    state: str,
    code: str,
    redirect_uri: Optional[str] = None,
):
    try:
        status = mail_agent_service.complete_oauth_from_state(
            state=state,
            code=code,
            redirect_uri=redirect_uri,
        )
        email = status.get("email") or "unknown"
        return {
            "message": f"Gmail connected for {email}. You can close this tab and return to app.",
            "status": status,
        }
    except Exception as e:
        logger.error("Error completing mail oauth via GET callback: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mail/disconnect")
async def disconnect_mail(req: MailSessionRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        return _invoke_mail_service(mail_agent_service.disconnect, sid, owner_ctx=owner_ctx)
    except Exception as e:
        logger.error("Error disconnecting mail oauth: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mail/whitelist")
async def get_mail_whitelist(request: Request, session_id: str = Query(...)):
    try:
        sid = _normalize_session_id(session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        return {
            "session_id": sid,
            "user_id": owner_ctx.get("user_id"),
            "owner_type": owner_ctx.get("owner_type"),
            "senders": _invoke_mail_service(mail_agent_service.get_whitelist, sid, owner_ctx=owner_ctx),
        }
    except Exception as e:
        logger.error("Error fetching whitelist: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mail/whitelist")
async def set_mail_whitelist(req: MailWhitelistRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        senders = _invoke_mail_service(mail_agent_service.set_whitelist, sid, req.senders, owner_ctx=owner_ctx)
        return {
            "session_id": sid,
            "user_id": owner_ctx.get("user_id"),
            "owner_type": owner_ctx.get("owner_type"),
            "senders": senders,
        }
    except Exception as e:
        logger.error("Error setting whitelist: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mail/poll")
async def poll_mail_now(req: MailSessionRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        poll_owner = getattr(mail_agent_service, "poll_owner", None)
        if callable(poll_owner):
            return poll_owner(owner_ctx, max_messages=20)
        return mail_agent_service.poll_session(sid)
    except MailOAuthRefreshError as e:
        logger.warning("Gmail OAuth refresh requires reconnect: %s", e)
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error("Error polling mail manually: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mail/candidates")
async def list_mail_candidates(
    request: Request,
    session_id: str = Query(...),
    status: Optional[str] = Query(default=None),
):
    try:
        sid = _normalize_session_id(session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        items = _invoke_mail_service(mail_agent_service.list_candidates, sid, status=status, owner_ctx=owner_ctx)
        return {
            "session_id": sid,
            "user_id": owner_ctx.get("user_id"),
            "owner_type": owner_ctx.get("owner_type"),
            "count": len(items),
            "candidates": items,
        }
    except Exception as e:
        logger.error("Error listing mail candidates: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mail/candidates/{candidate_id}/apply")
async def apply_mail_candidate(candidate_id: str, req: MailCandidateActionRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        result = _invoke_mail_service(mail_agent_service.apply_candidate, sid, candidate_id, owner_ctx=owner_ctx)
        try:
            _scan_resources_with_owner(sid, user_id=owner_ctx.get("user_id"))
        except Exception as scan_err:
            logger.warning("scan_resources after apply failed: %s", scan_err)
        return {
            "session_id": sid,
            "user_id": owner_ctx.get("user_id"),
            "owner_type": owner_ctx.get("owner_type"),
            "candidate": result,
        }
    except Exception as e:
        logger.error("Error applying mail candidate %s: %s", candidate_id, e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mail/candidates/{candidate_id}/reject")
async def reject_mail_candidate(candidate_id: str, req: MailCandidateActionRequest, request: Request):
    try:
        sid = _normalize_session_id(req.session_id)
        owner_ctx = _resolve_mail_owner(request, sid)
        result = _invoke_mail_service(mail_agent_service.reject_candidate, sid, candidate_id, req.reason, owner_ctx=owner_ctx)
        return {
            "session_id": sid,
            "user_id": owner_ctx.get("user_id"),
            "owner_type": owner_ctx.get("owner_type"),
            "candidate": result,
        }
    except Exception as e:
        logger.error("Error rejecting mail candidate %s: %s", candidate_id, e)
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global last_file_id, last_uploaded_file_ids
    try:
        original_name = _validate_upload_file(file, {".pdf"}, {"application/pdf"})
        stem = Path(original_name).stem
        ext = Path(original_name).suffix or ".pdf"
        file_id = f"{stem}_{uuid4().hex[:8]}{ext}"
        dest_path = PDF_DIR / file_id

        _copy_upload_to_path(file, dest_path, MAX_TRANSCRIPT_UPLOAD_BYTES)
        logger.info("Da luu PDF %s, se xu ly khi truy van dau tien", file_id)

        file_meta[file_id] = original_name
        loaded_file_ids.add(file_id)
        last_uploaded_file_ids = [file_id]

        return {"message": "PDF da duoc xu ly thanh cong", "file_id": file_id, "file_name": original_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Loi khi xu ly PDF: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    return [{"file_id": fid, "file_name": file_meta.get(fid, fid)} for fid in loaded_file_ids]


@app.post("/upload_pdfs")
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)):
    global last_file_id, last_uploaded_file_ids
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file PDF")
    _validate_batch_size(files)

    results = []
    errors = []

    def handle_one(upload_file: UploadFile):
        original_name = _validate_upload_file(upload_file, {".pdf"}, {"application/pdf"})
        stem = Path(original_name).stem
        ext = Path(original_name).suffix or ".pdf"
        file_id_local = f"{stem}_{uuid4().hex[:8]}{ext}"
        dest_path = PDF_DIR / file_id_local
        _copy_upload_to_path(upload_file, dest_path, MAX_TRANSCRIPT_UPLOAD_BYTES)
        logger.info("Da luu PDF %s, se xu ly khi truy van dau tien", file_id_local)
        return file_id_local, original_name

    with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
        future_map = {executor.submit(handle_one, f): f.filename for f in files}
        for fut in as_completed(future_map):
            try:
                fid, fname = fut.result()
                file_meta[fid] = fname
                loaded_file_ids.add(fid)
                last_file_id = fid
                results.append({"file_id": fid, "file_name": fname})
            except Exception as exc:
                errors.append(_http_error_message(exc))

    if not results and errors:
        status_code = 413 if any("gioi han" in err.lower() for err in errors) else 400
        raise HTTPException(status_code=status_code, detail="; ".join(errors))
    if results:
        last_uploaded_file_ids = [item["file_id"] for item in results]

    return {"uploaded": results, "errors": errors}

@app.post("/ask")
async def ask_question(http_request: Request, payload: QueryRequest):
    # Orchestrator flow:
    # 1) plan -> 2) parse -> 3) execute -> 4) respond
    query = payload.query
    session_id = payload.session_id or "user_session_1"
    user_id = _current_user_id_from_request(http_request)
    state_before = _load_structured_state(session_id, user_id=user_id)
    resolution = resolve_query_with_state(query, state_before)
    resolved_query = str(resolution.get("resolved_query") or query or "").strip()
    if resolved_query and resolved_query != query:
        logger.info(
            "Structured state resolved query for session %s: raw='%s' -> resolved='%s' refs=%s",
            session_id,
            query,
            resolved_query,
            resolution.get("applied_referents") or [],
        )
    selected_files = _normalize_file_ids(payload.file_ids or [])
    session_meta = _load_session_meta(session_id)
    if not selected_files:
        cached_files = session_meta.get("file_ids") or []
        if cached_files:
            selected_files = _normalize_file_ids(cached_files)
        elif last_uploaded_file_ids:
            selected_files = _normalize_file_ids(last_uploaded_file_ids)

    if not selected_files and _query_requires_transcript_files(resolved_query):
        return {
            "answer": (
                "Bạn chưa chọn file bảng điểm cho phiên này. "
                "Vui lòng tick các file trong mục 'File đã tải lên' rồi gửi lại."
            ),
            "selected_program_id": None,
        }

    requested_program_id = str(payload.program_id).strip() if payload.program_id else None
    cached_program_id = str(session_meta.get("program_id")).strip() if session_meta.get("program_id") else None
    effective_program_id = requested_program_id or cached_program_id

    session_lock = _get_session_lock(session_id)
    lock_acquired = session_lock.acquire(blocking=False)
    if not lock_acquired:
        return {
            "answer": "Hệ thống đang xử lý câu hỏi trước đó trong phiên này. Vui lòng đợi phản hồi rồi gửi tiếp.",
            "selected_program_id": effective_program_id,
        }

    try:
        try:
            programs = _fetch_available_programs(refresh=False)
        except Exception as e:
            logger.warning("Khong lay duoc danh sach CTDT: %s", e)
            return _program_selection_response(
                [],
                answer="Không thể tải danh sách chương trình đào tạo lúc này. Vui lòng thử lại sau vài giây.",
            )

        if not programs:
            _save_session_program(session_id, None)
            if selected_files:
                _save_session_files(session_id, selected_files)
            return _program_selection_response([])

        valid_program_ids = {p.get("id") for p in programs if p.get("id")}
        if effective_program_id and valid_program_ids and effective_program_id not in valid_program_ids:
            logger.warning("Program ID khong hop le hoac da thay doi: %s", effective_program_id)
            effective_program_id = None
            _save_session_program(session_id, None)

        if not effective_program_id:
            if selected_files:
                _save_session_files(session_id, selected_files)
            return _program_selection_response(programs)

        if selected_files:
            _save_session_files(session_id, selected_files)
        _save_session_program(session_id, effective_program_id)

        route: Dict[str, Any] = {"intent": None, "confidence": 0.0, "signals": []}
        route_intent = ""
        route_confidence = 0.0
        if STRUCTURED_TKB_ENABLED:
            route = _structured_intent_classifier(resolved_query)
            route_intent = str(route.get("intent") or "")
            route_confidence = float(route.get("confidence") or 0.0)
            logger.info(
                "[route] session=%s intent=%s confidence=%.2f signals=%s",
                session_id,
                route_intent or "none",
                route_confidence,
                route.get("signals") or [],
            )
        else:
            logger.info("[route] structured TKB disabled by flag for session=%s", session_id)

        memory_context = _load_memory_context_for_session(session_id=session_id, max_rows=10, user_id=user_id)
        planner_orchestration_required = (
            STRUCTURED_TKB_ENABLED
            and _query_requires_planner_orchestration(query=resolved_query, route_intent=route_intent)
        )
        if planner_orchestration_required:
            logger.info(
                "[route] forcing planner orchestration for complex query. session=%s intent=%s confidence=%.2f",
                session_id,
                route_intent,
                route_confidence,
            )
        structured_prefetch: Optional[Dict[str, Any]] = None
        if STRUCTURED_TKB_ENABLED and route_intent and route_confidence >= 0.45:
            structured_prefetch = _build_structured_route_payload(
                query=resolved_query,
                session_id=session_id,
                program_id=effective_program_id,
                intent=route_intent,
                confidence=route_confidence,
                memory_context=memory_context,
                user_id=user_id,
            )

        obj: Dict[str, Any]
        advisor_priority_query = bool(selected_files) and _query_requires_advisor_priority(resolved_query)
        if (
            STRUCTURED_TKB_ENABLED
            and structured_prefetch
            and route_confidence >= 0.75
            and not advisor_priority_query
            and not planner_orchestration_required
        ):
            logger.info(
                "[route] high-confidence structured route chosen. session=%s intent=%s confidence=%.2f",
                session_id,
                route_intent,
                route_confidence,
            )
            obj = structured_prefetch
        elif (
            selected_files
            and _query_requires_transcript_files(resolved_query)
            and (advisor_priority_query or not route_intent or route_confidence < 0.45)
        ):
            logger.info(
                "[route] transcript-intensive query bypasses planner; using deterministic fallback tool chain. session=%s",
                session_id,
            )
            obj = _fallback_planner_payload(
                query=resolved_query,
                session_id=session_id,
                selected_files=selected_files,
                program_id=effective_program_id,
                structured_prefetch=structured_prefetch,
                force_advisor=advisor_priority_query,
                user_id=user_id,
            )
        else:
            # 1) PLAN
            files_hint = f"[FILES:{','.join(selected_files)}]" if selected_files else "[FILES:none]"
            planner_agent = get_mcp_planner_agent(allow_web_search=payload.allow_web_search)
            planner_input = f"[SESSION:{session_id}] [PROGRAM:{effective_program_id}] {files_hint} {resolved_query}"
            planner_output = ""
            planner_error: Optional[Exception] = None
            for attempt in range(1, 3):
                try:
                    planner_output = planner_agent.run(planner_input).content
                    planner_error = None
                    break
                except Exception as planner_run_err:
                    planner_error = planner_run_err
                    logger.warning(
                        "Planner run failed (attempt %s/2) session=%s: %s",
                        attempt,
                        session_id,
                        planner_run_err,
                    )

            # 2) PARSE / FALLBACK
            if planner_error is not None:
                logger.warning(
                    "Planner unavailable after retries; using fallback payload. session=%s error=%s",
                    session_id,
                    planner_error,
                )
                obj = _fallback_planner_payload(
                    query=resolved_query,
                    session_id=session_id,
                    selected_files=selected_files,
                    program_id=effective_program_id,
                    structured_prefetch=structured_prefetch,
                    force_advisor=advisor_priority_query,
                    user_id=user_id,
                )
            else:
                obj = _parse_planner_output(planner_output)
                if obj is None:
                    preview = (planner_output or "").strip().replace("\n", "\\n")
                    logger.warning(
                        "Planner output khong parse duoc JSON (fallback). session=%s len=%s preview=%s",
                        session_id,
                        len(planner_output or ""),
                        preview[:200],
                    )
                    obj = _fallback_planner_payload(
                        query=resolved_query,
                        session_id=session_id,
                        selected_files=selected_files,
                        program_id=effective_program_id,
                        structured_prefetch=structured_prefetch,
                        force_advisor=advisor_priority_query,
                        user_id=user_id,
                    )
                elif (
                    STRUCTURED_TKB_ENABLED
                    and structured_prefetch
                    and 0.45 <= route_confidence < 0.75
                    and not planner_orchestration_required
                ):
                    planner_source = str(obj.get("source") or "")
                    planner_context = _context_to_text(obj.get("context"))
                    route_targets_structured_schedule = route_intent in {
                        "teacher_by_subject",
                        "course_schedule",
                        "classes_by_teacher",
                        "electives_overview",
                        "course_offering_status",
                        "semester_code_lookup",
                    }
                    should_promote_structured = (
                        planner_source in {"vector_store", "schedule_lookup", "error"}
                        or not planner_context.strip()
                        or (route_targets_structured_schedule and planner_source == "academic_advisor")
                    )
                    if should_promote_structured:
                        logger.info(
                            "[route] medium-confidence blend promoted structured payload. session=%s planner_source=%s",
                            session_id,
                            planner_source,
                        )
                        obj = structured_prefetch

        source = str(obj.get("source") or "")
        context = _context_to_text(obj.get("context"))
        memory_context = _context_to_text(obj.get("memory")) or memory_context
        chunk_index = obj.get("chunk_index")

        if source in {"vector_store", "vector_store_compare"} and selected_files and _query_prefers_global_resource_retrieval(resolved_query):
            try:
                global_ctx = _invoke_mcp_tool(
                    "retrieve_chunks",
                    {"question": resolved_query, "top_k": 25, "file_ids": [], "session_id": session_id},
                    timeout_seconds=MCP_TOOL_TIMEOUTS.get("retrieve_chunks"),
                )
                global_context_text = _context_to_text(global_ctx)
                if global_context_text.strip():
                    logger.info(
                        "[route] replaced planner vector context with global-scope retrieve for policy/time-slot query. session=%s",
                        session_id,
                    )
                    context = global_context_text
                    source = "vector_store"
            except Exception as global_retrieve_err:
                logger.info(
                    "[route] global retrieve override skipped session=%s err=%s",
                    session_id,
                    global_retrieve_err,
                )

        if source == "program_selection" or obj.get("requires_selection") is True:
            planner_programs = _normalize_program_list(context)
            if not planner_programs:
                planner_programs = programs
            return _program_selection_response(planner_programs)

        if source == "error":
            logger.warning("Planner tra ve error: %s", context)
            friendly = context or "Khong lay duoc ke hoach. Thu lai hoac bat tim kiem web."
            _save_structured_state(
                session_id=session_id,
                prev_state=state_before,
                raw_query=query,
                resolved_query=resolved_query,
                answer=friendly,
                planner_source=source,
                planner_context=context,
                selected_program_id=effective_program_id,
            )
            return {"answer": friendly, "selected_program_id": effective_program_id}

        route_meta = obj.get("route_meta") if isinstance(obj, dict) else None
        if isinstance(route_meta, dict):
            logger.info(
                "[route] tool_used=%s fallback_stage=%s intent=%s confidence=%s",
                route_meta.get("tool_used"),
                route_meta.get("fallback_stage"),
                route_meta.get("intent"),
                route_meta.get("confidence"),
            )

        # 3) EXECUTE
        # Keep planner/orchestration, but avoid a second LLM hop for advisor results.
        # `consult_advisor` already returns a full final answer; re-generating here can
        # introduce extra 503 failures without adding value.
        if source == "academic_advisor":
            answer = (context or "").strip()
            if not answer:
                answer = (
                    "Hệ thống tư vấn học vụ chưa trả về nội dung ở lần gọi này. "
                    "Vui lòng thử lại sau vài giây."
                )
        elif source == "structured_schedule":
            answer = _render_structured_schedule_answer(query=resolved_query, context=context)
        elif source == "time_slot_lookup":
            answer = _render_time_slot_lookup_answer(query=resolved_query, context=context)
        elif source == "semester_code_lookup":
            answer = _render_semester_code_lookup_answer(context=context)
        elif source == "electives_recommendation":
            answer = _render_electives_recommendation_answer(context=context)
        elif source == "electives_schedule":
            answer = _render_electives_schedule_answer(context=context)
        elif source == "course_offering_status":
            answer = _render_course_offering_status_answer(context=context)
        else:
            answer = answer_agent.run(resolved_query, context, source, memory_context)

        # Normalize escaped unicode and strip any duplicated source footer block.
        answer = _normalize_output_text(answer)

        citations: List[Dict[str, Any]] = []
        if source in {"vector_store", "vector_store_compare"}:
            citations = _extract_retrieve_citations(
                context,
                max_items=10,
                query=resolved_query,
                answer=answer,
            )
        elif source == "time_slot_lookup":
            citations = _extract_time_slot_citations(context, max_items=4)
        elif source == "semester_code_lookup":
            citations = _extract_semester_code_citations(context, max_items=4)
        elif source in {"structured_schedule", "electives_schedule", "electives_recommendation", "course_offering_status"}:
            citations = _extract_structured_schedule_citations(context, max_items=12)
        elif source == "academic_advisor":
            # Some advisor answers embed retrieve-style context; parse directly first.
            citations = _extract_retrieve_citations(
                context,
                max_items=10,
                query=resolved_query,
                answer=answer,
            )

        if not citations and source not in {"error", "program_selection"}:
            citations = _backfill_retrieve_citations_for_answer(
                query=resolved_query,
                session_id=session_id,
                file_ids=selected_files,
                max_items=10,
            )

        if user_id:
            try:
                memory.ensure_chat_session(
                    session_id=session_id,
                    user_id=user_id,
                    title=_derive_chat_title(query),
                    selected_program_id=effective_program_id,
                    selected_file_ids=selected_files,
                )
                memory.add_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=query,
                )
                memory.add_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=answer,
                    citations=citations,
                )
            except Exception as e:
                logger.warning("Khong luu duoc chat session/messages cho user=%s session=%s: %s", user_id, session_id, e)

        try:
            mcp_client.invoke(
                "memory_add",
                _with_memory_owner(
                    {
                        "session_id": session_id,
                        "query": query,
                        "answer": answer,
                        "chunk_index": chunk_index,
                    },
                    user_id,
                ),
            )
        except Exception as e:
            logger.warning("Luu lich su loi (bo qua): %s", e)

        _save_structured_state(
            session_id=session_id,
            prev_state=state_before,
            raw_query=query,
            resolved_query=resolved_query,
            answer=answer,
            planner_source=source,
            planner_context=context,
            selected_program_id=effective_program_id,
            user_id=user_id,
        )

        # 4) RESPOND
        return {
            "answer": answer,
            "selected_program_id": effective_program_id,
            "citations": citations,
            "source": source,
        }
    except Exception as e:
        logger.error("Loi khi xu ly cau hoi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            session_lock.release()


@app.get("/history", response_model=List[HistoryItem])
async def get_history(
    http_request: Request,
    session_id: str = "user_session_1",
    page: int = 1,
    per_page: int = 25,
):
    user_id = _current_user_id_from_request(http_request)
    try:
        history_lines = mcp_client.invoke(
            "memory_get",
            _with_memory_owner({"session_id": session_id, "max_rows": per_page}, user_id),
        )
        history_items = []
        for line in history_lines:
            try:
                timestamp_end = line.find("] Query: ")
                if timestamp_end == -1:
                    continue
                timestamp = line[1:timestamp_end]
                query_start = timestamp_end + len("] Query: ")
                query_end = line.find("\nResponse: ")
                if query_end == -1:
                    continue
                query_val = line[query_start:query_end]
                response_val = line[query_end + len("\nResponse: "):]
                history_items.append(HistoryItem(query=query_val, response=response_val, timestamp=timestamp))
            except Exception as e:
                logger.warning("Loi khi parse lich su: %s (line=%s)", e, line)
                continue
        return history_items
    except Exception as e:
        logger.error("Loi khi lay lich su: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/sessions")
async def list_chat_sessions(http_request: Request):
    user_id = _require_authenticated_user_id(http_request)
    sessions = memory.list_chat_sessions(user_id=user_id)
    if not sessions:
        memory.migrate_legacy_history_to_chat_sessions(user_id=user_id)
        sessions = memory.list_chat_sessions(user_id=user_id)
    return {"sessions": sessions}


@app.post("/api/chat/sessions")
async def create_chat_session(http_request: Request, req: ChatSessionCreateRequest):
    user_id = _require_authenticated_user_id(http_request)
    session_id = str(req.session_id or "").strip() or f"session-{uuid4().hex}"
    session = memory.ensure_chat_session(
        session_id=session_id,
        user_id=user_id,
        title=req.title or "Phiên mới",
        selected_program_id=req.selected_program_id,
        selected_file_ids=req.selected_file_ids,
    )
    return {"session": session}


@app.post("/api/chat/migrate")
async def migrate_browser_chat_sessions(http_request: Request, req: ChatMigrationRequest):
    user_id = _require_authenticated_user_id(http_request)
    sessions = list(req.sessions or [])[:100]
    results: List[Dict[str, Any]] = []
    for item in sessions:
        session_id = str(item.session_id or "").strip()
        if not session_id:
            continue
        messages = [
            {
                "role": msg.role,
                "type": msg.type,
                "content": msg.content,
                "text": msg.text,
                "citations": msg.citations or [],
            }
            for msg in list(item.messages or [])[:500]
        ]
        results.append(
            memory.import_chat_session(
                session_id=session_id,
                user_id=user_id,
                title=item.title,
                selected_program_id=item.selected_program_id,
                selected_file_ids=item.selected_file_ids,
                messages=messages,
            )
        )
    return {
        "results": results,
        "imported_sessions": sum(1 for item in results if item.get("status") in {"imported", "metadata_only"}),
        "imported_messages": sum(int(item.get("imported_messages") or 0) for item in results),
    }


@app.get("/api/chat/sessions/{session_id}/messages")
async def get_chat_session_messages(http_request: Request, session_id: str, limit: int = 50):
    user_id = _require_authenticated_user_id(http_request)
    return {"messages": memory.get_chat_messages(session_id=session_id, user_id=user_id, limit=limit)}


@app.patch("/api/chat/sessions/{session_id}")
async def update_chat_session(http_request: Request, session_id: str, req: ChatSessionUpdateRequest):
    user_id = _require_authenticated_user_id(http_request)
    session = memory.update_chat_session(
        session_id=session_id,
        user_id=user_id,
        title=req.title,
        selected_program_id=req.selected_program_id,
        selected_file_ids=req.selected_file_ids,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"session": session}


@app.delete("/api/chat/sessions/{session_id}")
async def archive_chat_session(http_request: Request, session_id: str):
    user_id = _require_authenticated_user_id(http_request)
    if not memory.archive_chat_session(session_id=session_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"ok": True}


@app.delete("/session")
async def delete_session(http_request: Request, req: SessionRequest):
    user_id = _current_user_id_from_request(http_request)
    try:
        memory.clear_session(req.session_id, user_id=user_id)
        session_dir = _session_dir(req.session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        return {"message": f"Da xoa lich su session {req.session_id}"}
    except Exception as e:
        logger.error("Loi khi xoa session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
