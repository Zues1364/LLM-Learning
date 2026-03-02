import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Set, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AnswerGeneratorAgent, get_mcp_planner_agent, get_rag_agent
from env_loader import load_env

# Initial Env Load & Conflict Resolution
load_env()
gemini_key = os.getenv("GEMINI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
if gemini_key:
    if google_key and google_key != gemini_key:
        logging.warning(
            "Conflict detected: GOOGLE_API_KEY and GEMINI_API_KEY differ. "
            "Overriding GOOGLE_API_KEY with GEMINI_API_KEY."
        )
    os.environ["GOOGLE_API_KEY"] = gemini_key
    # Keep one canonical env var to prevent SDK ambiguity.
    os.environ.pop("GEMINI_API_KEY", None)

from mcp_client.client import MCPClient
from persistent_memory import PersistentMemory
# resource_loader import NOT needed here if we delegate to scan_resources via MCP? 
# Use resource_loader for 'get_resources' list only? 
# Or just duplicate the listing logic to avoid dependency issues if separate process?
# They likely share the 'data' dir.
from resource_loader import resource_loader 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
_session_locks: Dict[str, Lock] = {}
_session_locks_guard = Lock()

def _session_dir(session_id: str) -> Path:
    return SESSION_CACHE_DIR / session_id

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


def _fallback_planner_payload(
    query: str,
    session_id: str,
    selected_files: List[str],
    program_id: Optional[str],
) -> Dict[str, Any]:
    memory_context = ""
    try:
        memory_result = mcp_client.invoke("memory_get", {"session_id": session_id, "max_rows": 10})
        memory_context = _context_to_text(memory_result)
    except Exception as mem_err:
        logger.warning("Fallback memory_get failed for session %s: %s", session_id, mem_err)

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
    prefer_advisor = any(marker in lowered for marker in academic_markers)

    tool_chain = (
        [
            ("consult_advisor", {"query": query, "file_ids": selected_files, "session_id": session_id, "program_id": program_id}, "academic_advisor"),
            ("retrieve_chunks", {"question": query, "top_k": 15, "file_ids": selected_files}, "vector_store"),
        ]
        if prefer_advisor
        else [
            ("retrieve_chunks", {"question": query, "top_k": 15, "file_ids": selected_files}, "vector_store"),
            ("consult_advisor", {"query": query, "file_ids": selected_files, "session_id": session_id, "program_id": program_id}, "academic_advisor"),
        ]
    )

    for tool_name, args, source in tool_chain:
        try:
            tool_result = mcp_client.invoke(tool_name, args)
            return {
                "source": source,
                "context": _context_to_text(tool_result),
                "memory": memory_context,
                "chunk_index": None,
            }
        except Exception as tool_err:
            logger.warning("Planner fallback %s failed for session %s: %s", tool_name, session_id, tool_err)

    return {
        "source": "error",
        "context": "He thong khong tao duoc ke hoach fallback. Vui long thu lai.",
        "memory": memory_context,
        "chunk_index": None,
    }

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

class UrlRequest(BaseModel):
    url: str

# --- Resource Endpoints ---

def _is_allowed_extension(filename: Optional[str], allowed_exts: Set[str]) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in allowed_exts


def _save_resource_batch(files: List[UploadFile], target_dir: Path, allowed_exts: Set[str]) -> Dict[str, Any]:
    uploaded: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for upload_file in files:
        original_name = Path(upload_file.filename or "").name or "unnamed"
        if not _is_allowed_extension(original_name, allowed_exts):
            errors.append({"name": original_name, "error": "invalid extension"})
            continue

        target_path = target_dir / original_name
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            uploaded.append({"name": original_name})
        except Exception as e:
            errors.append({"name": original_name, "error": str(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "uploaded_count": len(uploaded),
        "error_count": len(errors),
    }

@app.get("/api/resources")
async def get_resources():
    # Use locally imported resource_loader just to LIST.
    # It reads from disk/config.json.
    return resource_loader.get_resources()

@app.get("/api/programs")
async def get_programs(refresh: bool = False):
    try:
        programs = _fetch_available_programs(refresh=refresh)
        return {"programs": programs, "count": len(programs)}
    except Exception as e:
        logger.error("Loi khi lay danh sach chuong trinh dao tao: %s", e)
        raise HTTPException(status_code=500, detail="Khong the lay danh sach chuong trinh dao tao.")

@app.post("/api/resources/pdf")
async def upload_resource_pdf(file: UploadFile = File(...)):
    if not _is_allowed_extension(file.filename, {".pdf"}):
        raise HTTPException(status_code=400, detail="File phai la PDF")
    
    try:
        file_name = Path(file.filename or "").name or "uploaded.pdf"
        # Save directly to resource dir
        target_path = RESOURCE_PDF_DIR / file_name
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Notify MCP Server to scan
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {"message": "PDF added to resources successfully", "name": file_name}
    except Exception as e:
        logger.error(f"Error adding PDF resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resources/html")
async def upload_resource_html(file: UploadFile = File(...)):
    if not _is_allowed_extension(file.filename, {".html", ".htm"}):
        raise HTTPException(status_code=400, detail="File phai la HTML")
    
    try:
        file_name = Path(file.filename or "").name or "uploaded.html"
        # Save directly to resource dir
        target_path = RESOURCE_HTML_DIR / file_name
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Notify MCP Server to scan
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {"message": "HTML added to resources successfully", "name": file_name}
    except Exception as e:
        logger.error(f"Error adding HTML resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resources/pdfs")
async def upload_resource_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file PDF")

    result = _save_resource_batch(files, RESOURCE_PDF_DIR, {".pdf"})
    if result["uploaded_count"] > 0:
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")

    return result


@app.post("/api/resources/htmls")
async def upload_resource_htmls(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file HTML")

    result = _save_resource_batch(files, RESOURCE_HTML_DIR, {".html", ".htm"})
    if result["uploaded_count"] > 0:
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")

    return result

@app.post("/api/resources/url")
async def add_resource_url(req: UrlRequest):
    try:
        # We can use resource_loader.add_url locally which updates config.json
        # Then trigger scan on server
        resource_loader.add_url(req.url)
        
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
             
        return {"message": "URL added to resources successfully", "url": req.url}
    except Exception as e:
        logger.error(f"Error adding URL resource: {e}")
        # Specific handler for WAF/Crawler errors
        if "WAF Blocked" in str(e):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/resources/{resource_id}")
async def delete_resource(resource_id: str):
    try:
        success = resource_loader.delete_resource(resource_id)
        if not success:
             raise HTTPException(status_code=404, detail="Resource not found")
        
        # Trigger Reset Scan
        try:
            mcp_client.invoke("scan_resources", {"reset": True})
        except Exception as e:
            logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {"message": "Resource deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global last_file_id, last_uploaded_file_ids
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File phai la PDF")

    try:
        original_name = Path(file.filename).name or "uploaded.pdf"
        stem = Path(original_name).stem
        ext = Path(original_name).suffix or ".pdf"
        file_id = f"{stem}_{uuid4().hex[:8]}{ext}"
        dest_path = PDF_DIR / file_id

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Da luu PDF %s, se xu ly khi truy van dau tien", file_id)

        file_meta[file_id] = original_name
        loaded_file_ids.add(file_id)
        last_uploaded_file_ids = [file_id]

        return {"message": "PDF da duoc xu ly thanh cong", "file_id": file_id, "file_name": original_name}
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

    results = []
    errors = []

    def handle_one(upload_file: UploadFile):
        original_name = Path(upload_file.filename).name or "uploaded.pdf"
        stem = Path(original_name).stem
        ext = Path(original_name).suffix or ".pdf"
        file_id_local = f"{stem}_{uuid4().hex[:8]}{ext}"
        dest_path = PDF_DIR / file_id_local
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.info("Da luu PDF %s, se xu ly khi truy van dau tien", file_id_local)
        return file_id_local, original_name

    with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
        future_map = {executor.submit(handle_one, f): f.filename for f in files if f.filename.endswith(".pdf")}
        for fut in as_completed(future_map):
            try:
                fid, fname = fut.result()
                file_meta[fid] = fname
                loaded_file_ids.add(fid)
                last_file_id = fid
                results.append({"file_id": fid, "file_name": fname})
            except Exception as exc:
                errors.append(str(exc))

    if not results and errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))
    if results:
        last_uploaded_file_ids = [item["file_id"] for item in results]

    return {"uploaded": results, "errors": errors}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Orchestrator flow:
    # 1) plan -> 2) parse -> 3) execute -> 4) respond
    query = request.query
    session_id = request.session_id or "user_session_1"
    selected_files = _normalize_file_ids(request.file_ids or [])
    session_meta = _load_session_meta(session_id)
    if not selected_files:
        cached_files = session_meta.get("file_ids") or []
        if cached_files:
            selected_files = _normalize_file_ids(cached_files)
        elif last_uploaded_file_ids:
            selected_files = _normalize_file_ids(last_uploaded_file_ids)

    requested_program_id = str(request.program_id).strip() if request.program_id else None
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

        # 1) PLAN
        files_hint = f"[FILES:{','.join(selected_files)}]" if selected_files else "[FILES:none]"
        planner_agent = get_mcp_planner_agent(allow_web_search=request.allow_web_search)
        planner_input = f"[SESSION:{session_id}] [PROGRAM:{effective_program_id}] {files_hint} {query}"
        planner_output = planner_agent.run(planner_input).content

        # 2) PARSE
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
                query=query,
                session_id=session_id,
                selected_files=selected_files,
                program_id=effective_program_id,
            )

        source = str(obj.get("source") or "")
        context = _context_to_text(obj.get("context"))
        memory_context = _context_to_text(obj.get("memory"))
        chunk_index = obj.get("chunk_index")

        if source == "program_selection" or obj.get("requires_selection") is True:
            planner_programs = _normalize_program_list(context)
            if not planner_programs:
                planner_programs = programs
            return _program_selection_response(planner_programs)

        if source == "error":
            logger.warning("Planner tra ve error: %s", context)
            friendly = context or "Khong lay duoc ke hoach. Thu lai hoac bat tim kiem web."
            return {"answer": friendly, "selected_program_id": effective_program_id}

        # 3) EXECUTE
        answer = answer_agent.run(query, context, source, memory_context)

        try:
            mcp_client.invoke(
                "memory_add",
                {
                    "session_id": session_id,
                    "query": query,
                    "answer": answer,
                    "chunk_index": chunk_index,
                },
            )
        except Exception as e:
            logger.warning("Luu lich su loi (bo qua): %s", e)

        # 4) RESPOND
        return {"answer": answer, "selected_program_id": effective_program_id}
    except Exception as e:
        logger.error("Loi khi xu ly cau hoi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            session_lock.release()


@app.get("/history", response_model=List[HistoryItem])
async def get_history(session_id: str = "user_session_1", page: int = 1, per_page: int = 25):
    try:
        history_lines = mcp_client.invoke(
            "memory_get", {"session_id": session_id, "max_rows": per_page}
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


@app.delete("/session")
async def delete_session(req: SessionRequest):
    try:
        memory.clear_session(req.session_id)
        session_dir = _session_dir(req.session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        return {"message": f"Da xoa lich su session {req.session_id}"}
    except Exception as e:
        logger.error("Loi khi xoa session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
