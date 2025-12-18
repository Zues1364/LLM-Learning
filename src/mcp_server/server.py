import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from env_loader import load_env
from utils import web_search, VietnameseEmbedder, FAISSVectorStore, process_pdf, generate_summary, load_embeddings_with_cache
from persistent_memory import PersistentMemory
from agents import get_academic_advisor_agent
import google.generativeai as genai

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-Tools MCP Server")

TOOL_REGISTRY: Dict[str, callable] = {}


def mcp_tool(name: str):
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


@app.get("/mcp/discover")
def discover() -> dict:
    return {"tools": list(TOOL_REGISTRY.keys())}


class InvokeRequest(BaseModel):
    tool: str
    args: dict


@app.post("/mcp/invoke")
def invoke(req: InvokeRequest):
    fn = TOOL_REGISTRY.get(req.tool)
    if not fn:
        logger.error(f"Tool not found: {req.tool}")
        raise HTTPException(404, "Tool not found")
    try:
        result = fn(**req.args)
        logger.info(f"Tool {req.tool} invoked successfully with args: {req.args}")
        return {"result": result}
    except HTTPException:
        # Preserve HTTP-specific errors (e.g., 404) instead of wrapping them as 500
        raise
    except Exception as e:
        logger.error(f"Error invoking tool {req.tool}: {str(e)}")
        raise HTTPException(500, str(e))


# === Tool implementations =======================================

@mcp_tool("web_search_tool")
def web_search_tool(query: str, num_results: int = 10) -> List[str]:
    """Search snippets via Serper API."""
    try:
        logger.info(f"Performing web search for query: {query}")
        results = web_search(query, num_results)
        return results
    except Exception as e:
        logger.error(f"Error in web_search_tool: {str(e)}")
        raise


BASE_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = BASE_DIR / "data" / "pdfs"
MEMORY_DB = BASE_DIR / "data" / "memory.db"

load_env()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

_embedder: Optional[VietnameseEmbedder] = None
_store: Optional[FAISSVectorStore] = None  
_loaded_files: Set[str] = set()


def _resolve_pdf_path(file_id: str) -> Path:
    """
    Resolve an incoming file_id to an actual PDF path.
    Accepts:
      - full filename stored under data/pdfs
      - the short hash/id suffix (e.g., f99fbe39 or f99fbe39.pdf)
      - any substring that uniquely matches a PDF filename
    """
    candidate = PDF_DIR / file_id
    if candidate.exists():
        return candidate

    needle = file_id
    if needle.lower().endswith(".pdf"):
        needle = needle[:-4]

    matches = [p for p in PDF_DIR.glob(f"*{needle}*.pdf")]
    if not matches:
        raise HTTPException(404, f"File_id khong ton tai: {file_id}")
    if len(matches) > 1:
        raise HTTPException(400, f"Tim thay {len(matches)} file khop {file_id}, hay chi ro file_id day du.")
    return matches[0]


def _ensure_file_loaded(file_id: str) -> str:
    """Lazy load a PDF into the shared FAISS store. Returns the resolved file_id."""
    global _embedder, _store
    pdf_path = _resolve_pdf_path(file_id)
    resolved_id = pdf_path.name
    if resolved_id in _loaded_files:
        return resolved_id

    if _embedder is None:
        _embedder = VietnameseEmbedder()

    docs = process_pdf(str(pdf_path))
    embeddings = load_embeddings_with_cache(str(pdf_path), _embedder, docs)
    if _store is None:
        _store = FAISSVectorStore([], _embedder)
    _store.add_documents_with_embeddings(docs, embeddings)

    if _memory.get_summary(resolved_id) is None:
        full_text = "\n".join([d.page_content for d in docs])
        summary = generate_summary(full_text)
        _memory.save_summary(resolved_id, summary)
        logger.info(f"Generated and saved summary for {resolved_id}")

    _loaded_files.add(resolved_id)
    logger.info(f"Loaded {resolved_id} into shared FAISS store ({len(docs)} chunks)")
    return resolved_id


@mcp_tool("analyze_transcript")
def analyze_transcript(file_ids: str | List[str]) -> str:
    """
    Trich xuat du lieu co cau truc tu bang diem sinh vien (PDF) bang Gemini.
    Ho tro nhieu file bang diem, noi noi dung lai truoc khi tinh toan.
    Tra ve chuoi JSON.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY/GEMINI_API_KEY missing for analyze_transcript")
        raise HTTPException(500, "Missing GOOGLE_API_KEY/GEMINI_API_KEY for Gemini")

    # Chuan hoa danh sach file_ids (chuoi phan cach dau phay hoac list)
    ids_input = file_ids
    if isinstance(ids_input, str):
        ids: List[str] = [p.strip() for p in ids_input.split(",") if p.strip()]
    else:
        ids = list(ids_input or [])

    if not ids:
        raise HTTPException(400, "Khong co file_id de phan tich bang diem")

    sections: List[str] = []
    try:
        for fid in ids:
            pdf_path = _resolve_pdf_path(fid)
            docs = process_pdf(str(pdf_path))
            file_text = "\n".join(doc.page_content for doc in docs)
            sections.append(f"--- FILE {pdf_path.name} ---\n{file_text}")
    except Exception as e:
        logger.error(f"Error reading transcript {fid}: {e}")
        raise HTTPException(500, f"Khong doc duoc file {fid}: {e}")

    full_text = "\n\n".join(sections)

    prompt = (
        "Hay trich xuat thong tin tu bang diem sinh vien (du lieu gop tu nhieu trang/nhieu file, "
        "cac file duoc ngan cach boi '--- FILE <ten file> ---'): "
        "current_semester (int), total_credits (int tong tat ca hoc phan da hoan thanh), "
        "current_gpa (float GPA tich luy tren toan bo du lieu), passed_subjects (list string ten mon da qua). "
        "Tinh toan tren TOAN BO du lieu, tra ve duy nhat JSON hop le."
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(f"{prompt}\n\n{full_text}")
        raw_text = getattr(response, "text", "") or ""
    except Exception as e:
        logger.error(f"Gemini error during analyze_transcript: {e}")
        raise HTTPException(500, f"Loi goi Gemini: {e}")

    if not raw_text:
        raise HTTPException(500, "Khong nhan duoc ket qua tu Gemini")

    try:
        parsed = json.loads(raw_text)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        logger.warning("Gemini tra ve JSON khong hop le, tra ve nguyen van ban.")
        return raw_text.strip()


@mcp_tool("math_eval")
def math_eval(expression: str) -> str:
    """
    Danh gia bieu thuc toan hoc an toan voi eval, chi chap nhan ky tu so hoc co ban.
    """
    if not re.fullmatch(r"[0-9\\.\\+\\-\\*/()\\s]+", expression or ""):
        return "Error: Unsafe expression"
    try:
        safe_globals: Dict[str, object] = {"__builtins__": {}}
        result = eval(expression, safe_globals, {})
        return str(result)
    except Exception as e:
        logger.error(f"Error in math_eval: {e}")
        return f"Error: {e}"


@mcp_tool("consult_advisor")
def consult_advisor(query: str, file_ids: List[str] | None = None, session_id: str = "default") -> str:
    """
    Goi Academic Advisor Agent tu server, tra ve noi dung tu van.
    Tu dong lay lich su chat de Agent khong bi mat ngu canh.
    """
    ids = file_ids or []
    if isinstance(ids, str):
        ids = [p.strip() for p in ids.split(",") if p.strip()]

    try:
        history_context = _memory.get_context("", session_id=session_id, max_rows=5)
    except Exception as e:
        logger.warning(f"Failed to fetch history in consult_advisor: {e}")
        history_context = ""

    advisor_agent = get_academic_advisor_agent()
    
    prompt = (
        f"--- CONTEXT START ---\n"
        f"Chat History:\n{history_context}\n"
        f"Context Files: {ids}\n"
        f"--- CONTEXT END ---\n\n"
        f"User Query: {query}"
    )
    
    response = advisor_agent.run(prompt)
    return getattr(response, "content", "")


@mcp_tool("retrieve_chunks")
def retrieve_chunks(question: str, top_k: int = 5, file_ids: List[str] | None = None) -> List[str]:
    """Truy xuất các đoạn PDF liên quan cho danh sách file_ids."""
    ids_input = file_ids or []
    if isinstance(ids_input, str):
        ids_input = [p.strip() for p in ids_input.split(",")]
    ids = [fid for fid in ids_input if fid]
    if not ids:
        logger.warning("retrieve_chunks called without file_ids, returning empty.")
        return []

    resolved_ids: List[str] = []
    for fid in ids:
        resolved_ids.append(_ensure_file_loaded(fid))

    if _store is None:
        return []

    contexts: List[str] = []
    for fid in resolved_ids:
        chunks = _store.retrieve(question, top_k=top_k, file_ids=[fid])
        if not chunks:
            contexts.append(f"[{fid}] Khong tim thay doan phu hop.")
            continue
        formatted = "\n\n".join(
            [f"[{chunks[i].metadata.get('file_name', fid)} - Chunk {c.metadata.get('index')}] {c.page_content}"
             for i, c in enumerate(chunks)]
        )
        contexts.append(formatted)

    logger.info(f"Retrieved contexts for {len(ids)} file(s).")
    return contexts


@mcp_tool("compare_pdfs")
def compare_pdfs(query: str, file_ids: List[str], top_k: int = 5) -> List[str]:
    """So sánh/nêu bối cảnh theo query trên tối thiểu hai file."""
    ids_input = file_ids or []
    if isinstance(ids_input, str):
        ids_input = [p.strip() for p in ids_input.split(",")]
    ids = [fid for fid in ids_input if fid]
    if len(ids) < 2:
        raise HTTPException(400, "Can it nhat 2 file_id de so sanh.")

    resolved_ids: List[str] = []
    for fid in ids:
        resolved_ids.append(_ensure_file_loaded(fid))

    if _store is None:
        return []

    selected = resolved_ids[:2]
    contexts: List[str] = []
    for fid in selected:
        chunks = _store.retrieve(query, top_k=top_k, file_ids=[fid])
        if not chunks:
            contexts.append(f"[{fid}] Khong tim thay noi dung phu hop.")
            continue
        ctx = "\n\n".join([f"[{c.metadata.get('file_name', fid)} - Chunk {c.metadata.get('index')}] {c.page_content}" for c in chunks])
        contexts.append(ctx)

    return contexts


@mcp_tool("get_file_summaries")
def get_file_summaries(file_ids: List[str]) -> List[str]:
    """Lấy bản tóm tắt nội dung chính của danh sách file_ids."""
    ids_input = file_ids or []
    if isinstance(ids_input, str):
        ids_input = [p.strip() for p in ids_input.split(",")]
    ids = [fid for fid in ids_input if fid]
    if not ids:
        raise HTTPException(400, "file_ids khong duoc de trong.")

    summaries: List[str] = []
    for fid in ids:
        resolved_id = _ensure_file_loaded(fid)
        summary = _memory.get_summary(resolved_id)
        summaries.append(f"--- Summary [{resolved_id}] ---\n{summary if summary else '(Khong co tom tat)'}")

    return summaries


_memory = PersistentMemory(db_path=str(MEMORY_DB), max_history=25)


@mcp_tool("memory_get")
def memory_get(session_id: str, max_rows: int = 10) -> List[str]:
    """Lay lich su hoi thoai"""
    try:
        logger.info(f"Retrieving history for session: {session_id}")
        ctx = _memory.get_context("", session_id=session_id, max_rows=max_rows)
        result = ctx.splitlines()
        logger.info(f"Retrieved {len(result)} history entries.")
        return result
    except Exception as e:
        logger.error(f"Error in memory_get: {str(e)}")
        raise


@mcp_tool("memory_add")
def memory_add(
    session_id: str,
    query: str,
    answer: str,
    chunk_index: int | None = None
):
    """Luu Q/A vao history"""
    try:
        logger.info(f"Adding to history for session: {session_id}, query: {query}")
        _memory.add_to_history(query, answer, session_id, chunk_index)
        logger.info("History entry added successfully.")
        return "ok"
    except Exception as e:
        logger.error(f"Error in memory_add: {str(e)}")
        raise
