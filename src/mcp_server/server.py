import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from env_loader import load_env
from utils import web_search, VietnameseEmbedder, FAISSVectorStore, process_pdf, generate_summary, load_embeddings_with_cache
from persistent_memory import PersistentMemory

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
