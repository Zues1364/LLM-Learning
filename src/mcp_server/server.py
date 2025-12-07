import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from utils import web_search, VietnameseEmbedder, FAISSVectorStore, process_pdf
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

_embedder: Optional[VietnameseEmbedder] = None
_store: Optional[FAISSVectorStore] = None  # shared store across PDFs
_loaded_files: Set[str] = set()


def _ensure_file_loaded(file_id: str):
    """Lazy load a PDF into the shared FAISS store."""
    global _embedder, _store
    if file_id in _loaded_files:
        return

    pdf_path = PDF_DIR / file_id
    if not pdf_path.exists():
        raise HTTPException(404, f"File_id khong ton tai: {file_id}")

    if _embedder is None:
        _embedder = VietnameseEmbedder()

    docs = process_pdf(str(pdf_path))
    if _store is None:
        _store = FAISSVectorStore(docs, _embedder)
    else:
        _store.add_documents(docs)

    _loaded_files.add(file_id)
    logger.info(f"Loaded {file_id} into shared FAISS store ({len(docs)} chunks)")


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

    for fid in ids:
        _ensure_file_loaded(fid)

    if _store is None:
        return []

    contexts: List[str] = []
    for fid in ids:
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

    for fid in ids:
        _ensure_file_loaded(fid)

    if _store is None:
        return []

    selected = ids[:2]
    contexts: List[str] = []
    for fid in selected:
        chunks = _store.retrieve(query, top_k=top_k, file_ids=[fid])
        if not chunks:
            contexts.append(f"[{fid}] Khong tim thay noi dung phu hop.")
            continue
        ctx = "\n\n".join([f"[{c.metadata.get('file_name', fid)} - Chunk {c.metadata.get('index')}] {c.page_content}" for c in chunks])
        contexts.append(ctx)

    return contexts


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
