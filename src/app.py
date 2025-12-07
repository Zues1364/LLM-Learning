import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AnswerGeneratorAgent, get_mcp_planner_agent, get_rag_agent
from env_loader import load_env
from mcp_client.client import MCPClient
from persistent_memory import PersistentMemory
from utils import FAISSVectorStore, VietnameseEmbedder, process_pdf

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
os.makedirs(PDF_DIR, exist_ok=True)

# Globals
memory = PersistentMemory(db_path=str(BASE_DIR / "data" / "memory.db"), max_history=25)
embedder: Optional[VietnameseEmbedder] = None
vector_store: Optional[FAISSVectorStore] = None  # shared store for all PDFs
loaded_file_ids: Set[str] = set()
file_meta: Dict[str, str] = {}  # file_id -> original filename
last_file_id: Optional[str] = None
rag_agent = None
mcp_client = MCPClient()

# Load env and map GEMINI_API_KEY to GOOGLE_API_KEY for Gemini SDK compatibility
load_env()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

answer_agent = AnswerGeneratorAgent(get_rag_agent())


def _ensure_file_loaded(file_id: str):
    """Load PDF chunks into the shared vector store if not already loaded."""
    global embedder, vector_store
    if file_id in loaded_file_ids:
        return

    pdf_path = PDF_DIR / file_id
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File_id khong ton tai: {file_id}")

    if embedder is None:
        embedder = VietnameseEmbedder()

    docs = process_pdf(str(pdf_path))
    if vector_store is None:
        vector_store = FAISSVectorStore(docs, embedder)
    else:
        vector_store.add_documents(docs)

    loaded_file_ids.add(file_id)
    file_meta.setdefault(file_id, pdf_path.name)


class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False
    session_id: str = "user_session_1"
    file_ids: List[str] | None = None


class HistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str


class CompareRequest(BaseModel):
    file_ids: List[str]
    query: str


class SessionRequest(BaseModel):
    session_id: str


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a single PDF, assign a unique file_id, and add chunks into the shared FAISS store.
    """
    global embedder, last_file_id, vector_store
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

        documents = process_pdf(str(dest_path))
        logger.info("Da xu ly PDF %s, tao %s chunks", file_id, len(documents))

        if embedder is None:
            embedder = VietnameseEmbedder()
        if vector_store is None:
            vector_store = FAISSVectorStore(documents, embedder)
        else:
            vector_store.add_documents(documents)

        file_meta[file_id] = original_name
        loaded_file_ids.add(file_id)
        last_file_id = file_id

        return {"message": "PDF da duoc xu ly thanh cong", "file_id": file_id, "file_name": original_name}
    except Exception as e:
        logger.error("Loi khi xu ly PDF: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    """List uploaded PDFs in current runtime."""
    return [{"file_id": fid, "file_name": file_meta.get(fid, fid)} for fid in loaded_file_ids]


@app.post("/upload_pdfs")
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDFs concurrently and add all chunks into the shared FAISS store.
    """
    global embedder, last_file_id, vector_store
    if not files:
        raise HTTPException(status_code=400, detail="Chua chon file PDF")

    if embedder is None:
        embedder = VietnameseEmbedder()

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
        docs = process_pdf(str(dest_path))
        return file_id_local, original_name, docs

    with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
        future_map = {executor.submit(handle_one, f): f.filename for f in files if f.filename.endswith(".pdf")}
        for fut in as_completed(future_map):
            try:
                fid, fname, docs = fut.result()
                if vector_store is None:
                    vector_store = FAISSVectorStore(docs, embedder)
                else:
                    vector_store.add_documents(docs)
                file_meta[fid] = fname
                loaded_file_ids.add(fid)
                last_file_id = fid
                results.append({"file_id": fid, "file_name": fname})
            except Exception as exc:
                errors.append(str(exc))

    if not results and errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    return {"uploaded": results, "errors": errors}


@app.post("/compare_pdfs")
async def compare_pdfs(request: CompareRequest):
    """
    Compare/query across two PDFs that have been uploaded (shared FAISS store).
    """
    if len(request.file_ids) < 2:
        raise HTTPException(status_code=400, detail="Can cung cap it nhat 2 file_id de so sanh.")

    for fid in request.file_ids:
        _ensure_file_loaded(fid)

    if vector_store is None:
        raise HTTPException(status_code=400, detail="Chua co PDF nao duoc tai len.")

    selected_ids = request.file_ids[:2]
    contexts = []
    for fid in selected_ids:
        docs = vector_store.retrieve(request.query, top_k=3, file_ids=[fid])
        if not docs:
            contexts.append(f"[{file_meta.get(fid, fid)}] Khong tim thay noi dung phu hop.")
            continue
        chunk_text = "\n\n".join([f"[{doc.metadata.get('file_name', fid)} - Chunk {doc.metadata.get('index')}] {doc.page_content}" for doc in docs])
        contexts.append(f"[{file_meta.get(fid, fid)}]\n{chunk_text}")

    combined_context = "\n\n-----\n\n".join(contexts)
    answer = answer_agent.run(request.query, combined_context, "vector_store_compare", "")

    return {"answer": answer, "file_ids": selected_ids}


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Nhận câu hỏi, dùng planner (MCP tools) để lấy context, sau đó Gemini trả lời.
    Mọi lỗi planner/thiếu dữ liệu sẽ trả lời thân thiện thay vì 500.
    """
    query = request.query
    session_id = request.session_id or "user_session_1"
    selected_files = request.file_ids or ([last_file_id] if last_file_id else [])

    try:
        files_hint = f"[FILES:{','.join(selected_files)}]" if selected_files else "[FILES:none]"
        planner_agent = get_mcp_planner_agent(allow_web_search=request.allow_web_search)
        planner_output = planner_agent.run(f"[SESSION:{session_id}] {files_hint} {query}").content

        try:
            match = re.search(r"{.*}", planner_output, re.DOTALL)
            payload = match.group(0) if match else planner_output
            obj = json.loads(payload)
            source = obj.get("source", "")
            context = obj.get("context", "")
            memory_context = obj.get("memory", "")
            chunk_index = obj.get("chunk_index")
        except Exception:
            logger.warning("Planner output khong parse duoc JSON: %s", planner_output)
            friendly = "Khong doc duoc ke hoach, ban co the hoi lai hoac bat tim kiem web."
            return {"answer": friendly}

        if source == "error":
            logger.warning("Planner tra ve error: %s", context)
            friendly = context or "Khong lay duoc ke hoach. Thu lai hoac bat tim kiem web."
            return {"answer": friendly}

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

        return {"answer": answer}
    except Exception as e:
        logger.error("Loi khi xu ly cau hoi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Xoa toan bo lich su hoi thoai cua mot session_id.
    """
    try:
        memory.clear_session(req.session_id)
        return {"message": f"Da xoa lich su session {req.session_id}"}
    except Exception as e:
        logger.error("Loi khi xoa session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
