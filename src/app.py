import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AnswerGeneratorAgent, get_mcp_planner_agent, get_rag_agent
from env_loader import load_env
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
SESSION_CACHE_DIR = BASE_DIR / "data" / "session_cache"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(RESOURCE_PDF_DIR, exist_ok=True)
os.makedirs(SESSION_CACHE_DIR, exist_ok=True)

# Globals
memory = PersistentMemory(db_path=str(BASE_DIR / "data" / "memory.db"), max_history=25)
loaded_file_ids: Set[str] = set()
file_meta: Dict[str, str] = {}  # file_id -> original filename
last_uploaded_file_ids: List[str] = []
rag_agent = None
mcp_client = MCPClient()

# Load env
load_env()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

answer_agent = AnswerGeneratorAgent(get_rag_agent())

def _session_dir(session_id: str) -> Path:
    return SESSION_CACHE_DIR / session_id

def _session_meta_path(session_id: str) -> Path:
    return _session_dir(session_id) / "meta.json"

def _load_session_files(session_id: str) -> List[str]:
    meta_path = _session_meta_path(session_id)
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        ids = data.get("file_ids", [])
        return ids if isinstance(ids, list) else []
    except Exception:
        return []

def _save_session_files(session_id: str, file_ids: List[str]):
    if not file_ids:
        return
    try:
        dir_path = _session_dir(session_id)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta_path = _session_meta_path(session_id)
        meta_path.write_text(json.dumps({"file_ids": file_ids}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Khong luu duoc session files cho %s: %s", session_id, e)

class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False
    session_id: str = "user_session_1"
    file_ids: List[str] | None = None

class HistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str

class SessionRequest(BaseModel):
    session_id: str

class UrlRequest(BaseModel):
    url: str

# --- Resource Endpoints ---

@app.get("/api/resources")
async def get_resources():
    # Use locally imported resource_loader just to LIST.
    # It reads from disk/config.json.
    return resource_loader.get_resources()

@app.post("/api/resources/pdf")
async def upload_resource_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File phai la PDF")
    
    try:
        # Save directly to resource dir
        target_path = RESOURCE_PDF_DIR / file.filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Notify MCP Server to scan
        try:
            mcp_client.invoke("scan_resources", {})
        except Exception as e:
             logger.warning(f"Failed to trigger MCP scan: {e}")
            
        return {"message": "PDF added to resources successfully", "name": file.filename}
    except Exception as e:
        logger.error(f"Error adding PDF resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    query = request.query
    session_id = request.session_id or "user_session_1"
    selected_files = request.file_ids or []
    if not selected_files:
        cached_files = _load_session_files(session_id)
        if cached_files:
            selected_files = cached_files
        elif last_uploaded_file_ids:
            selected_files = last_uploaded_file_ids
    selected_files = list(dict.fromkeys([f for f in selected_files if f]))

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

        if selected_files:
            _save_session_files(session_id, selected_files)

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
    try:
        memory.clear_session(req.session_id)
        session_dir = _session_dir(req.session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        return {"message": f"Da xoa lich su session {req.session_id}"}
    except Exception as e:
        logger.error("Loi khi xoa session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
