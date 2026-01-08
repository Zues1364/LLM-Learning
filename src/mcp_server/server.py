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
    Trich xuat du lieu chi tiet tu bang diem sinh vien (PDF) bang Gemini.
    Parse theo hoc ky, tra ve JSON cau truc va chuan hoa diem.
    """
    logger.info("analyze_transcript start: file_ids=%s", file_ids)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY/GEMINI_API_KEY missing for analyze_transcript")
        raise HTTPException(500, "Missing API KEY")

    ids_input = file_ids
    if isinstance(ids_input, str):
        ids: List[str] = [p.strip() for p in ids_input.split(',') if p.strip()]
    else:
        ids = list(ids_input or [])

    logger.info("normalized file_ids: %s", ids)
    if not ids:
        logger.warning("No file_ids provided to analyze_transcript")
        raise HTTPException(400, "Thieu file_id bang diem")

    preview_len = 500
    texts: List[Dict[str, str]] = []
    for fid in ids:
        try:
            logger.info("Processing transcript file_id=%s", fid)
            try:
                resolved_id = _ensure_file_loaded(fid)
                pdf_path = _resolve_pdf_path(resolved_id)
            except HTTPException:
                logger.warning(f"File ID not found or invalid: {fid}. Skipping.")
                continue

            logger.info("Resolved path for %s: %s", fid, pdf_path)
            docs = process_pdf(str(pdf_path))
            logger.info("Extracted %s chunks from %s", len(docs), pdf_path.name)
            file_text = "\n".join(doc.page_content for doc in docs)
            logger.info("file_text length for %s: %s", pdf_path.name, len(file_text))
            if file_text:
                preview = file_text[:preview_len].replace("\n", " ")
                logger.info("file_text preview for %s: %s", pdf_path.name, preview)
            texts.append({"file_id": resolved_id, "text": file_text})
        except Exception as e:
            logger.error(f"Loi doc file transcript {fid}: {e}")
            # Do not raise, continue to next file
            continue

    if not texts:
        msg = "Khong tim thay bat ky file bang diem nao hop le."
        logger.error(msg)
        raise HTTPException(400, msg)

    prompt = (
        "Ban la he thong trich xuat du lieu bang diem dai hoc.\n"
        "INPUT:\n"
        "- Van ban chua du lieu bang diem, phan chia theo tung hoc ky (Header dang 'HOC KY... MA HOC KY...').\n"
        "- Cac cot du lieu: STT, Ma MH, Ten Mon Hoc, So TC, Diem he 10, Diem chu, Diem he 4.\n"
        "\n"
        "OUTPUT JSON FORMAT (chi tra ve JSON hop le, dung dau nhay kep, KHONG markdown):\n"
        "{\n"
        "  \"student_info\": {\"name\": \"...\", \"id\": \"...\", \"class\": \"...\"},\n"
        "  \"semesters\": [\n"
        "    {\n"
        "      \"semester_code\": \"Ma hoc ky (vi du 231, 232)\",\n"
        "      \"semester_title\": \"Ten day du hoc ky\",\n"
        "      \"subjects\": [\n"
        "        {\n"
        "          \"code\": \"Ma mon\",\n"
        "          \"name\": \"Ten mon (noi cac dong neu bi ngat)\",\n"
        "          \"credits\": 3,\n"
        "          \"grade_10\": 8.5,\n"
        "          \"grade_letter\": \"A+\",\n"
        "          \"grade_4\": 4.0\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        "  \"overview\": {\"raw_gpa_4\": 3.21, \"total_credits_accumulated\": 90}\n"
        "}\n"
        "Neu \"grade_4\" bi trong/khong ro, tu quy doi tu diem chu: "
        "A+=4.0, A=3.7, B+=3.5, B=3.0, C+=2.5, C=2.0, D+=1.5, D=1.0, F=0.0."
    )
    logger.info("prompt length: %s", len(prompt))

    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(',', '.')
            if not text:
                return None
            try:
                return float(text)
            except Exception:
                return None
        return None

    def _to_int(value):
        number = _to_float(value)
        if number is None:
            return None
        return int(round(number))

    grade_map = {
        "A+": 4.0, "A": 3.7,
        "B+": 3.5, "B": 3.0,
        "C+": 2.5, "C": 2.0,
        "D+": 1.5, "D": 1.0,
        "F": 0.0,
    }

    def _normalize_data(data: Dict) -> Dict:
        semesters = data.get("semesters")
        if not isinstance(semesters, list):
            semesters = []
            data["semesters"] = semesters

        all_subjects: List[dict] = []
        for sem in semesters:
            if not isinstance(sem, dict):
                continue
            subjects = sem.get("subjects")
            if not isinstance(subjects, list):
                subjects = []
                sem["subjects"] = subjects
            for sub in subjects:
                if not isinstance(sub, dict):
                    continue
                credits = _to_int(sub.get("credits"))
                if credits is not None:
                    sub["credits"] = credits
                grade_10 = _to_float(sub.get("grade_10"))
                if grade_10 is not None:
                    sub["grade_10"] = grade_10
                grade_letter = str(sub.get("grade_letter", "")).strip().upper()
                if grade_letter:
                    sub["grade_letter"] = grade_letter
                grade_4 = _to_float(sub.get("grade_4"))
                if grade_4 is None and grade_letter:
                    grade_4 = grade_map.get(grade_letter)
                if grade_4 is not None:
                    sub["grade_4"] = grade_4
                all_subjects.append(sub)

        overview = data.get("overview")
        if not isinstance(overview, dict):
            overview = {}
            data["overview"] = overview

        raw_gpa_4 = _to_float(overview.get("raw_gpa_4"))
        if raw_gpa_4 is not None:
            overview["raw_gpa_4"] = raw_gpa_4

        # FORCE RECALCULATION: Ignore Gemini's partial/hallucinated overview stats
        # Always compute from the full list of merged subjects to ensure accuracy.
        total_credits_accumulated = 0
        total_points = 0.0
        
        # Smart Dedup for Accumulated Credits
        # If subject code repeated, keep the one with HIGHER grade_4.
        unique_passed = {} # code -> subject_dict
        
        for sub in all_subjects:
            code = sub.get("code")
            credits = sub.get("credits")
            grade_4 = sub.get("grade_4")

            if not code or credits is None or grade_4 is None:
                continue
            
            # F grade does not count to accumulation
            if grade_4 == 0.0:
                continue

            if code not in unique_passed:
                unique_passed[code] = sub
            else:
                # Duplicate found
                existing = unique_passed[code]
                if grade_4 > existing.get("grade_4", -1.0):
                        logger.info(f"Detected duplicate subject [{code}]. New grade {grade_4} > Old {existing.get('grade_4')}. Updating.")
                        unique_passed[code] = sub
                else:
                        logger.info(f"Detected duplicate subject [{code}]. New grade {grade_4} <= Old {existing.get('grade_4')}. Ignoring.")
        
        # Sum up valid unique subjects
        for sub in unique_passed.values():
            c = sub.get("credits")
            g = sub.get("grade_4")
            total_credits_accumulated += c
            total_points += g * c
        
        overview["total_credits_accumulated"] = total_credits_accumulated
        if total_credits_accumulated > 0:
            overview["raw_gpa_4"] = round(total_points / total_credits_accumulated, 4)
            logger.info(f"Computed raw_gpa_4: {overview['raw_gpa_4']} from {len(unique_passed)} unique subjects")
        else:
             overview["raw_gpa_4"] = 0.0
        
        logger.info(f"Computed total_credits_accumulated: {total_credits_accumulated}")

        return data

    def _parse_raw_json(raw_text: str) -> Dict | None:
        cleaned = raw_text.strip()
        if "```" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        data = None
        try:
            data = json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(cleaned[start:end + 1])
                except Exception as e:
                    logger.warning("Gemini returned invalid JSON: %s", e)
                    logger.info("Gemini raw JSON (full): %s", cleaned)
                    return None
            else:
                logger.warning("Gemini returned non-JSON content")
                logger.info("Gemini raw response (full): %s", cleaned)
                return None
        if not isinstance(data, dict):
            logger.warning("Gemini JSON root is not an object")
            return None
        return data

    merged: Dict[str, object] = {"student_info": None, "semesters": [], "overview": {}}
    errors: List[str] = []

    # Tach theo hoc ky trong tung file de giam kich thuoc dau vao cho Gemini
    semester_pattern = re.compile(r"(?:HỌC KỲ|HOC KY|H\s*¯OC\s*K\s*¯ý)[^\\n]*(?:MÃ HỌC KỲ|MA HOC KY|MAŸ\s*H\s*¯OC\s*K\s*¯ý)[^\\n]*", re.IGNORECASE)

    for entry in texts:
        file_id = entry["file_id"]
        text = entry["text"]
        if not text:
            logger.warning("Empty text for %s", file_id)
            continue

        # Tach segment theo hoc ky
        segments: List[str] = []
        positions = list(semester_pattern.finditer(text))
        if positions:
            for idx, match in enumerate(positions):
                start = match.start()
                end = positions[idx + 1].start() if idx + 1 < len(positions) else len(text)
                segments.append(text[start:end].strip())
        else:
            segments.append(text.strip())

        for seg_idx, segment in enumerate(segments):
            label = f"{file_id}#seg{seg_idx+1}"
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(
                    f"{prompt}\n\nDATA ({label}):\n{segment}",
                    generation_config={"max_output_tokens": 4000, "response_mime_type": "application/json"},
                )
                raw_text = getattr(response, "text", "") or ""
                logger.info("Gemini response length for %s: %s", label, len(raw_text))
                if raw_text:
                    preview = raw_text[:preview_len].replace("\n", " ")
                    logger.info("Gemini response preview for %s: %s", label, preview)
            except Exception as e:
                logger.error(f"Gemini parse error for {label}: {e}")
                errors.append(f"{label}: {e}")
                continue

            if not raw_text:
                logger.warning("Gemini returned empty response for %s", label)
                errors.append(f"{label}: empty response")
                continue

            data = _parse_raw_json(raw_text)
            if data is None:
                errors.append(f"{label}: invalid JSON")
                continue

            if merged.get("student_info") is None and data.get("student_info"):
                merged["student_info"] = data.get("student_info")
            
            sems = data.get("semesters")
            if isinstance(sems, list):
                # Smart Merge: Deduplicate semesters and subjects
                existing_sems = {s.get("semester_code"): s for s in merged["semesters"] if s.get("semester_code")}
                
                for incoming_sem in sems:
                    sem_code = incoming_sem.get("semester_code")
                    if not sem_code:
                        # If no code, just add it (fallback)
                        merged["semesters"].append(incoming_sem)
                        continue

                    if sem_code in existing_sems:
                        # Merge subjects into existing semester
                        target_sem = existing_sems[sem_code]
                        existing_subjects = {subj.get("code"): subj for subj in target_sem.get("subjects", []) if subj.get("code")}
                        
                        inc_subjects = incoming_sem.get("subjects", [])
                        for sub in inc_subjects:
                            sub_code = sub.get("code")
                            if sub_code and sub_code in existing_subjects:
                                continue # Skip duplicate subject
                            
                            # Add new subject
                            if "subjects" not in target_sem: target_sem["subjects"] = []
                            target_sem["subjects"].append(sub)
                    else:
                        # New semester found
                        merged["semesters"].append(incoming_sem)
                        existing_sems[sem_code] = incoming_sem

            ov = data.get("overview")
            if isinstance(ov, dict):
                for k in ["raw_gpa_4", "total_credits_accumulated"]:
                    if k not in merged["overview"] or merged["overview"].get(k) is None:
                        merged["overview"][k] = ov.get(k)

    if not merged["semesters"]:
        msg = "No semesters parsed from Gemini"
        if errors:
            msg += f". Errors: {errors}"
        logger.warning(msg)
        return json.dumps({"error": msg}, ensure_ascii=False)

    normalized = _normalize_data(merged)
    final_json = json.dumps(normalized, ensure_ascii=False)
    logger.info(f"analyze_transcript complete. Final JSON length: {len(final_json)} chars. Semesters found: {len(normalized.get('semesters', []))}")
    return final_json
@mcp_tool("math_eval")
def math_eval(expression: str) -> str:
    """
    Danh gia bieu thuc toan hoc an toan voi eval, chi chap nhan ky tu so hoc co ban.
    """
    if expression is None:
        return "Error: Empty expression"
    # Support Vietnamese number format: replace comma with dot
    # But be careful not to break function calls like pow(a, b) if we supported them (we don't currently)
    expression_clean = str(expression).replace(",", ".")
    
    cleaned = expression_clean or ""
    # Regex allows: 0-9, dot, +, -, *, /, (, ), space. 
    # Using explicit character class without unneeded escapes to avoid confusion.
    if not re.fullmatch(r"[0-9.+-/*()\s]+", cleaned):
        return f"Error: Unsafe expression: {expression}"
    try:
        safe_globals: Dict[str, object] = {"__builtins__": {}}
        result = eval(cleaned, safe_globals, {})
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
    
    # Pre-fetch transcript data to inject into prompt
    transcript_data = ""
    if ids:
        try:
            logger.info(f"consult_advisor: Auto-extracting transcript for files {ids}")
            # Call the local function directly
            transcript_data = analyze_transcript(ids)
        except Exception as e:
            logger.error(f"Failed to auto-extract transcript in consult_advisor: {e}")
            transcript_data = f"Error extracting transcript: {e}"

    prompt = (
        f"--- CONTEXT START ---\n"
        f"Chat History:\n{history_context}\n"
        f"Context Files: {ids}\n"
        f"TRANSCRIPT DATA (Parsed from PDFs):\n{transcript_data}\n"
        f"--- CONTEXT END ---\n\n"
        f"User Query: {query}"
    )
    
    response = advisor_agent.run(prompt)
    return getattr(response, "content", "")


@mcp_tool("retrieve_chunks")
def retrieve_chunks(question: str, top_k: int = 20, file_ids: List[str] | None = None) -> List[str]:
    """Truy xu §t cA­c Ž`o §­n PDF liA¦n quan cho danh sA­ch file_ids."""
    
    # FORCE INCREASE top_k because client might be sending default 5
    if top_k < 20: 
        logger.info(f"Boosting top_k from {top_k} to 20 to ensure recall.")
        top_k = 20

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
    """So sA­nh/nA¦u b ¯`i c §œnh theo query trA¦n t ¯`i thi ¯Ÿu hai file."""
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
    """L §y b §œn tA3m t §_t n ¯Ti dung chA-nh c ¯a danh sA­ch file_ids."""
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
        logger.info(f"Retrieved {len(result)} lines of history context.")
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
