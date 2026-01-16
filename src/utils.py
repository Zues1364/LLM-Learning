import json
import os
from bs4 import BeautifulSoup
from pathlib import Path
import numpy as np
import faiss
import pdfplumber
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Tuple
import logging
import urllib.parse
import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import unicodedata
import re
from types import SimpleNamespace

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.2
# Absolute cache dir to avoid CWD mismatch between services
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"
# Versioning to invalidate stale cached chunks/embeddings when parsing logic changes
CHUNK_CACHE_VERSION = "v5"
EMB_CACHE_VERSION = "v3"

# Prepare cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Embedding cache naming
EMB_SUFFIX = "_embeddings.npy"
EMB_META_SUFFIX = "_embeddings_meta.json"


def normalize_for_match(text: str) -> str:
    """
    Lightweight normalization for lexical matching.
    - Remove accents
    - Lowercase
    - Collapse whitespace
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    without_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = without_accents.lower()
    return re.sub(r"\s+", " ", lowered).strip()


# Helper: md5 hash of file
def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Helper for Vision
def describe_image_with_gemini(image) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("[VISION] No API Key found, skipping Vision.")
        return ""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")
        
        prompt = (
            "This image is a page from a document. "
            "Please transcribe the content into Markdown. "
            "If there are tables, represent them as Markdown tables. "
            "If there is text, preserve headers and structure. "
            "Focus on accuracy for numbers."
        )
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return ""


# PDF processing
def clean_and_fill_table(table: List[List[str]]) -> List[List[str]]:
    """
    Cleans text and applies Forward Fill logic for merged cells (specifically for first columns).
    """
    if not table or len(table) < 2:
        return []

    cleaned_table = []
    last_valid_first_col = ""

    for row_idx, row in enumerate(table):
        # Clean text: remove newlines, strip whitespace
        new_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]

        # Logic Forward Fill for the first column (index 0) - Skip header
        if row_idx > 0:
            current_first_col = new_row[0]
            if current_first_col:
                last_valid_first_col = current_first_col
            elif last_valid_first_col:
                new_row[0] = last_valid_first_col

        cleaned_table.append(new_row)

    return cleaned_table


def convert_table_to_markdown(table: List[List[str]]) -> str:
    """Converts List[List] to Markdown Table string."""
    if not table:
        return ""

    header = table[0]
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"

    for row in table[1:]:
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown + "\n"


def normalize_table(table: List[List[str]]) -> List[List[str]]:
    if not table:
        return []
    # Simple normalization: ensure all rows have same length as max_cols
    max_cols = max(len(row) for row in table)
    normalized = [row + [""] * (max_cols - len(row)) for row in table]
    return normalized


def table_to_row_lines(table: List[List[str]]) -> Tuple[str, str, List[str]]:
    normalized = normalize_table(table)
    if not normalized:
        return "", "", []

    header = normalized[0]
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"

    data_lines = []
    for row in normalized[1:]:
        row_line = "| " + " | ".join(row) + " |"
        data_lines.append(row_line)

    return header_line, separator_line, data_lines


def chunk_table_rows(table: List[List[str]], chunk_size: int = 1000) -> List[str]:
    """
    Splits a large table into smaller markdown tables (preserving headers),
    each fitting within `chunk_size` characters.
    """
    header_line, separator_line, data_lines = table_to_row_lines(table)
    if not header_line:
        return []

    header_block = header_line + "\n" + separator_line + "\n"
    chunks = []
    current_chunk = header_block

    for row_line in data_lines:
        addition = row_line + "\n"
        # If adding this row exceeds chunk_size, push current_chunk and start new
        # But always keep at least one row if possible
        if len(current_chunk) + len(addition) > chunk_size and len(current_chunk) > len(header_block):
            chunks.append(current_chunk.strip())
            current_chunk = header_block + addition
        else:
            current_chunk += addition

    if current_chunk.strip() != header_block.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pdf_name = os.path.basename(file_path)
    file_id = pdf_name  # use filename as source id
    cache_file = os.path.join(CACHE_DIR, f"{pdf_name}.pkl")
    cache_metadata_file = os.path.join(CACHE_DIR, f"{pdf_name}_metadata.pkl")

    pdf_hash = get_file_hash(file_path)

    if os.path.exists(cache_file) and os.path.exists(cache_metadata_file):
        try:
            with open(cache_metadata_file, "rb") as f:
                meta = pickle.load(f)
            cached_hash = meta.get("pdf_hash") if isinstance(meta, dict) else meta
            cached_version = meta.get("version") if isinstance(meta, dict) else None
            if cached_hash == pdf_hash and cached_version == CHUNK_CACHE_VERSION:
                logger.info(f"Loading chunks from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Loaded {len(documents)} cached chunks.")
                return documents
            else:
                logger.info("PDF changed or cache version bumped, re-processing...")
        except Exception as e:
            logger.error(f"Loi khi load cache: {e}, re-processing PDF...")

    logger.info(f"Processing PDF with img2table & pdfplumber: {file_path}")
    source = os.path.basename(file_path)

    # --- 1. EXTRACT TABLES (img2table) ---
    try:
        from img2table.document import PDF as Img2TablePDF
        from img2table.ocr import TesseractOCR
        
        ocr = TesseractOCR(lang="vie+eng")
        img_pdf = Img2TablePDF(file_path)
        
        # 2 strategies: implicit_rows and borderless
        tables_by_page = img_pdf.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
        )
        
        # Helper to get image sizes for bbox normalization
        # (Simplified logic: we'll get page size from pdfplumber)
    except Exception as e:
        logger.error(f"img2table extraction failed: {e}")
        tables_by_page = {}

    final_chunks = []
    
    # --- 2. EXTRACT CONTENT (Page by Page) ---
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            
            # A. Get Tables for this page
            # img2table returns dict {page_idx: [tables...]}
            page_tables = list(tables_by_page.get(i) or [])

            # Fallback: if img2table fails or returns empty, try pdfplumber tables
            if not page_tables:
                try:
                    raw_tables = page.extract_tables()
                    for raw_tbl in raw_tables or []:
                        if not raw_tbl:
                            continue
                        normalized_rows = [[(cell or "").strip() for cell in row] for row in raw_tbl]
                        page_tables.append(SimpleNamespace(df=None, content=normalized_rows, bbox=None))
                    if page_tables:
                        logger.info(f"[PDF FALLBACK] pdfplumber extracted {len(page_tables)} tables on page {page_num}")
                except Exception as e:
                    logger.error(f"[PDF FALLBACK] pdfplumber tables failed on page {page_num}: {e}")

            table_bboxes = []
            
            # --- PROCESS TABLES ---
            if page_tables:
                for t_idx, table_obj in enumerate(page_tables, start=1):
                    # Extract Data
                    try:
                        df = getattr(table_obj, "df", None)
                        if df is not None:
                            table_data = df.fillna("").values.tolist()
                        else:
                            content = getattr(table_obj, "content", None)
                            table_data = [[str(c.value) for c in r] for r in content] if content else []
                    except:
                        table_data = []

                    if not table_data: 
                        continue

                    # Validate (simplified check)
                    flat_text = "".join([str(c) for r in table_data for c in r])
                    if len(flat_text) < 10: 
                        continue

                    # Clean and Chunk Table Data (Row-based Chunking)
                    cleaned_data = clean_and_fill_table(table_data)
                    
                    # chunk_size=800 ensures small enough chunks for vector search
                    # avoiding the issue where "Lý thuyết thông tin" is lost in a huge page
                    table_markdown_chunks = chunk_table_rows(cleaned_data, chunk_size=800)
                    
                    for c_idx, chunk_text in enumerate(table_markdown_chunks, start=1):
                        doc = Document(
                            page_content=f"### TABLE (Page {page_num})\n{chunk_text}",
                            metadata={
                                "source": source,
                                "page": page_num,
                                "file_path": file_path,
                                "type": "table",
                                "table_index": t_idx,
                                "table_chunk_index": c_idx
                            }
                        )
                        final_chunks.append(doc)
                    
                    # Store BBox for text exclusion
                    # img2table bbox: (x1, y1, x2, y2)
                    bbox_obj = getattr(table_obj, "bbox", None)
                    if bbox_obj:
                         pass

            # --- PROCESS TEXT ---
            # For now, use pdfplumber text but we accept duplication is better than loss.
            # To avoid total duplication, we can check if text is significantly similar,
            # but for retrieval, having "Lý thuyết thông tin" in a Table Chunk IS the fix.
            raw_text = page.extract_text() or ""
            
            # --- VISION LOGIC (Preserved) ---
            # Check for large images (e.g. Tables as Images)
            images = page.images
            use_vision = False
            if images:
                for img in images:
                     w = img.get('width', 0)
                     h = img.get('height', 0)
                     if w * h > 20000: # Significant image threshold
                         use_vision = True
                         break
            
            vision_text = ""
            if use_vision:
                 logger.info(f"Page {page_num}: Significant image detected. Running Gemini Vision...")
                 try:
                     # resolution=300 is good for OCR
                     page_img_obj = page.to_image(resolution=300)
                     pil_image = page_img_obj.original
                     vision_desc = describe_image_with_gemini(pil_image)
                     if vision_desc:
                         vision_text = f"\n\n### AI EXTRACTED CONTENT FROM IMAGES:\n{vision_desc}"
                 except Exception as e:
                     logger.error(f"Vision failed on page {page_num}: {e}")

            if vision_text:
                raw_text += vision_text
            # -------------------------------------

            # Create TEXT CHUNK
            if raw_text.strip():
                doc = Document(
                    page_content=f"Page {page_num}:\n{raw_text}",
                    metadata={
                       "source": source,
                       "page": page_num,
                       "file_path": file_path,
                       "type": "text"
                    }
                )
                
                # Split huge text pages if needed
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    separators=["\n### ", "\n\n", "\n", " ", ""]
                )
                splits = text_splitter.split_documents([doc])
                final_chunks.extend(splits)

    # Post-processing metadata
    for idx, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["index"] = idx + 1
        # timestamp etc
        chunk.metadata["timestamp"] = datetime.now().isoformat()
        chunk.metadata["file_name"] = pdf_name
        chunk.metadata["file_id"] = file_id

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(final_chunks, f)
        with open(cache_metadata_file, "wb") as f:
            pickle.dump({"pdf_hash": pdf_hash, "version": CHUNK_CACHE_VERSION}, f)
        logger.info(f"Saved {len(final_chunks)} chunks to cache: {cache_file}")
    except Exception as e:
        logger.error(f"Loi khi luu cache: {e}")

    logger.info(f"PDF processed. Generated {len(final_chunks)} chunks.")
    return final_chunks

def load_embeddings_with_cache(file_path: str, embedder: Embeddings, documents: List[Document]) -> np.ndarray:
    """
    Return normalized embeddings for the given documents, caching to disk by PDF hash + embedder model.
    """
    pdf_name = os.path.basename(file_path)
    emb_cache = CACHE_DIR / f"{pdf_name}{EMB_SUFFIX}"
    emb_meta = CACHE_DIR / f"{pdf_name}{EMB_META_SUFFIX}"
    pdf_hash = get_file_hash(file_path)
    embedder_name = getattr(embedder, "model_name", embedder.__class__.__name__)

    if emb_cache.exists() and emb_meta.exists():
        try:
            meta = json.loads(emb_meta.read_text(encoding="utf-8"))
            if (
                meta.get("pdf_hash") == pdf_hash
                and meta.get("embedder") == embedder_name
                and meta.get("version") == EMB_CACHE_VERSION
            ):
                emb_np = np.load(emb_cache)
                if emb_np.ndim == 1:
                    emb_np = np.expand_dims(emb_np, axis=0)
                logger.info(f"Loading cached embeddings for {pdf_name}")
                return emb_np
            else:
                logger.info("Embedding cache invalid (hash/model changed), recomputing...")
        except Exception as e:
            logger.warning(f"Loi khi doc cache embedding: {e}, recompute...")

    texts = [doc.page_content for doc in documents]
    embeddings = embedder.embed_documents(texts)
    emb_np = np.array(embeddings, dtype="float32")
    if emb_np.ndim == 1:
        emb_np = np.expand_dims(emb_np, axis=0)

    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_np = emb_np / norms

    try:
        np.save(emb_cache, emb_np)
        emb_meta.write_text(
            json.dumps(
                {"pdf_hash": pdf_hash, "embedder": embedder_name, "version": EMB_CACHE_VERSION},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(f"Saved embedding cache for {pdf_name} ({emb_np.shape[0]} vectors).")
    except Exception as e:
        logger.warning(f"Khong luu duoc cache embedding cho {pdf_name}: {e}")

    return emb_np


def generate_summary(text: str) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not set; cannot generate summary.")
        return "(Khong the tao tom tat: thieu GOOGLE_API_KEY)"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        max_len = 500_000
        trimmed_text = text[:max_len]
        if len(text) > max_len:
            logger.info(f"Input text too long ({len(text)} chars); truncated to {max_len}.")

        prompt = (
            "Bạn là chuyên gia phân tích. Hãy tóm tắt nội dung tài liệu sau bằng tiếng Việt, "
            "tập trung vào các ý chính, kết luận và số liệu quan trọng. Độ dài khoảng 300-500 từ."
        )
        response = model.generate_content(f"{prompt}\n\n{trimmed_text}")
        summary = getattr(response, "text", "") or ""
        cleaned = summary.strip()
        return cleaned if cleaned else "(Khong co tom tat)"
    except Exception as e:
        logger.error(f"Loi khi sinh tom tat: {e}")
        return "(Khong the tao tom tat)"


# Embedding
class VietnameseEmbedder(Embeddings):
    def __init__(self, model_name="AITeamVN/Vietnamese_Embedding"):
        logger.info(f"Loading Vietnamese Embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Loi khi tao embeddings cho documents: {e}")
            return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Loi khi tao embedding cho query: {e}")
            return [0.0] * 768


# FAISS Vector Store (supports multiple PDFs in one index)
class FAISSVectorStore:
    def __init__(self, documents: List[Document], embedder: Embeddings):
        self.embedder = embedder
        self.documents: List[Document] = []
        self.embeddings_np: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        logger.info(f"[DEBUG] Init FAISSVectorStore with {len(documents)} documents")
        self.add_documents(documents)

    def _rebuild_index(self):
        if self.embeddings_np is None or self.embeddings_np.size == 0:
            self.index = None
            return
        d = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings_np)
        logger.info(f"[DEBUG] Rebuilt FAISS index with {self.index.ntotal} vectors")

    def add_documents(self, documents: List[Document]):
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to FAISSVectorStore")
        for doc in documents:
            if "_norm_text" not in doc.metadata:
                doc.metadata["_norm_text"] = normalize_for_match(doc.page_content)
        embeddings = self.embedder.embed_documents([doc.page_content for doc in documents])
        emb_np = np.array(embeddings, dtype="float32")
        if emb_np.ndim == 1:
            emb_np = np.expand_dims(emb_np, axis=0)

        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_np = emb_np / norms

        if self.embeddings_np is None or self.embeddings_np.size == 0:
            self.embeddings_np = emb_np
            self.documents = list(documents)
        else:
            if emb_np.shape[1] != self.embeddings_np.shape[1]:
                raise ValueError("Embedding dimension mismatch when adding documents.")
            self.embeddings_np = np.vstack([self.embeddings_np, emb_np])
            self.documents.extend(documents)

        logger.info(f"[DEBUG] Total documents now: {len(self.documents)}")
        self._rebuild_index()

    def add_documents_with_embeddings(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents with precomputed (already normalized) embeddings.
        """
        if not documents:
            return

        for doc in documents:
            if "_norm_text" not in doc.metadata:
                doc.metadata["_norm_text"] = normalize_for_match(doc.page_content)

        emb_np = np.array(embeddings, dtype="float32")
        if emb_np.ndim == 1:
            emb_np = np.expand_dims(emb_np, axis=0)

        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_np = emb_np / norms

        if self.embeddings_np is None or self.embeddings_np.size == 0:
            self.embeddings_np = emb_np
            self.documents = list(documents)
        else:
            if emb_np.shape[1] != self.embeddings_np.shape[1]:
                raise ValueError("Embedding dimension mismatch when adding documents with embeddings.")
            self.embeddings_np = np.vstack([self.embeddings_np, emb_np])
            self.documents.extend(documents)

        logger.info(f"[DEBUG] Total documents now: {len(self.documents)}")
        self._rebuild_index()

    def _embed_query(self, query: str) -> np.ndarray:
        q_embedding = self.embedder.embed_query(query)
        q_embedding = np.array(q_embedding, dtype="float32")
        if q_embedding.ndim == 1:
            q_embedding = np.expand_dims(q_embedding, axis=0)
        q_norm = np.linalg.norm(q_embedding, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        return q_embedding / q_norm

    def retrieve(self, query: str, top_k=5, threshold=SIMILARITY_THRESHOLD, file_ids: List[str] | None = None) -> List[Document]:
        logger.info(f"Retrieve for query: {query}")
        if self.embeddings_np is None or self.embeddings_np.size == 0:
            logger.warning("Vector store empty.")
            return []

        q_vec = self._embed_query(query)
        allowed_set = set(file_ids) if file_ids else None

        # Prepare lexical features (diacritic-insensitive) for fallback boosting
        norm_query = normalize_for_match(query)
        query_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", norm_query) if len(t) >= 3]
        phrase_candidates: set[str] = set()
        for n in (3, 2):
            for i in range(len(query_tokens) - n + 1):
                phrase = " ".join(query_tokens[i : i + n])
                if len(phrase) >= 8:
                    phrase_candidates.add(phrase)
        # Also build looser n-grams (keeps short tokens like "tin") to catch course names verbatim
        loose_tokens = re.findall(r"[a-z0-9]+", norm_query)
        for n in (3, 2):
            for i in range(len(loose_tokens) - n + 1):
                phrase = " ".join(loose_tokens[i : i + n])
                if len(phrase) >= 6:
                    phrase_candidates.add(phrase)

        # Vector search (FAISS)
        candidate_indices: List[int]
        if allowed_set:
            candidate_indices = [i for i, doc in enumerate(self.documents) if doc.metadata.get("file_id") in allowed_set]
            if not candidate_indices:
                logger.info("No chunks match given file_ids.")
                return []
            subset = self.embeddings_np[candidate_indices]
            sims = subset @ q_vec.T
            sims = sims.flatten()
            top_n = min(top_k, len(candidate_indices))
            top_idx_sorted = np.argsort(-sims)[:top_n]
            scored = [(candidate_indices[i], sims[i]) for i in top_idx_sorted]
        else:
            if self.index is None:
                logger.warning("FAISS index not initialized.")
                return []
            D, I = self.index.search(q_vec, top_k)
            scored = list(zip(I[0], D[0]))

        scored = [(idx, sim) for idx, sim in scored if idx >= 0]

        # Base scores from vector search
        combined_scores: dict[int, float] = {idx: sim for idx, sim in scored}

        # Lexical boosting: bring exact/near-exact string matches into the result set
        if query_tokens:
            for idx, doc in enumerate(self.documents):
                if allowed_set and doc.metadata.get("file_id") not in allowed_set:
                    continue

                norm_doc = doc.metadata.get("_norm_text") or normalize_for_match(doc.page_content)
                doc.metadata["_norm_text"] = norm_doc

                phrase_hit = any(p in norm_doc for p in phrase_candidates) if phrase_candidates else False
                token_hits = sum(1 for t in query_tokens if t in norm_doc)

                if phrase_hit or token_hits:
                    boost = 0.0
                    if phrase_hit:
                        boost += 0.60  # strong boost for multi-word match (e.g., course name)
                    if token_hits:
                        boost += min(0.05 * token_hits, 0.35)  # bounded token bonus

                    prev = combined_scores.get(idx, 0.0)
                    combined_scores[idx] = prev + boost
                    # logger.info(f"[DEBUG] Lexical boost chunk {idx + 1}: base={prev:.4f} boost={boost:.4f} phrase_hit={phrase_hit} token_hits={token_hits}")

        # Re-rank with combined scores
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Document] = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            doc_file = self.documents[idx].metadata.get("file_id", "unknown")
            logger.info(f"[DEBUG] Chunk {idx + 1} ({doc_file}): combined_score = {score:.4f}, threshold = {threshold}")
            if score >= threshold:
                results.append(self.documents[idx])

        if not results and threshold > 0.05:
            logger.info(f"[DEBUG] No hits above {threshold}, retry with 0.05")
            for idx, score in ranked:
                if len(results) >= top_k:
                    break
                if score >= 0.05:
                    results.append(self.documents[idx])

        logger.info(f"[DEBUG] Returned {len(results)} documents")
        return results


# Web searching tool
def web_search(query: str, num_results=10, api_key="b91e335ef3ef0b0f01dceef77c1c057d0d538bed") -> List[str]:
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={api_key}"
    try:
        response = requests.get(url)
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets if snippets else ["(Khong co ket qua web)"]
    except Exception as e:
        logger.error(f"Loi khi tim kiem web: {e}")
        return ["(Khong co ket qua web)"]


def parse_curriculum_from_html_content(html_content: str) -> List[dict]:
    """
    Parses the curriculum HTML content into a structured list of blocks and sub-blocks.
    Returns a hierarchy of Knowledge Blocks with credit requirements and subjects.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the main table - heuristic: tables with "Mã" and "Số tín chỉ"
    target_table = None
    for t in soup.find_all('table'):
        t_text = t.get_text()
        if "Mã" in t_text and "học phần" in t_text and "Số tín chỉ" in t_text:
            target_table = t
            break
    
    if not target_table:
        logger.warning("[Curriculum Parsing] Could not find curriculum table in HTML.")
        return []

    rows = target_table.find_all('tr')
    
    structure = []
    current_block = None
    current_sub_block = None

    # Regex patterns
    block_pattern = re.compile(r"^[IVX]+\s*$") # I, II, III...
    sub_block_pattern = re.compile(r"^[IVX]+\.\d+(\.\d+)?\s*$") # V.1, V.2.1

    for row in rows:
        cols = row.find_all('td')
        if not cols: continue
        
        col_texts = [c.get_text(separator=" ", strip=True) for c in cols]
        
        # Skip header rows
        if len(col_texts) > 3 and "Số tín chỉ" in col_texts[3]:
            continue

        first_col = col_texts[0]
        
        # 1. Detect Main Block (I, II...)
        if block_pattern.match(first_col):
            block_name = col_texts[1] if len(col_texts) > 1 else ""
            credits = 0
            for c in col_texts[2:]:
                if re.match(r"^\d+(/\d+)?$", c):
                     parts = c.split('/')
                     credits = int(parts[0])
                     break
            
            current_block = {
                "id": first_col,
                "name": block_name,
                "required_credits": credits,
                "type": "main",
                "subjects": [],
                "sub_blocks": []
            }
            structure.append(current_block)
            current_sub_block = None
            continue

        # 2. Detect Sub-Block (V.1, V.2...)
        if sub_block_pattern.match(first_col):
            sub_name = col_texts[1] if len(col_texts) > 1 else ""
            credits = 0
            for c in col_texts[2:]:
                 if re.match(r"^\d+(/\d+)?$", c):
                         parts = c.split('/')
                         credits = int(parts[0])
                         break
            
            current_sub_block = {
                "id": first_col,
                "name": sub_name,
                "required_credits": credits,
                "type": "sub",
                "subjects": [],
                "sub_blocks": [] 
            }
            if current_block:
                current_block["sub_blocks"].append(current_sub_block)
            continue
            
        # 3. Detect Subject
        if len(col_texts) >= 4:
            code = col_texts[1]
            name = col_texts[2]
            credit_text = col_texts[3]
            
            if re.match(r"^[A-Z]{3}\d{4}[A-Z]?$", code):
                try:
                    creds = int(credit_text)
                except:
                    creds = 0
                
                subj = {"code": code, "name": name, "credits": creds}
                
                if current_sub_block:
                    current_sub_block["subjects"].append(subj)
                elif current_block:
                    current_block["subjects"].append(subj)
    
    return structure


def compute_curriculum_missing_credits(structure: List[dict], transcript_codes: set) -> List[dict]:
    """
    Computes missing credits per block based on structure and user transcript.
    Handles hierarchy by aggregating subjects from children sub-blocks if necessary.
    """
    missing_details = []

    for block in structure:
        # Check if block has sub-blocks
        if not block["sub_blocks"]:
            # Simple case: Main Block with direct subjects
            main_collected = [s for s in block["subjects"] if s["code"] in transcript_codes]
            main_accumulated = sum(s["credits"] for s in main_collected)
            
            missing = block["required_credits"] - main_accumulated
            if missing > 0:
                candidates = [s for s in block["subjects"] if s["code"] not in transcript_codes]
                missing_details.append({
                    "block_name": block['name'],
                    "missing_credits": missing,
                    "candidates": candidates
                })
        else:
            # Complex case: Block has sub-buckets
            # 1. Identify Requirement Buckets (Sub-blocks with req > 0)
            # 2. Each non-bucket sub-block (req=0) belongs to the nearest preceding bucket?
            #    Or belongs to Main Block? In the VNU HTML:
            #    V.2 (21 credits) -> [V.2.1, V.2.2, V.2.3, V.2.4] (0 credits explicitly, but contain subjects)
            
            buckets = []
            current_bucket = None
            
            # Treat Main Block subjects as a "Main" bucket if it has requirements?
            # Actually Main Block req (51) = sum of V.1 (18) + V.2 (21) + V.3 (5) + V.4 (7)
            # So we iterate sub-blocks only.
            
            for sub in block["sub_blocks"]:
                if sub["required_credits"] > 0:
                    current_bucket = {
                        "name": sub["name"],
                        "required_credits": sub["required_credits"],
                        "subjects": list(sub["subjects"])
                    }
                    buckets.append(current_bucket)
                else:
                    if current_bucket:
                        current_bucket["subjects"].extend(sub["subjects"])
            
            # Process each bucket
            for bucket in buckets:
                bucket_accumulated = 0
                bucket_completed_codes = set()
                
                for s in bucket["subjects"]:
                     if s["code"] in transcript_codes and s["code"] not in bucket_completed_codes:
                         bucket_accumulated += s["credits"]
                         bucket_completed_codes.add(s["code"])
                
                bs_missing = bucket["required_credits"] - bucket_accumulated
                if bs_missing > 0:
                     candidates = [s for s in bucket["subjects"] if s["code"] not in transcript_codes]
                     missing_details.append({
                        "block_name": f"{block['name']} - {bucket['name']}",
                        "missing_credits": bs_missing,
                        "candidates": candidates
                     })

    return missing_details
