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
from typing import List, Tuple, Dict, Any, Optional, Set
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
from runtime_paths import BASE_DIR, CACHE_DIR

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.2
# Versioning to invalidate stale cached chunks/embeddings when parsing logic changes
CHUNK_CACHE_VERSION = "v5"
EMB_CACHE_VERSION = "v3"
VECTOR_SNAPSHOT_VERSION = "v1"

# Prepare cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Embedding cache naming
EMB_SUFFIX = "_embeddings.npy"
EMB_META_SUFFIX = "_embeddings_meta.json"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LOG_CHUNK_LOADING = _env_flag("LOG_CHUNK_LOADING", default=False)

_CP1252_CHAR_TO_BYTE: Dict[str, int] = {}
for _b in range(256):
    try:
        _ch = bytes([_b]).decode("cp1252")
    except Exception:
        continue
    _CP1252_CHAR_TO_BYTE[_ch] = _b


def _mojibake_score(text: str) -> int:
    if not text:
        return 0
    score_patterns = ("Ã", "Â", "Ä", "áº", "á»", "â€“", "â€”", "â€œ", "â€")
    score = sum(text.count(pattern) for pattern in score_patterns)
    score += sum(1 for ch in text if 0x80 <= ord(ch) <= 0x9F) * 2
    score += sum(text.count(ch) for ch in ("\u2018", "\u2019", "\u201c", "\u201d", "\u2039", "\u203a", "\u20ac"))
    return score


def _reconstruct_misdecoded_bytes(text: str) -> Optional[bytes]:
    raw = str(text or "")
    if not raw:
        return b""
    buf = bytearray()
    for ch in raw:
        code = ord(ch)
        if code <= 0xFF:
            buf.append(code)
            continue
        mapped = _CP1252_CHAR_TO_BYTE.get(ch)
        if mapped is None:
            return None
        buf.append(mapped)
    return bytes(buf)


def _repair_common_mojibake(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    best = raw
    best_score = _mojibake_score(raw)
    if best_score == 0:
        return raw

    reconstructed = _reconstruct_misdecoded_bytes(raw)
    if reconstructed:
        try:
            candidate = reconstructed.decode("utf-8")
            candidate_score = _mojibake_score(candidate)
            if candidate and candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        except Exception:
            pass

    for source_encoding in ("latin1", "cp1252"):
        try:
            candidate = raw.encode(source_encoding).decode("utf-8")
        except Exception:
            continue
        candidate_score = _mojibake_score(candidate)
        if candidate and candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def normalize_for_match(text: str) -> str:
    """
    Lightweight normalization for lexical matching.
    - Repair common UTF-8/latin1 mojibake (for example: "mÃ´n" -> "mon")
    - Remove accents
    - Lowercase
    - Collapse whitespace
    """
    if not text:
        return ""
    repaired = _repair_common_mojibake(str(text))
    if _mojibake_score(repaired) > 0:
        pieces = re.split(r"(\s+)", repaired)
        repaired = "".join(
            _repair_common_mojibake(piece) if piece and not piece.isspace() else piece
            for piece in pieces
        )
    decomposed = unicodedata.normalize("NFD", repaired)
    without_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = (
        without_accents
        .lower()
        .replace("đ", "d")
        .replace("Đ", "d")
        .replace("Ä‘", "d")
        .replace("Ä", "d")
    )
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
    api_key = os.environ.get("GEMINI_API_KEY")
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
                if LOG_CHUNK_LOADING:
                    logger.info(f"Loading chunks from cache: {cache_file}")
                else:
                    logger.debug(f"Loading chunks from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    documents = pickle.load(f)
                if LOG_CHUNK_LOADING:
                    logger.info(f"Loaded {len(documents)} cached chunks.")
                else:
                    logger.debug(f"Loaded {len(documents)} cached chunks.")
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
                    # avoiding the issue where "LÃ½ thuyáº¿t thÃ´ng tin" is lost in a huge page
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
            # but for retrieval, having "LÃ½ thuyáº¿t thÃ´ng tin" in a Table Chunk IS the fix.
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
    expected_dim = int(getattr(embedder, "embedding_dim", 0) or 0)
    expected_docs = len(documents)

    if emb_cache.exists() and emb_meta.exists():
        try:
            meta = json.loads(emb_meta.read_text(encoding="utf-8"))
            emb_np = np.load(emb_cache)
            if emb_np.ndim == 1:
                emb_np = np.expand_dims(emb_np, axis=0)

            dim_ok = (expected_dim <= 0) or (emb_np.shape[1] == expected_dim)
            docs_ok = (meta.get("docs_count") is None) or (int(meta.get("docs_count")) == expected_docs)

            if (
                meta.get("pdf_hash") == pdf_hash
                and meta.get("embedder") == embedder_name
                and meta.get("version") == EMB_CACHE_VERSION
                and dim_ok
                and docs_ok
            ):
                logger.info(f"Loading cached embeddings for {pdf_name}")
                return emb_np
            else:
                if not dim_ok:
                    logger.warning(
                        "Embedding cache invalid (dim mismatch) for %s: cached=%s expected=%s. Recomputing...",
                        pdf_name,
                        emb_np.shape[1],
                        expected_dim,
                    )
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
                {
                    "pdf_hash": pdf_hash,
                    "embedder": embedder_name,
                    "version": EMB_CACHE_VERSION,
                    "dim": int(emb_np.shape[1]),
                    "docs_count": expected_docs,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(f"Saved embedding cache for {pdf_name} ({emb_np.shape[0]} vectors).")
    except Exception as e:
        logger.warning(f"Khong luu duoc cache embedding cho {pdf_name}: {e}")

    return emb_np


def generate_summary(text: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; cannot generate summary.")
        return "(Khong the tao tom tat: thieu GEMINI_API_KEY)"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        max_len = 500_000
        trimmed_text = text[:max_len]
        if len(text) > max_len:
            logger.info(f"Input text too long ({len(text)} chars); truncated to {max_len}.")

        prompt = (
            "Báº¡n lÃ  chuyÃªn gia phÃ¢n tÃ­ch. HÃ£y tÃ³m táº¯t ná»™i dung tÃ i liá»‡u sau báº±ng tiáº¿ng Viá»‡t, "
            "táº­p trung vÃ o cÃ¡c Ã½ chÃ­nh, káº¿t luáº­n vÃ  sá»‘ liá»‡u quan trá»ng. Äá»™ dÃ i khoáº£ng 300-500 tá»«."
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
        self.embedding_dim = self._infer_embedding_dim()
        logger.info("Model loaded.")

    def _infer_embedding_dim(self) -> int:
        """Resolve embedding width once so cache validation can detect stale/corrupt vectors."""
        try:
            dim = int(self.model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
            if dim > 0:
                return dim
        except Exception:
            pass

        try:
            vec = self.model.encode(["dim_probe"], show_progress_bar=False)[0]
            return int(len(vec))
        except Exception:
            return 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            # Enable progress bar for visibility on large batches
            embeddings = self.model.encode(texts, show_progress_bar=True)
            emb_np = np.asarray(embeddings, dtype="float32")
            if emb_np.ndim == 1:
                emb_np = np.expand_dims(emb_np, axis=0)
            return emb_np.tolist()
        except Exception as e:
            logger.error(f"Loi khi tao embeddings cho documents: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            emb_np = np.asarray(embedding, dtype="float32")
            if emb_np.ndim > 1:
                emb_np = emb_np[0]
            return emb_np.tolist()
        except Exception as e:
            logger.error(f"Loi khi tao embedding cho query: {e}")
            return [0.0] * self.embedding_dim


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

    def rebuild_index(self):
        """Public wrapper for explicit single rebuild after batched appends."""
        self._rebuild_index()

    def add_documents(self, documents: List[Document], rebuild_index: bool = True):
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
        if rebuild_index:
            self._rebuild_index()

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        rebuild_index: bool = True,
    ):
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
        if rebuild_index:
            self._rebuild_index()

    @staticmethod
    def _sanitize_snapshot_value(
        value: Any,
        *,
        _seen: Optional[Set[int]] = None,
        _depth: int = 0,
        _max_depth: int = 6,
    ) -> Any:
        """
        Convert arbitrary metadata payload into a pickle-safe acyclic structure.
        """
        if _seen is None:
            _seen = set()

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (np.generic,)):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        if _depth >= _max_depth:
            return str(value)

        if isinstance(value, (dict, list, tuple, set, SimpleNamespace)):
            obj_id = id(value)
            if obj_id in _seen:
                return "<cycle>"
            _seen.add(obj_id)

        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for k, v in value.items():
                key = str(k)
                sanitized[key] = FAISSVectorStore._sanitize_snapshot_value(
                    v,
                    _seen=_seen,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                )
            return sanitized

        if isinstance(value, (list, tuple, set)):
            return [
                FAISSVectorStore._sanitize_snapshot_value(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                )
                for item in value
            ]

        if isinstance(value, SimpleNamespace):
            return FAISSVectorStore._sanitize_snapshot_value(
                vars(value),
                _seen=_seen,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )

        return str(value)

    def save_snapshot(self, snapshot_path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Persist current vector state (documents + normalized embeddings) to disk.
        Returns False when the store is empty.
        """
        if self.embeddings_np is None or self.embeddings_np.size == 0 or not self.documents:
            logger.info("Skip saving vector snapshot: empty store.")
            return False

        snapshot_file = Path(snapshot_path)
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)

        payload_docs: List[Dict[str, Any]] = []
        for doc in self.documents:
            raw_metadata = dict(doc.metadata or {})
            safe_metadata = self._sanitize_snapshot_value(raw_metadata)
            if not isinstance(safe_metadata, dict):
                safe_metadata = {"_raw": str(safe_metadata)}
            payload_docs.append(
                {
                    "page_content": doc.page_content,
                    "metadata": safe_metadata,
                }
            )

        safe_meta = self._sanitize_snapshot_value(dict(metadata or {}))
        if not isinstance(safe_meta, dict):
            safe_meta = {"_raw": str(safe_meta)}

        payload = {
            "version": VECTOR_SNAPSHOT_VERSION,
            "metadata": safe_meta,
            "documents": payload_docs,
            "embeddings": np.array(self.embeddings_np, dtype="float32"),
        }

        tmp_file = snapshot_file.with_suffix(snapshot_file.suffix + ".tmp")
        try:
            with open(tmp_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_file, snapshot_file)
            logger.info(
                "Saved vector snapshot: %s (docs=%s)",
                snapshot_file,
                len(payload_docs),
            )
            return True
        except Exception as e:
            logger.warning("Failed to save vector snapshot %s: %s", snapshot_file, e)
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
            return False

    def load_snapshot(
        self,
        snapshot_path: str | Path,
        expected_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load vector state from disk. Returns snapshot metadata when successful.
        If expected_signature is provided, only restores when signature matches.
        """
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.exists():
            return None

        try:
            with open(snapshot_file, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to read vector snapshot %s: %s", snapshot_file, e)
            return None

        if not isinstance(payload, dict):
            logger.warning("Vector snapshot has invalid payload type: %s", type(payload))
            return None

        if payload.get("version") != VECTOR_SNAPSHOT_VERSION:
            logger.info(
                "Skip vector snapshot %s due to version mismatch: %s != %s",
                snapshot_file,
                payload.get("version"),
                VECTOR_SNAPSHOT_VERSION,
            )
            return None

        meta = payload.get("metadata")
        if not isinstance(meta, dict):
            meta = {}

        if expected_signature is not None:
            snapshot_signature = str(meta.get("resource_signature") or "")
            if snapshot_signature != str(expected_signature):
                logger.info(
                    "Skip vector snapshot %s due to signature mismatch.",
                    snapshot_file,
                )
                return None

        raw_docs = payload.get("documents")
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_docs, list) or raw_embeddings is None:
            logger.warning("Vector snapshot %s missing documents/embeddings.", snapshot_file)
            return None

        emb_np = np.array(raw_embeddings, dtype="float32")
        if emb_np.ndim == 1:
            emb_np = np.expand_dims(emb_np, axis=0)
        if emb_np.ndim != 2 or emb_np.shape[0] != len(raw_docs):
            logger.warning(
                "Vector snapshot %s shape mismatch: embeddings=%s docs=%s",
                snapshot_file,
                emb_np.shape,
                len(raw_docs),
            )
            return None

        documents: List[Document] = []
        for item in raw_docs:
            if not isinstance(item, dict):
                continue
            page_content = str(item.get("page_content") or "")
            doc_metadata = item.get("metadata")
            metadata_dict: Dict[str, Any] = dict(doc_metadata or {}) if isinstance(doc_metadata, dict) else {}
            if "_norm_text" not in metadata_dict:
                metadata_dict["_norm_text"] = normalize_for_match(page_content)
            documents.append(Document(page_content=page_content, metadata=metadata_dict))

        if len(documents) != emb_np.shape[0]:
            logger.warning(
                "Vector snapshot %s contains invalid document rows: docs=%s embeddings=%s",
                snapshot_file,
                len(documents),
                emb_np.shape[0],
            )
            return None

        self.documents = documents
        self.embeddings_np = emb_np
        self._rebuild_index()
        logger.info("Loaded vector snapshot: %s (docs=%s)", snapshot_file, len(documents))
        return meta

    def _embed_query(self, query: str) -> np.ndarray:
        q_embedding = self.embedder.embed_query(query)
        q_embedding = np.array(q_embedding, dtype="float32")
        if q_embedding.ndim == 1:
            q_embedding = np.expand_dims(q_embedding, axis=0)
        q_norm = np.linalg.norm(q_embedding, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        return q_embedding / q_norm

    def retrieve(self, query: str, top_k=25, threshold=SIMILARITY_THRESHOLD, file_ids: List[str] | None = None) -> List[Document]:
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
                        boost += 2.0  # strong boost for multi-word match (e.g., course name, "loáº¡i há»c pháº§n")
                    if token_hits:
                        boost += min(0.05 * token_hits, 0.35)  # bounded token bonus

                    prev = combined_scores.get(idx, 0.0)
                    combined_scores[idx] = prev + boost
                    # logger.info(f"[DEBUG] Lexical boost chunk {idx + 1}: base={prev:.4f} boost={boost:.4f} phrase_hit={phrase_hit} token_hits={token_hits}")

        # --- SPECIAL BOOST FOR SUBJECT CODES ---
        # Regex for subject codes: 3 letters + 4 digits + optional 1 letter (e.g. INT3306, PEC1008, INT3420E)
        subject_code_pattern = r"\b[A-Z]{3}\d{4}[A-Z]?\b"
        subject_codes_in_query = set(re.findall(subject_code_pattern, query.upper()))
        
        if subject_codes_in_query:
            logger.info(f"[DEBUG] Valid Subject Codes in Query: {subject_codes_in_query}")
            for idx, doc in enumerate(self.documents):
                if allowed_set and doc.metadata.get("file_id") not in allowed_set:
                    continue
                
                # Check if ANY subject code from query exists in the document content
                # Use raw content check for accuracy
                content_upper = doc.page_content.upper()
                for code in subject_codes_in_query:
                    if code in content_upper:
                        # MASSIVE BOOST to ensure it bubbles to the top
                        prev = combined_scores.get(idx, 0.0)
                        # Boost by +3.0 is usually enough to overtake any semantic noise
                        new_score = prev + 3.0
                        combined_scores[idx] = new_score
                        logger.info(f"[DEBUG] ðŸš€ SUBJECT CODE BOOST for Chunk {idx + 1}: {code} found -> +3.0 (Score: {new_score:.4f})")

        # --- HEURISTIC BOOST FOR DEFINITIONS (HANDBOOK/REGULATIONS) & SCHEDULE PENALTY ---
        # 1. Update Definition Patterns: Broaden to include general concepts, regulations, and exemption/certificates
        definition_patterns = [
            r"lÃ  gÃ¬", r"Ä‘á»‹nh nghÄ©a", r"gá»“m nhá»¯ng gÃ¬", r"tháº¿ nÃ o lÃ ", r"nhÆ° tháº¿ nÃ o",
            r"quy cháº¿", r"quy Ä‘á»‹nh", r"Ä‘iá»u kiá»‡n", r"bao nhiÃªu tÃ­n", r"há»c láº¡i", 
            r"cáº£i thiá»‡n", r"Ä‘Äƒng kÃ½", r"há»§y", r"tiÃªn quyáº¿t", r"miá»…n", r"chá»©ng chá»‰", 
            r"ngoáº¡i ngá»¯", r"ielts", r"toeic", r"cÃ¡c loáº¡i", r"danh sÃ¡ch", r"cáº¥u trÃºc"
        ]
        is_general_query = any(re.search(p, query.lower()) for p in definition_patterns)
        language_patterns = [
            r"ielts", r"toeic", r"toefl", r"vstep", r"aptis", r"cambridge",
            r"ngoáº¡i ngá»¯", r"chuan dau ra ngoai ngu", r"chuáº©n Ä‘áº§u ra ngoáº¡i ngá»¯",
            r"\bbac\s*3\b", r"\bbac\s*4\b", r"\bbac\s*5\b", r"knlnn"
        ]
        is_language_query = any(re.search(p, query.lower()) for p in language_patterns)
        
        # 2. Check for Schedule Intent (time/location/teacher/class lookup from timetable)
        # Use normalized query to support both accented and non-accented Vietnamese.
        schedule_intent_patterns = [
            r"\blich\b",
            r"\bthoi khoa bieu\b",
            r"\btkb\b",
            r"\bphong\b",
            r"\bthu\b",
            r"\btiet\b",
            r"\bca\b",
            r"\bgio\b",
            r"\bbao gio\b",
            r"\bo dau\b",
            r"\blop\b",
            r"\bmo lop\b",
        ]
        teacher_intent_patterns = [
            r"\bgiang vien\b",
            r"\bgv\b",
            r"\bai day\b",
            r"\bco ai day\b",
            r"\bthay nao day\b",
            r"\bco nao day\b",
        ]
        is_schedule_query = any(re.search(p, norm_query) for p in schedule_intent_patterns) or any(
            re.search(p, norm_query) for p in teacher_intent_patterns
        )
        is_teacher_query = any(re.search(p, norm_query) for p in teacher_intent_patterns)
        
        # Determine Boost/Penalty Strategy
        authority_keywords = ["Sá»” TAY", "QUY CHáº¾", "QUY Äá»ŠNH", "HÆ¯á»šNG DáºªN"]
        schedule_keywords = ["tkb", "thoi khoa bieu", "lich hoc", "phu luc"]

        for idx, doc in enumerate(self.documents):
            if allowed_set and doc.metadata.get("file_id") not in allowed_set:
                continue
            
            source_name = str(doc.metadata.get("source", ""))
            file_id = str(doc.metadata.get("file_id", ""))
            source_name_norm = doc.metadata.get("_source_norm") or normalize_for_match(source_name)
            file_id_norm = doc.metadata.get("_file_id_norm") or normalize_for_match(file_id)
            doc.metadata["_source_norm"] = source_name_norm
            doc.metadata["_file_id_norm"] = file_id_norm
            
            # A. AUTHORITY BOOST (Handbook wins for general/policy queries)
            source_name_upper = source_name.upper()
            file_id_upper = file_id.upper()
            is_authority = any(k in source_name_upper or k in file_id_upper for k in authority_keywords)
            if is_authority and is_general_query:
                prev = combined_scores.get(idx, 0.0)
                new_score = prev + 1.5 # Boost authoritative sources for definitions/policy
                combined_scores[idx] = new_score
                # logger.info(f"[DEBUG] ðŸ“– AUTHORITY BOOST for Chunk {idx + 1}: +1.5")

            # B. SCHEDULE PENALTY (Schedule loses if query is NOT strictly about schedule)
            # Logic: If chunk comes from a Schedule file, BUT the query doesn't look like a schedule request -> PENALIZE
            norm_doc = doc.metadata.get("_norm_text") or normalize_for_match(doc.page_content)
            doc.metadata["_norm_text"] = norm_doc
            is_schedule_doc = any(k in source_name_norm or k in file_id_norm for k in schedule_keywords) or (
                doc.metadata.get("type") == "table" and "thu " in norm_doc
            )
            
            if is_schedule_doc and not is_schedule_query:
                prev = combined_scores.get(idx, 0.0)
                penalty = 0.5
                new_score = prev - penalty
                combined_scores[idx] = new_score
                # logger.info(f"[DEBUG] ðŸ“‰ SCHEDULE PENALTY for Chunk {idx + 1}: -{penalty} (Query not requesting schedule)")
            elif is_schedule_doc and is_teacher_query:
                # Teacher lookup queries (e.g. "mon X co nhung ai day") should prioritize timetable chunks.
                prev = combined_scores.get(idx, 0.0)
                combined_scores[idx] = prev + 0.9

            # C. LANGUAGE MAPPING BOOST (IELTS/TOEFL/TOEIC/VSTEP)
            if is_language_query:
                has_test_mapping_signal = any(
                    token in norm_doc for token in [
                        "ielts", "toeic", "toefl", "vstep", "aptis", "cambridge",
                        "bang tham chieu", "knlnnvn", "bac 3", "bac 4", "bac 5"
                    ]
                )
                if has_test_mapping_signal:
                    prev = combined_scores.get(idx, 0.0)
                    extra = 2.2
                    if "bang tham chieu" in norm_doc or "knlnnvn" in norm_doc:
                        extra += 1.0
                    combined_scores[idx] = prev + extra
                else:
                    # Slightly suppress unrelated HTML chunks for language-equivalence questions.
                    is_html_chunk = source_name_upper.endswith(".HTML") or file_id_upper.endswith(".HTML")
                    if is_html_chunk and not is_authority:
                        prev = combined_scores.get(idx, 0.0)
                        combined_scores[idx] = prev - 0.4

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
    Parse curriculum HTML into hierarchical blocks/sub-blocks/subjects.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    def _parse_credit_token(value: str) -> int:
        text = (value or "").strip()
        if not text:
            return 0
        norm = normalize_for_match(text)

        # Only accept dedicated credit-cell formats to avoid false positives from names
        # like "(... tá»« 11 Ä‘áº¿n 14)" in block descriptions.
        if not (
            re.fullmatch(r"\d{1,3}(?:\s*/\s*\d{1,3})?", text)
            or re.fullmatch(r"\d{1,3}\s*tin\s*chi", norm)
        ):
            return 0

        match = re.search(r"\b(\d{1,3})\b", text)
        if not match:
            return 0
        try:
            return int(match.group(1))
        except Exception:
            return 0

    def _extract_block_name(values: List[str], start_idx: int = 1) -> str:
        for raw in values[start_idx:]:
            text = (raw or "").strip()
            if not text:
                continue
            if text in {"+", "-"}:
                continue
            if _parse_credit_token(text) > 0:
                continue
            return text
        return ""

    def _extract_required_credits(values: List[str], start_idx: int = 2) -> int:
        for raw in values[start_idx:]:
            parsed = _parse_credit_token(raw)
            if parsed > 0:
                return parsed
        return 0

    target_table = None
    for table in soup.find_all("table"):
        table_norm = normalize_for_match(table.get_text(separator=" ", strip=True))
        if "ma" in table_norm and "hoc phan" in table_norm and "so tin chi" in table_norm:
            target_table = table
            break

    if target_table is None:
        best_score = -1
        for table in soup.find_all("table"):
            raw_text = table.get_text(separator=" ", strip=True)
            score = len(re.findall(r"\b[A-Z]{2,6}\d{4}[A-Z]?\b", raw_text.upper()))
            if score > best_score:
                best_score = score
                target_table = table

    if target_table is None:
        logger.warning("[Curriculum Parsing] Could not find curriculum table in HTML.")
        return []

    rows = target_table.find_all("tr")

    structure: List[dict] = []
    current_block: dict | None = None
    current_sub_block: dict | None = None

    block_pattern = re.compile(r"^[IVX]+\s*$")
    sub_block_pattern = re.compile(r"^[IVX]+\.\d+(?:\.\d+)?\s*$")
    # Support both legacy codes (INT3306, MAT1101, INT3404E)
    # and prefixed modern codes (UET.CE2020, UET.AI3064, UET.DSE3155).
    subject_code_pattern = re.compile(r"^(?:[A-Z]{2,8}\.)?[A-Z]{2,8}\d{4}[A-Z]?$")
    note_tokens = (
        "nhom nganh",
        "tu chon",
        "lua chon",
        "bo tro",
        "dien tu",
        "vien thong",
        "kinh te",
        "luat",
    )

    for row in rows:
        cols = row.find_all("td")
        if not cols:
            continue

        col_texts = [c.get_text(separator=" ", strip=True) for c in cols]
        col_norms = [normalize_for_match(text) for text in col_texts]
        if not col_texts:
            continue

        if any("so tin chi" in item for item in col_norms):
            continue

        first_col = col_texts[0].strip()
        first_col_norm = normalize_for_match(first_col).upper().replace(" ", "")

        if block_pattern.match(first_col_norm):
            block_name = _extract_block_name(col_texts, start_idx=1)
            credits = _extract_required_credits(col_texts, start_idx=2)

            current_block = {
                "id": first_col_norm,
                "name": block_name,
                "required_credits": credits,
                "type": "main",
                "subjects": [],
                "sub_blocks": [],
            }
            structure.append(current_block)
            current_sub_block = None
            continue

        if sub_block_pattern.match(first_col_norm):
            sub_name = _extract_block_name(col_texts, start_idx=1)
            credits = _extract_required_credits(col_texts, start_idx=2)

            current_sub_block = {
                "id": first_col_norm,
                "name": sub_name,
                "required_credits": credits,
                "type": "sub",
                "subjects": [],
                "sub_blocks": [],
            }
            if current_block:
                current_block["sub_blocks"].append(current_sub_block)
            continue

        subject_code = ""
        subject_code_idx = -1
        for idx, raw in enumerate(col_texts[:5]):
            candidate = raw.strip().upper().replace(" ", "")
            if subject_code_pattern.match(candidate):
                subject_code = candidate
                subject_code_idx = idx
                break

        if subject_code:
            name_idx = subject_code_idx + 1
            credit_idx = name_idx + 1
            name = col_texts[name_idx] if 0 <= name_idx < len(col_texts) else ""
            if not name and len(col_texts) > 2:
                name = col_texts[2]
            credits = _parse_credit_token(col_texts[credit_idx]) if 0 <= credit_idx < len(col_texts) else 0

            subject = {"code": subject_code, "name": name, "credits": credits}
            if current_sub_block:
                current_sub_block["subjects"].append(subject)
            elif current_block:
                current_block["subjects"].append(subject)
            continue

        note_text = None
        note_norm = None
        for raw, norm in zip(col_texts, col_norms):
            if len(norm) < 6:
                continue
            if any(token in norm for token in note_tokens):
                note_text = raw
                note_norm = norm
                break

        if note_text:
            note_item = {"text": note_text, "norm": note_norm or normalize_for_match(note_text)}
            target_node = current_sub_block if current_sub_block else current_block
            if target_node is not None:
                target_node.setdefault("notes", []).append(note_item)

    return structure


def compute_curriculum_missing_credits(structure: List[dict], completed_map: Dict[str, Any]) -> List[dict]:
    """
    Compute missing credits per curriculum block, including open-group external credits.
    External credits are only counted when block/sub-block notes explicitly allow them.
    """
    missing_details: List[Dict[str, Any]] = []
    used_codes: Set[str] = set()

    norm_completed_map: Dict[str, Dict[str, Any]] = {}
    for code, data in completed_map.items():
        norm_code = str(code or "").upper().replace(" ", "")
        if norm_code:
            norm_completed_map[norm_code] = data

    external_prefix_map: Dict[str, Set[str]] = {
        "kinh_te": {"BSA", "INE", "UEB", "MKT", "FIN"},
        "luat": {"LAW", "JUS", "THL", "LLM"},
        "dien_tu_vien_thong": {"ELT", "ECE", "FET"},
    }

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _normalize_code(value: str) -> str:
        return str(value or "").upper().replace(" ", "")

    def _notes_to_norm(notes: List[Any]) -> str:
        parts: List[str] = []
        for note in notes or []:
            if isinstance(note, dict):
                part = str(note.get("norm") or normalize_for_match(str(note.get("text") or ""))).strip()
            else:
                part = normalize_for_match(str(note))
            if part:
                parts.append(part)
        return " ".join(parts).strip()

    def _classify_block_type(name: str, notes_norm: str = "") -> str:
        name_norm = normalize_for_match(name)
        if "tu chon" in name_norm or "lua chon" in name_norm or "bo tro" in name_norm:
            return "elective"
        if any(token in notes_norm for token in ("nhom nganh", "kinh te", "luat", "dien tu", "vien thong")):
            return "elective"
        return "required"

    def _eligible_external_subjects(notes_norm: str, excluded_codes: Set[str]) -> List[Dict[str, Any]]:
        families: Set[str] = set()
        if "kinh te" in notes_norm or "marketing" in notes_norm:
            families.add("kinh_te")
        if "luat" in notes_norm:
            families.add("luat")
        if "dien tu" in notes_norm or "vien thong" in notes_norm:
            families.add("dien_tu_vien_thong")

        if not families:
            return []

        matches: List[Dict[str, Any]] = []
        for raw_code, subj in completed_map.items():
            norm_code = _normalize_code(raw_code)
            if not norm_code or norm_code in excluded_codes:
                continue

            name_norm = normalize_for_match(str(subj.get("name") or ""))
            prefix = norm_code[:3]
            matched = False

            if "kinh_te" in families and (
                prefix in external_prefix_map["kinh_te"]
                or "kinh te" in name_norm
                or "marketing" in name_norm
            ):
                matched = True

            if (not matched) and "luat" in families and (
                prefix in external_prefix_map["luat"]
                or "luat" in name_norm
                or "phap luat" in name_norm
            ):
                matched = True

            if (not matched) and "dien_tu_vien_thong" in families and (
                prefix in external_prefix_map["dien_tu_vien_thong"]
                or "dien tu" in name_norm
                or "vien thong" in name_norm
            ):
                matched = True

            if matched:
                matches.append(subj)

        matches.sort(key=lambda x: _safe_int(x.get("credits")), reverse=True)
        return matches

    def _append_result(
        block_name: str,
        block_type: str,
        required_credits: int,
        completed_credits: int,
        candidates: List[Dict[str, Any]],
        applied_external_subjects: List[Dict[str, Any]],
        block_id: Optional[str] = None,
        notes_norm: str = "",
    ):
        missing_credits = max(required_credits - completed_credits, 0)
        result = {
            "block_id": block_id,
            "block_name": block_name,
            "block_type": block_type,
            "required_credits": required_credits,
            "completed_credits": completed_credits,
            "missing_credits": missing_credits,
            "candidates": candidates,
            "applied_external_subjects": applied_external_subjects,
            "notes_norm": notes_norm,
        }

        if missing_credits > 0 or applied_external_subjects:
            missing_details.append(result)

    for block in structure or []:
        block_name = str(block.get("name") or "")
        block_notes_norm = _notes_to_norm(block.get("notes") or [])
        sub_blocks = block.get("sub_blocks") or []

        if not sub_blocks:
            required = _safe_int(block.get("required_credits"))
            completed = 0
            candidates: List[Dict[str, Any]] = []

            for subj in block.get("subjects") or []:
                norm_code = _normalize_code(subj.get("code"))
                if not norm_code:
                    continue

                user_sub = norm_completed_map.get(norm_code)
                if user_sub is None:
                    candidates.append(subj)
                    continue

                original_code = _normalize_code(user_sub.get("code"))
                if not original_code or original_code in used_codes:
                    continue

                gained = _safe_int(user_sub.get("credits")) or _safe_int(subj.get("credits"))
                completed += gained
                used_codes.add(original_code)

            _append_result(
                block_name=block_name,
                block_type=_classify_block_type(block_name, block_notes_norm),
                required_credits=required,
                completed_credits=completed,
                candidates=candidates,
                applied_external_subjects=[],
                block_id=block.get("id"),
                notes_norm=block_notes_norm,
            )
            continue

        buckets: List[Dict[str, Any]] = []
        current_bucket: Optional[Dict[str, Any]] = None

        for sub in sub_blocks:
            sub_required = _safe_int(sub.get("required_credits"))
            if sub_required > 0:
                current_bucket = {
                    "id": sub.get("id"),
                    "name": sub.get("name"),
                    "required_credits": sub_required,
                    "subjects": list(sub.get("subjects") or []),
                    "notes": list(sub.get("notes") or []),
                }
                buckets.append(current_bucket)
            elif current_bucket is not None:
                current_bucket["subjects"].extend(list(sub.get("subjects") or []))
                current_bucket["notes"].extend(list(sub.get("notes") or []))

        for bucket in buckets:
            required = _safe_int(bucket.get("required_credits"))
            completed = 0
            candidates: List[Dict[str, Any]] = []
            bucket_used: Set[str] = set()

            for subj in bucket.get("subjects") or []:
                norm_code = _normalize_code(subj.get("code"))
                if not norm_code:
                    continue

                user_sub = norm_completed_map.get(norm_code)
                if user_sub is None:
                    candidates.append(subj)
                    continue

                original_code = _normalize_code(user_sub.get("code"))
                if not original_code or original_code in used_codes or original_code in bucket_used:
                    continue

                gained = _safe_int(user_sub.get("credits")) or _safe_int(subj.get("credits"))
                completed += gained
                bucket_used.add(original_code)
                used_codes.add(original_code)

            bucket_notes_norm = _notes_to_norm(bucket.get("notes") or [])
            notes_norm = " ".join(p for p in [block_notes_norm, bucket_notes_norm] if p).strip()

            applied_external_subjects: List[Dict[str, Any]] = []
            if completed < required:
                external_pool = _eligible_external_subjects(notes_norm, used_codes | bucket_used)
                for external_subj in external_pool:
                    remaining = required - completed
                    if remaining <= 0:
                        break

                    external_code = _normalize_code(external_subj.get("code"))
                    external_credits = _safe_int(external_subj.get("credits"))
                    if not external_code or external_credits <= 0:
                        continue

                    counted_credits = min(external_credits, remaining)
                    completed += counted_credits
                    used_codes.add(external_code)

                    applied_external_subjects.append(
                        {
                            "code": external_code,
                            "name": external_subj.get("name"),
                            "credits": external_credits,
                            "counted_credits": counted_credits,
                        }
                    )

            bucket_name = str(bucket.get("name") or "")
            merged_name = f"{block_name} - {bucket_name}" if block_name else bucket_name
            _append_result(
                block_name=merged_name,
                block_type=_classify_block_type(bucket_name, notes_norm),
                required_credits=required,
                completed_credits=completed,
                candidates=candidates,
                applied_external_subjects=applied_external_subjects,
                block_id=bucket.get("id"),
                notes_norm=notes_norm,
            )

    return missing_details

