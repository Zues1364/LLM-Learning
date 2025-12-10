import json
import os
from pathlib import Path
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from datetime import datetime
from typing import List
import logging
import urllib.parse
import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import re

from unstructured.partition.pdf import partition_pdf

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.3
# Absolute cache dir to avoid CWD mismatch between services
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"

# Prepare cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Embedding cache naming
EMB_SUFFIX = "_embeddings.npy"
EMB_META_SUFFIX = "_embeddings_meta.json"


# Helper: md5 hash of file
def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# PDF processing
def _clean_text(text: str) -> str:
    """Light clean to make table text more searchable."""
    text = text.replace("|", " | ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(\d)\s{2,}(\d)", r"\1 | \2", text)
    while "  " in text:
        text = text.replace("  ", " ")
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def process_pdf(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        logger.error(f"File PDF khong ton tai: {file_path}")
        raise FileNotFoundError(f"File PDF khong ton tai: {file_path}")

    pdf_name = os.path.basename(file_path)
    file_id = pdf_name  # use filename as source id
    cache_file = os.path.join(CACHE_DIR, f"{pdf_name}.pkl")
    cache_metadata_file = os.path.join(CACHE_DIR, f"{pdf_name}_metadata.pkl")

    pdf_hash = get_file_hash(file_path)

    if os.path.exists(cache_file) and os.path.exists(cache_metadata_file):
        try:
            with open(cache_metadata_file, "rb") as f:
                cached_hash = pickle.load(f)
            if cached_hash == pdf_hash:
                logger.info(f"Loading chunks from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Loaded {len(documents)} cached chunks.")
                return documents
            else:
                logger.info("PDF changed, re-processing...")
        except Exception as e:
            logger.error(f"Loi khi load cache: {e}, re-processing PDF...")

    logger.info(f"Processing PDF: {file_path}")

    def _partition(strategy: str, use_ocr: bool):
        return partition_pdf(
            filename=file_path,
            strategy=strategy,
            infer_table_structure=True if use_ocr else False,
            extract_images_in_pdf=True,
            languages=["vie"],
            ocr_languages=["vie"] if use_ocr else None,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            pdf_image_dpi=300,
        )

    raw_elements = _partition(strategy="hi_res", use_ocr=True)

    documents: List[Document] = []
    for i, element in enumerate(raw_elements):
        content = _clean_text(str(element))
        doc = Document(
            page_content=content,
            metadata={
                "index": i + 1,
                "file_name": pdf_name,
                "file_id": file_id,
                "source_path": file_path,
                "timestamp": datetime.now().isoformat(),
            },
        )
        documents.append(doc)

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)
        with open(cache_metadata_file, "wb") as f:
            pickle.dump(pdf_hash, f)
        logger.info(f"Saved {len(documents)} chunks to cache: {cache_file}")
    except Exception as e:
        logger.error(f"Loi khi luu cache: {e}")

    logger.info(f"Extracted {len(documents)} chunks from PDF.")
    return documents


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
            if meta.get("pdf_hash") == pdf_hash and meta.get("embedder") == embedder_name:
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
            json.dumps({"pdf_hash": pdf_hash, "embedder": embedder_name}, ensure_ascii=False),
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

        max_len = 100_000
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

        # Filter by file_ids if provided
        candidate_indices: List[int]
        if file_ids:
            allow = set(file_ids)
            candidate_indices = [i for i, doc in enumerate(self.documents) if doc.metadata.get("file_id") in allow]
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

        results = []
        found_any = False
        for idx, sim in scored:
            logger.info(f"[DEBUG] Chunk {idx + 1}: sim = {sim:.4f}, threshold = {threshold}")
            if sim >= threshold:
                results.append(self.documents[idx])
                found_any = True
            else:
                logger.info(f"[DEBUG] Chunk {idx + 1} dropped, sim {sim:.4f} < {threshold}")

        if not found_any and threshold > 0.05:
            logger.info(f"[DEBUG] No hits above {threshold}, retry with 0.05")
            fallback_threshold = 0.05
            for idx, sim in scored:
                if sim >= fallback_threshold:
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
