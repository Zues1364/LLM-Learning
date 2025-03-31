# Sử dụng sentence-window chunking thay vì context-aware chunking dạng ký tự
import os
import tempfile
from datetime import datetime
from typing import List, Dict
import urllib.parse
import requests
import sys
import numpy as np
import faiss
import fitz  # PyMuPDF

import google.generativeai as genai
import bs4
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_core.embeddings import Embeddings
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- Cấu hình API key ---
GEMINI_API_KEY = ""
SERPER_API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"

# --- Web Search Tool ---
def web_search(query, num_results=3):
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={SERPER_API_KEY}"
    try:
        response = requests.get(url)
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets if snippets else ["(Không có kết quả tìm kiếm)"]
    except Exception as e:
        print(f"Lỗi trong tìm kiếm web: {e}")
        return ["(Không có kết quả tìm kiếm)"]

# --- Gemini Embedder ---
class GeminiEmbedder(Embeddings):
    def __init__(self, model_name="models/text-embedding-004"):
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

# --- Constants ---
PDF_PATH = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"
SIMILARITY_THRESHOLD = 0.5

# --- Sentence-Window Chunking ---
def sentence_window_chunking(text: str, window_size: int = 5, stride: int = 3) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i:i + window_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- Document Processing Functions ---
def process_pdf(file_path: str) -> List:
    try:
        doc = fitz.open(file_path)
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text()
            page_chunks = sentence_window_chunking(text, window_size=5, stride=3)
            for chunk in page_chunks:
                document = type("Document", (), {})()
                document.page_content = chunk
                document.metadata = {
                    "page": i + 1,
                    "file_name": os.path.basename(file_path),
                    "timestamp": datetime.now().isoformat()
                }
                pages.append(document)
        return pages
    except Exception as e:
        print(f"Lỗi xử lý PDF với PyMuPDF: {e}")
        sys.exit(1)

# --- FAISS Vector Store ---
class FAISSVectorStore:
    def __init__(self, documents: List, embedder: GeminiEmbedder):
        self.documents = documents
        self.embedder = embedder
        self.embeddings = [embedder.embed_query(doc.page_content) for doc in documents]
        self.embeddings_np = np.array(self.embeddings).astype("float32")
        norms = np.linalg.norm(self.embeddings_np, axis=1, keepdims=True)
        self.embeddings_np = self.embeddings_np / norms
        d = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings_np)

    def retrieve(self, query: str, top_k=10, threshold=SIMILARITY_THRESHOLD):
        q_embedding = self.embedder.embed_query(query)
        q_embedding = np.array(q_embedding, dtype="float32")
        q_norm = np.linalg.norm(q_embedding)
        if q_norm > 0:
            q_embedding = q_embedding / q_norm
        q_embedding = np.expand_dims(q_embedding, axis=0)
        D, I = self.index.search(q_embedding, top_k)
        results = []
        for idx, sim in zip(I[0], D[0]):
            if sim >= threshold:
                results.append(self.documents[idx])
        return results

# --- Cosine Similarity (optional for diagnostics) ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

# --- Agents ---
def get_query_rewriter_agent() -> Agent:
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions=(
            "Bạn là chuyên gia trong việc viết lại câu hỏi để trở nên chính xác và chi tiết hơn. "
            "Phân tích câu hỏi của người dùng và viết lại sao cho cụ thể và dễ tìm kiếm hơn. "
            "Trả về CHỈ câu hỏi đã được viết lại."
        ),
        show_tool_calls=False,
        markdown=True,
    )

def get_rag_agent() -> Agent:
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions=(
            "Bạn là một đại lý thông minh chuyên cung cấp câu trả lời chính xác.\n"
            "Khi được cung cấp bối cảnh từ tài liệu:\n"
            "- Tập trung vào thông tin từ các tài liệu đã cho\n"
            "- Trả lời chính xác và trích dẫn chi tiết cụ thể\n"
            "Khi được cung cấp kết quả tìm kiếm web:\n"
            "- Chỉ rõ rằng thông tin đến từ tìm kiếm web\n"
            "- Tổng hợp thông tin một cách rõ ràng\n"
            "Luôn duy trì độ chính xác và độ rõ ràng cao trong câu trả lời của bạn."
        ),
        show_tool_calls=True,
        markdown=True,
    )

# --- Main Execution ---
def main():
    print("Đang xử lý file PDF...")
    documents = process_pdf(PDF_PATH)
    print(f"Đã tạo {len(documents)} chunk từ file PDF.")

    embedder = GeminiEmbedder()
    vector_store = FAISSVectorStore(documents, embedder)

    query = input("Nhập câu hỏi của bạn: ").strip()
    if not query:
        print("Không có câu hỏi nào được nhập. Thoát.")
        return

    try:
        query_rewriter = get_query_rewriter_agent()
        rewritten = query_rewriter.run(query).content.strip()
        print("\n--- Viết lại câu hỏi ---")
        print(f"Câu hỏi gốc: {query}")
        print(f"Câu hỏi sau khi viết lại: {rewritten}")
    except Exception as e:
        print(f"Lỗi khi viết lại câu hỏi: {e}")
        rewritten = query

    retrieved_docs = vector_store.retrieve(rewritten)
    context = ""
    if retrieved_docs:
        for doc in retrieved_docs:
            print(f"Đoạn được truy xuất từ trang: {doc.metadata.get('page')}")
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"\nĐã tìm thấy {len(retrieved_docs)} đoạn có liên quan.")
    else:
        print("Không tìm thấy thông tin phù hợp trong tài liệu. Sẽ thử tìm kiếm trên web...")
        snippets = web_search(rewritten, num_results=3)
        context = "Kết quả tìm kiếm web:\n" + "\n".join(snippets)

    try:
        rag_agent = get_rag_agent()
        if context:
            full_prompt = (
                f"Bối cảnh: {context}\n\n"
                f"Câu hỏi gốc: {query}\n"
                f"Câu hỏi sau khi viết lại: {rewritten}\n\n"
                "Hãy cung cấp một câu trả lời chi tiết dựa trên thông tin có sẵn."
            )
        else:
            full_prompt = f"Câu hỏi gốc: {query}\nCâu hỏi sau khi viết lại: {rewritten}"
            print("Không có bối cảnh truy xuất được.")

        print("\nĐang sinh câu trả lời...")
        response = rag_agent.run(full_prompt)
        answer = response.content
        print("\n=== Câu trả lời ===")
        print(answer)
    except Exception as e:
        print(f"Lỗi khi sinh câu trả lời: {e}")

if __name__ == "__main__":
    main()
