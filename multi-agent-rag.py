import sys
import numpy as np
import urllib.parse
import requests
from sentence_transformers import SentenceTransformer
import ollama
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# ========= Data & Retrieval Setup =========
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data for Collection A (cat-facts.txt)
with open('cat-facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset_lines_a = f.readlines()

# Load data for Collection B (facts.txt)
with open('facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset_lines_b = f.readlines()

# Create chunks for Collection A
chunks_a = []
window_size = 3
for line in dataset_lines_a:
    sentences = sent_tokenize(line.strip())
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks_a.append(chunk)
print(f'Created {len(chunks_a)} chunks for Collection A')

# Create chunks for Collection B
chunks_b = []
for line in dataset_lines_b:
    sentences = sent_tokenize(line.strip())
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks_b.append(chunk)
print(f'Created {len(chunks_b)} chunks for Collection B')

# Build vector databases
VECTOR_DB_A = []
VECTOR_DB_B = []

def add_to_vector_db(chunk, db):
    emb = embedding_model.encode(chunk).tolist()
    db.append((chunk, emb))

for chunk in chunks_a:
    add_to_vector_db(chunk, VECTOR_DB_A)
for chunk in chunks_b:
    add_to_vector_db(chunk, VECTOR_DB_B)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def retrieve(query, db, top_n=3):
    query_emb = embedding_model.encode(query).tolist()
    sims = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in db]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

# ========= Tools: Web Search =========
API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"  # Lưu ý: Nên lưu trữ an toàn

def web_search(query, num_results=3):
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets if snippets else ["(No web results)"]
    except Exception as e:
        print(f"Error in web search: {e}")
        return ["(No web results)"]

# ========= Generation with LLM =========
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def generate_answer(context, query):
    prompt = f"""You are a helpful chatbot. Use ONLY the following aggregated context to answer the question:
{context}

Question: {query}

Answer:"""
    response = ""
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True
        )
        for part in stream:
            response += part['message']['content']
    except Exception as e:
        print(f"Error in generation: {e}")
        response = "Unable to generate answer."
    return response.strip()

# ========= Multi-Agent RAG Agents =========
def internal_data_agent(query):
    """Agent thu thập thông tin từ Collection A và B."""
    results_a = retrieve(query, VECTOR_DB_A, top_n=3)
    results_b = retrieve(query, VECTOR_DB_B, top_n=3)
    context_a = "\n".join([f"- {chunk}" for chunk, _ in results_a])
    context_b = "\n".join([f"- {chunk}" for chunk, _ in results_b])
    aggregated_internal = "\n".join([context_a, context_b])
    print("Internal Data Agent output:")
    print(aggregated_internal)
    return aggregated_internal

def web_search_agent(query):
    """Agent thực hiện tìm kiếm trên web."""
    snippets = web_search(query, num_results=3)
    context_web = "\n".join([f"- {snippet}" for snippet in snippets])
    print("Web Search Agent output:")
    print(context_web)
    return context_web

def aggregator_agent(query):
    """Agent tổng hợp kết quả từ Internal Data và Web Search."""
    internal_context = internal_data_agent(query)
    web_context = web_search_agent(query)
    aggregated_context = "\n".join([internal_context, web_context])
    print("Aggregator Agent output (Aggregated Context):")
    print(aggregated_context)
    return aggregated_context

def final_generation_agent(query, aggregated_context):
    """Agent cuối cùng dùng context tổng hợp để sinh câu trả lời cuối cùng."""
    final_answer = generate_answer(aggregated_context, query)
    return final_answer

# ========= Main Execution =========
if __name__ == "__main__":
    user_query = input("Ask me a question: ")
    aggregated_context = aggregator_agent(user_query)
    final_answer = final_generation_agent(user_query, aggregated_context)
    print("\n=== Final Response ===")
    print(final_answer)
