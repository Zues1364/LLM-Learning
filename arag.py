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

# ========= 1) Data & Retrieval Setup =========
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


with open('cat-facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset_lines_a = f.readlines()


with open('facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset_lines_b = f.readlines()


chunks_a = []
window_size = 3
for line in dataset_lines_a:
    sentences = sent_tokenize(line.strip())
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks_a.append(chunk)
print(f'Created {len(chunks_a)} chunks for Collection A')


chunks_b = []
for line in dataset_lines_b:
    sentences = sent_tokenize(line.strip())
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks_b.append(chunk)
print(f'Created {len(chunks_b)} chunks for Collection B')





def add_to_vector_db(chunk, db):
    emb = embedding_model.encode(chunk).tolist()
    db.append((chunk, emb))

VECTOR_DB_A = []
for chunk in chunks_a:
    add_to_vector_db(chunk, VECTOR_DB_A)


VECTOR_DB_B = []
for chunk in chunks_b:
    add_to_vector_db(chunk, VECTOR_DB_B)


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def retrieve(query, db, top_n=3):
    query_emb = embedding_model.encode(query).tolist()
    similarities = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in db]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


# ========= 2) Tools =========
# Web Search Tool
API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"


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



# ========= 3) Memory =========
class Memory:
    def __init__(self):
        self.short_term = []  # Lưu trữ lịch sử ngắn hạn (mới nhất)
        self.long_term = []  # Lưu trữ lịch sử dài hạn

    def add_to_short_term(self, query, response):
        self.short_term.append((query, response))
        if len(self.short_term) > 5:  # Giới hạn short-term memory
            self.long_term.append(self.short_term.pop(0))

    def get_context(self):
        # Kết hợp short-term và long-term để tạo context
        context = "\n".join([f"Query: {q}\nResponse: {r}" for q, r in self.short_term])
        if self.long_term:
            context += "\nPrevious History:\n" + "\n".join([f"Query: {q}\nResponse: {r}" for q, r in self.long_term])
        return context


# ========= 4) Retriever Agent (Router) =========
def retriever_agent(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["search", "find", "lookup", "information about"]):
        print("Routing to Web Search...")
        return "web_search", "\n".join(web_search(query))
    if "cat" in query_lower:
        print("Routing to Vector Search Engine A (Cat Facts)...")
        results = retrieve(query, VECTOR_DB_A)
        return "vector_search_a", "\n".join([f"- {chunk}" for chunk, _ in results])
    else:
        print("Routing to Vector Search Engine B (General Facts)...")
        results = retrieve(query, VECTOR_DB_B)
        return "vector_search_b", "\n".join([f"- {chunk}" for chunk, _ in results])


# ========= 5) Planning (Reflection and Self-critics) =========
def reflect_and_critique(query, response, memory):
    query_emb = embedding_model.encode(query)
    response_emb = embedding_model.encode(response)
    similarity = cosine_similarity(query_emb, response_emb)

    if similarity < 0.7:
        print("Self-critics: Response is not relevant enough. Refining query...")
        history = memory.get_context()
        refined_query = f"Based on history:\n{history}\nRefine the query: {query}"
        return refined_query
    return query


# ========= 6) Generation with LLM =========
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


def generate_answer(context, query, memory):
    memory_context = memory.get_context()
    prompt = f"""You are a helpful chatbot. Use the following context and memory to answer the question:
Context:
{context}

Memory:
{memory_context}

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


# ========= 7) Agent with ReAct Loop =========
def agent(query, max_iterations=10):
    memory = Memory()
    current_query = query
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")


        tool, context = retriever_agent(current_query)
        print(f"Context from {tool}:\n{context}")


        response = generate_answer(context, current_query, memory)
        print(f"Response:\n{response}")


        memory.add_to_short_term(current_query, response)

        # Reflection and Self-critics
        refined_query = reflect_and_critique(current_query, response, memory)
        if refined_query == current_query:
            print("Response is relevant. Stopping.")
            return response
        else:
            current_query = refined_query
            print(f"Refined Query:\n{current_query}")

    print("Reached maximum iterations. Returning last response.")
    return response


# ========= 8) Main Execution =========
if __name__ == "__main__":
    query = input("Ask me a question: ")
    final_response = agent(query)
    print("\n=== Final Response ===")
    print(final_response)