import sys



from sentence_transformers import SentenceTransformer
import numpy as np

import ollama
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# 1. Khởi tạo mô hình SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load dataset
dataset = []
with open('cat-facts.txt', 'r', encoding='utf-8', errors='ignore') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')
chunks = []
window_size = 3
for line in dataset:
    sentences = sent_tokenize(line)
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size])
        chunks.append(chunk)

# 3. Cấu hình cho phần chatbot
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# VECTOR_DB lưu (chunk, embedding)
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = embedding_model.encode(chunk)
    embedding = embedding.tolist()
    VECTOR_DB.append((chunk, embedding))

# 4. Thêm các chunk vào cơ sở dữ liệu vector
for i, chunk in enumerate(chunks):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

# 5. Hàm tính cosine similarity (sử dụng list float)
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

# 6. Hàm truy xuất (retrieve)
def retrieve(query, top_n=3):
    query_embedding = embedding_model.encode(query).tolist()
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# 7. Chatbot
input_query = input('Ask me a question: ')

retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

# Tạo context cho prompt
context = "\n".join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}
'''

# Gọi ollama để sinh phản hồi (phần language model)
stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

print('Chatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
