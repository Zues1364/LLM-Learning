import sys
import json
import numpy as np
import urllib.parse
import requests
from sentence_transformers import SentenceTransformer
import ollama
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ========= 1) Data & Retrieval Setup =========
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open('cat-facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    dataset_lines = f.readlines()

# Create chunks
chunks = []
window_size = 3
for line in dataset_lines:
    sentences = sent_tokenize(line.strip())
    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks.append(chunk)
print(f'Created {len(chunks)} chunks')

# Vector database
VECTOR_DB = []


def add_to_vector_db(chunk):
    emb = embedding_model.encode(chunk).tolist()
    VECTOR_DB.append((chunk, emb))


for chunk in chunks:
    add_to_vector_db(chunk)


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def retrieve(query, top_n=3):
    query_emb = embedding_model.encode(query).tolist()
    similarities = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in VECTOR_DB]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


# ========= 2) Generation =========
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


def generate_answer(context, question):
    prompt = f"""You are a helpful chatbot. Use ONLY the following context to answer the question:
{context}

Question: {question}

Answer:"""
    response = ""
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question},
            ],
            stream=True
        )
        for part in stream:
            response += part['message']['content']
    except Exception as e:
        print(f"Error in generation: {e}")
        response = "Unable to generate answer."
    return response.strip()


# ========= 3) Evaluator with T5 Large =========
eval_model = T5ForConditionalGeneration.from_pretrained("t5-large")
eval_tokenizer = T5Tokenizer.from_pretrained("t5-large")


def evaluate_answer(context, question, answer):
    eval_prompt = f"""Your task is to evaluate if the answer correctly addresses the question based on the context. Output ONLY a JSON object with "label" ("CORRECT", "INCORRECT", "AMBIGUOUS") and "confidence" (float 0-1). No extra text.

    Example 1:
    Context: Cats are mammals.
    Question: Are cats mammals?
    Answer: Yes, cats are mammals.
    Output: {{"label": "CORRECT", "confidence": 0.95}}

    Example 2:
    Context: Cats have nine lives.
    Question: Do cats have ten lives?
    Answer: No, cats do not have ten lives.
    Output: {{"label": "INCORRECT", "confidence": 0.90}}

    Example 3:
    Context: Cats can purr.
    Question: Can cats fly?
    Answer: No definitive answer based on context.
    Output: {{"label": "AMBIGUOUS", "confidence": 0.60}}

    Context: {context}
    Question: {question}
    Answer: {answer}
    Output ONLY the JSON object."""
    input_ids = eval_tokenizer.encode(eval_prompt, return_tensors="pt", truncation=True)
    outputs = eval_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True, do_sample=False)
    eval_text = eval_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Evaluator raw output:", eval_text)

    label, confidence = "AMBIGUOUS", 0.0
    try:
        parsed = json.loads(eval_text)
        label = parsed.get("label", "AMBIGUOUS").upper()
        confidence = float(parsed.get("confidence", 0.0))
        print(f"Parsed JSON: {parsed}")
    except Exception as e:
        print(f"Error parsing evaluator output: {e}, Raw text: {eval_text}")
        # Fallback: Cosine similarity
        context_emb = embedding_model.encode(context)
        answer_emb = embedding_model.encode(answer)
        similarity = cosine_similarity(context_emb, answer_emb)
        if similarity > 0.85:
            label, confidence = "CORRECT", min(0.95, similarity)
        elif similarity < 0.5:
            label, confidence = "INCORRECT", 0.9
        else:
            label, confidence = "AMBIGUOUS", max(0.6, similarity * 0.8)
        print(f"Fallback evaluation: Label={label}, Confidence={confidence:.2f}")
    if label not in ["CORRECT", "INCORRECT", "AMBIGUOUS"]:
        label = "AMBIGUOUS"
    return label, confidence


# ========= 4) Correction Strategies with T5 Large =========
def refine_query(question, context, answer):
    refine_prompt = f"""Rewrite the question to make it clearer based on the context and answer. Output ONLY the rewritten question as a single sentence. Do not output True/False or unrelated text.

    Example 1:
    Original question: Are cats big?
    Context: Cats are typically small domesticated animals.
    Answer: No, cats are not big.
    Output: Are cats typically small domesticated animals?

    Example 2:
    Original question: Do cats fly?
    Context: Cats can purr.
    Answer: No definitive answer based on context.
    Output: What abilities do cats have besides purring?

    Original question: {question}
    Context: {context}
    Answer: {answer}
    Output ONLY the rewritten question."""
    input_ids = eval_tokenizer.encode(refine_prompt, return_tensors="pt", truncation=True)
    outputs = eval_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True, do_sample=False)
    refined_question = eval_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Refined question: {refined_question}")
    return refined_question if refined_question and "True" not in refined_question and "False" not in refined_question else question


API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"  # Store securely in production!


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


def expand_knowledge(question, context):
    extended_docs = retrieve(question, top_n=5)
    extended_context = "\n".join([f"- {doc[0]}" for doc in extended_docs])
    web_results = web_search(question)
    web_context = "\n".join([f"- {snippet}" for snippet in web_results])
    return f"{context}\n{extended_context}\n{web_context}"


# ========= 5) Corrective RAG (CRAG) Loop =========
def crag_inference(user_question, max_iterations=3):
    current_question = user_question
    original_question = user_question
    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")

        # Retrieve context
        retrieved_docs = retrieve(current_question)
        context = "\n".join([f"- {doc[0]}" for doc in retrieved_docs])
        print(f"Context:\n{context}")

        # Generate answer
        answer = generate_answer(context, current_question)
        print(f"Generated Answer:\n{answer}")

        # Evaluate answer
        label, confidence = evaluate_answer(context, current_question, answer)
        print(f"Evaluator: Label={label}, Confidence={confidence:.2f}")

        # Decision based on evaluation
        if label == "CORRECT" and confidence >= 0.9:
            print("Answer is CORRECT with high confidence. Stopping.")
            return answer

        elif label == "AMBIGUOUS" or (label == "CORRECT" and confidence < 0.9):
            print("AMBIGUOUS or low confidence: Refining query...")
            current_question = refine_query(current_question, context, answer)
            # Check if refined question deviates too far
            question_emb = embedding_model.encode(current_question)
            orig_emb = embedding_model.encode(original_question)
            if cosine_similarity(question_emb, orig_emb) < 0.5:
                print("Refined question deviated too far. Reverting to original.")
                current_question = original_question

        elif label == "INCORRECT" or confidence < 0.5:
            print("INCORRECT or very low confidence: Expanding knowledge...")
            new_context = expand_knowledge(current_question, context)
            print(f"Expanded Context:\n{new_context}")
            new_answer = generate_answer(new_context, current_question)
            print(f"New Answer:\n{new_answer}")
            new_label, new_conf = evaluate_answer(new_context, current_question, new_answer)
            print(f"New Evaluation: Label={new_label}, Confidence={new_conf:.2f}")
            if new_label == "CORRECT" and new_conf >= 0.9:
                print("New answer is CORRECT with high confidence. Stopping.")
                return new_answer
            else:
                current_question = refine_query(current_question, new_context, new_answer)
                question_emb = embedding_model.encode(current_question)
                orig_emb = embedding_model.encode(original_question)
                if cosine_similarity(question_emb, orig_emb) < 0.5:
                    print("Refined question deviated too far. Reverting to original.")
                    current_question = original_question

    print("Max iterations reached. Returning last answer.")
    return answer


# ========= Main Execution =========
if __name__ == "__main__":
    user_question = input("Ask me a question: ")
    final_answer = crag_inference(user_question)
    print("\n=== Corrective RAG with T5 Large Finished ===")
    print("Final Answer:")
    print(final_answer)