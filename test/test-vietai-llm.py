from transformers import pipeline

def test_llm_generator():
    model_name = "VietAI/gpt-neo-1.3B-vietnamese-news"
    max_length = 1024
    max_new_tokens = 256
    temperature = 0.7

    # Khởi tạo pipeline sinh văn bản
    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        truncation=True,
        temperature=temperature,
        do_sample=True
    )

    # Đặt prompt để sinh văn bản
    prompt = "biết đọc tài liệu hay không"
    result = llm_pipeline(prompt)

    # In kết quả sinh văn bản
    print("Kết quả sinh văn bản:")
    print(result[0]['generated_text'])

if __name__ == "__main__":
    test_llm_generator()
