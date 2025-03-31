# Tổng quan dự án

Dự án này là một bộ sưu tập các tệp mã Python và dữ liệu, tập trung vào việc xây dựng hệ thống **RAG (Retrieval Augmented Generation)** để trả lời câu hỏi về mèo, sự kiện chung, cùng với một thành phần phân loại văn bản liên quan đến tin tức kinh tế Mỹ. Các hệ thống RAG sử dụng kỹ thuật nhúng (embedding) và mô hình ngôn ngữ để tạo câu trả lời, với mức độ phức tạp từ chatbot đơn giản đến hệ thống đa tác nhân có cơ chế sửa lỗi.

---

## Các điểm nổi bật

- **Mục đích**: Xây dựng chatbot thông minh và phân loại văn bản dựa trên dữ liệu về mèo, sự kiện chung, và tin tức kinh tế Mỹ.
- **Tệp mã chính**: Bao gồm nhiều chatbot với độ phức tạp khác nhau (`arag.py`, `rag.py`, `multi-agent-rag.py`, `crag.py`, `testapi.py`).
- **Dữ liệu**: Gồm `cat-facts.txt`, `facts.txt`, và `us-economic news-relevance.csv`.

---

## Thành phần chính

### Tệp mã

| Tệp mã             | Mục đích chính                                          | Đặc điểm nổi bật                              |
|--------------------|--------------------------------------------------------|-----------------------------------------------|
| `arag.py`          | Trợ lý AI phức tạp, định tuyến câu hỏi                  | Bộ nhớ, tự đánh giá, hỗ trợ web và dữ liệu nội bộ |
| `rag.py`           | Chatbot đơn giản về mèo                                | Dùng nhúng và mô hình ngôn ngữ cho `cat-facts.txt` |
| `multi-agent-rag.py` | Hệ thống RAG đa tác nhân                             | Kết hợp dữ liệu nội bộ và tìm kiếm web         |
| `crag.py`          | RAG nâng cao với cơ chế sửa lỗi cho mèo                | Đánh giá độ chính xác, mở rộng qua web         |
| `testapi.py`       | Công cụ kiểm tra tìm kiếm web qua API Serper           | Trả về đoạn trích từ kết quả tìm kiếm          |

### Tệp dữ liệu

| Tệp dữ liệu                   | Nội dung chính                                   | Vai trò                                      |
|-------------------------------|-------------------------------------------------|---------------------------------------------|
| `cat-facts.txt`              | Sự kiện về mèo (hành vi, sinh học, lịch sử)     | Nguồn dữ liệu cho chatbot về mèo            |
| `facts.txt`                  | Sự kiện chung (khoa học, tự nhiên, lịch sử)     | Dữ liệu cho câu hỏi chung                   |
| `us-economic news-relevance.csv` | Nhãn nhị phân (0/1) cho tin tức kinh tế Mỹ    | Dùng trong phân loại văn bản                |

### Sổ tay Jupyter
- **`text-classification-nlp.ipynb`**: Minh họa phân loại văn bản bằng SVM, Random Forest, BERT trên dữ liệu tin tức kinh tế Mỹ.

---

## Cài đặt và phụ thuộc

Để chạy dự án, bạn cần cài đặt các thư viện sau:

```bash
pip install sentence_transformers nltk ollama requests urllib.parse huggingface_hub pandas numpy scikit-learn transformers
