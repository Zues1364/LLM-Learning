# LLM Learning - RAG Academic Advisor

## 1. Mục tiêu dự án
Hệ thống tư vấn học vụ dùng mô hình RAG (Retrieval-Augmented Generation) cho sinh viên UET, tập trung vào:
- Tra cứu bảng điểm PDF, CTĐT HTML, sổ tay học vụ, thời khóa biểu.
- Tính thiếu tín chỉ, môn còn thiếu, GPA projection.
- Kiểm tra mở lớp theo kỳ và trả lời truy vấn học vụ theo ngữ cảnh phiên.

## 2. Kiến trúc hiện tại
Dự án chạy theo mô hình 3 dịch vụ:
- `MCP Server` (`src/mcp_server/server.py`, cổng `8000`): lớp deterministic tools (retrieve, curriculum, schedule, advisor pipeline, memory).
- `Backend API` (`src/app.py`, cổng `9000`): lớp orchestrator `/ask`, session, upload PDF, resource management, gọi MCP qua HTTP.
- `Frontend` (`frontend`, Vite + React, cổng `5173`): giao diện chat, quản lý tài nguyên, chọn CTĐT, upload transcript.

Luồng chính:
1. Frontend gọi `POST /ask` ở Backend.
2. Backend gọi planner/answer agent và gọi tool qua MCP client.
3. MCP Server xử lý retrieval + logic học vụ, trả context.
4. Backend trả câu trả lời cuối cùng cho Frontend.

Tài liệu kiến trúc chi tiết: `docs/ARCHITECTURE.md`.

## 3. Cấu trúc thư mục
```text
LLM Learning/
|- src/
|  |- app.py                     # FastAPI orchestrator
|  |- agents.py                  # planner + answer/advisor agent
|  |- utils.py                   # OCR, chunking, embeddings, FAISS
|  |- resource_loader.py         # ingest local/global resources
|  |- persistent_memory.py       # SQLite memory theo session
|  |- mcp_client/client.py       # HTTP client gọi MCP tools
|  |- mcp_server/server.py       # MCP server + tool registry
|- frontend/
|  |- src/App.jsx                # giao diện chính
|  |- package.json
|- data/
|  |- pdfs/                      # transcript do user upload
|  |- resources/pdfs|html/       # local resources
|  |- cache/                     # chunk + embedding cache
|  |- memory.db                  # hội thoại/persistent memory
|- tests/
|  |- unit/
|  |- integration/
|- README.md
```

## 4. Yêu cầu môi trường
- Python `3.10+` (khuyến nghị `3.11`).
- Node.js `18+` và npm.
- Tesseract OCR (bắt buộc để OCR PDF bảng biểu):
  - Cài `tesseract` và language data `vie`, `eng`.
  - Kiểm tra: `tesseract --version`.

## 5. Cài đặt
### 5.1. Python dependencies
Từ thư mục gốc `LLM Learning`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install \
  fastapi uvicorn pydantic requests beautifulsoup4 numpy pdfplumber \
  sentence-transformers faiss-cpu google-generativeai agno \
  langchain-core langchain-text-splitters img2table pytesseract
```

### 5.2. Frontend dependencies
```powershell
cd frontend
npm install
cd ..
```

## 6. Cấu hình `.env`
Tạo file `.env` tại root `LLM Learning`:
```env
GEMINI_API_KEY=your_gemini_key_here
MCP_SERVER_URL=http://127.0.0.1:8000
LOG_CHUNK_LOADING=false
```

Ghi chú:
- `GEMINI_API_KEY` là bắt buộc cho agent/advisor và tóm tắt.
- `MCP_SERVER_URL` dùng cho Backend gọi MCP server.
- Hệ thống có cơ chế normalize `GOOGLE_API_KEY -> GEMINI_API_KEY` trong `src/env_loader.py`, nhưng nên khai báo trực tiếp `GEMINI_API_KEY`.

## 7. Chạy hệ thống
Mở 3 terminal độc lập tại root `LLM Learning`.

### Terminal 1 - MCP Server
```powershell
python -m uvicorn src.mcp_server.server:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Backend API
```powershell
python -m uvicorn src.app:app --host 0.0.0.0 --port 9000 --reload
```

### Terminal 3 - Frontend
```powershell
cd frontend
npm run dev
```

Mặc định truy cập UI tại `http://localhost:5173`.

## 8. Demo end-to-end
### Bước 1: Nạp tài nguyên học vụ
Trong UI, mở panel `Quản lý Tài nguyên` và upload:
- HTML CTĐT (để nhận diện chương trình đào tạo).
- PDF thời khóa biểu.
- PDF sổ tay học vụ/quy chế.

### Bước 2: Chọn chương trình đào tạo
Trong khung `Chương trình đào tạo hiện tại`:
- Chọn đúng chương trình (ví dụ `Khoa học máy tính (QH-2022-2024)`).
- Bấm `Xác nhận`.

### Bước 3: Upload bảng điểm
Upload 1 hoặc nhiều file PDF transcript ở khung `File đã tải lên`, tick file cần dùng cho phiên.

### Bước 4: Đặt câu hỏi demo
Ví dụ:
- `tôi còn thiếu những môn nào theo chương trình đào tạo`
- `môn INT3412E kỳ này có mở lớp không`
- `ca 1 bắt đầu từ mấy giờ`
- `với 6.5 ielts tôi có đủ điều kiện tiếng anh để ra trường không`

### Bước 5: Kiểm tra lịch sử phiên
Sidebar sẽ hiển thị lịch sử query theo từng session; backend lưu trong `data/memory.db`.

## 9. API chính
- `POST /upload_pdf`, `POST /upload_pdfs`: upload transcript.
- `GET /files`: danh sách transcript đã upload.
- `POST /ask`: hỏi đáp chính (orchestrator).
- `GET /history`: lịch sử hội thoại theo session.
- `DELETE /session`: xóa lịch sử session.
- `GET /api/resources`: danh sách local resources.
- `POST /api/resources/pdf|pdfs|html|htmls|url`: thêm resource.
- `DELETE /api/resources/{resource_id}`: xóa resource.
- `GET /api/programs`: danh sách chương trình đào tạo đã parse từ HTML.

## 10. Chạy test
```powershell
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

Nếu cần debug sâu luồng advisor/schedule, xem log runtime tại terminal của `src/mcp_server/server.py`.

## 11. Lỗi thường gặp
- `GEMINI_API_KEY missing`: kiểm tra `.env` và restart service.
- Backend không gọi được MCP: kiểm tra `MCP_SERVER_URL` và MCP server có đang chạy cổng `8000`.
- OCR không hoạt động: kiểm tra cài `tesseract` + language `vie`.
- Trả lời thiếu ngữ cảnh: đảm bảo đã tick đúng transcript file trong phiên hiện tại.

## 12. Ghi chú vận hành
- Dữ liệu runtime/cache nằm trong `data/`.
- Cache chunk/embedding được tái sử dụng để tăng tốc ingest.
- Session metadata được lưu ở `data/session_cache/` để giữ file/program theo phiên.
