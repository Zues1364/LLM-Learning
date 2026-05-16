# LLM Learning - RAG Academic Advisor

Hệ thống này xây dựng chatbot học vụ cho sinh viên UET theo kiểu RAG kết hợp deterministic tools. Bot có thể đọc bảng điểm PDF, chương trình đào tạo, sổ tay học vụ, thời khóa biểu và trả lời các câu hỏi như môn còn thiếu, GPA dự kiến, lịch học, điều kiện ngoại ngữ, lịch mở lớp và tra cứu theo giáo viên.

## 1. Cấu trúc repository

Yêu cầu tối thiểu của repo đã được giữ ở root:

```text
LLM Learning/
|- src/                # Mã nguồn backend, MCP server, agent và utility
|- references/         # PDF tham khảo và tài liệu demo
`- README.md           # Hướng dẫn cài đặt, chạy và demo
```

Các thư mục phụ đang dùng trong project:

```text
LLM Learning/
|- src/
|  |- app.py
|  |- agents.py
|  |- resource_loader.py
|  |- utils.py
|  |- mcp_client/
|  `- mcp_server/
|- frontend/           # Vite + React chat UI
|- tests/              # Unit + integration tests chính
|- test/               # Script debug / deep checks
|- scripts/            # Utility scripts
|- sql/                # Schema / migration SQL
|- data/               # Runtime data local: cache, memory, uploaded files
|- references/         # Tài liệu PDF để tham khảo và demo
|- docs/, doc/         # Tài liệu mô tả và file nội bộ
`- README.md
```

## 2. Thành phần chính

- `src/app.py`: FastAPI backend, route `/ask`, upload file, session, resource APIs.
- `src/mcp_server/server.py`: MCP server, deterministic tools, advisor pipeline, schedule tools, transcript analysis.
- `src/agents.py`: planner / answer generation.
- `frontend/src/App.jsx`: giao diện chat, session, file upload, chương trình đào tạo.
- `src/resource_loader.py`: tải local resources, sync scope local/user/session.
- `src/utils.py`: PDF extraction, OCR, chunking, embeddings.

## 3. Tài liệu trong `/references`

Thư mục `references/` chứa bộ PDF tham khảo được dùng để demo và kiểm thử:

- `SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf`
- `PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf`
- `QuyDinh_KhoaLuanTotNghiep_BoMonCNPM_2026-01.pdf`
- `2504.11094v2.pdf`

Lưu ý:

- `references/` là bộ tài liệu tham khảo để nộp repo.
- `data/` là dữ liệu runtime local. Code đang đọc cache, memory và uploaded file từ đây.
- Nếu muốn dùng nhanh bộ PDF tham khảo để demo local, bạn có thể upload qua UI hoặc copy thủ công vào `data/resources/pdfs/`.

## 4. Yêu cầu môi trường

### Bắt buộc

- Python `3.11` khuyến nghị, tối thiểu `3.10`
- Node.js `18+`
- npm `9+`
- Tesseract OCR có language data `vie` và `eng`

### Kiểm tra nhanh

```powershell
python --version
node --version
npm --version
tesseract --version
```

## 5. Cài đặt môi trường

### 5.1. Clone repo

```powershell
git clone https://github.com/Giang130604/LLM-Learning.git
cd "LLM Learning"
```

### 5.2. Tạo virtual environment và cài Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu cần chạy thêm một số script phụ, có thể cài editable package:

```powershell
python -m pip install -e .
```

### 5.3. Cài frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## 6. Cấu hình `.env`

Copy file mẫu:

```powershell
Copy-Item .env.example .env
```

Tối thiểu cần điền:

```env
APP_ENV=development
APP_DATA_DIR=data
GEMINI_API_KEY=your_gemini_key
MCP_SERVER_URL=http://127.0.0.1:8000
VITE_API_BASE=http://127.0.0.1:9000
```

Nếu muốn chạy đầy đủ storage / auth / Postgres memory:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_DB_URL=
SUPABASE_STORAGE_BUCKET=rag-files
APP_SESSION_SECRET=change-me-before-production
GOOGLE_OAUTH_CLIENT_ID=
GOOGLE_OAUTH_CLIENT_SECRET=
```

Danh sách biến đầy đủ nằm trong file `.env.example`.

## 7. Chạy code local

Mở 3 terminal trong root repo.

### Terminal 1 - MCP server

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.mcp_server.server:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Backend API

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app:app --host 0.0.0.0 --port 9000 --reload
```

### Terminal 3 - Frontend

```powershell
cd frontend
npm run dev
```

Mặc định:

- Frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- Backend docs: [http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)
- MCP server: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 8. Demo end-to-end

### Cách 1 - Demo bằng UI

1. Mở frontend tại [http://127.0.0.1:5173](http://127.0.0.1:5173)
2. Vào `Quản lý Tài nguyên`
3. Upload các file trong `references/` hoặc trong bộ resource riêng của bạn
4. Quay lại màn hình chat
5. Chọn chương trình đào tạo
6. Upload 1 hoặc nhiều file bảng điểm PDF
7. Tick các file cần dùng cho session hiện tại
8. Thử các câu hỏi:
   - `tôi còn thiếu những môn nào theo chương trình đào tạo`
   - `tôi cần bạn lập giúp tôi lịch học dựa trên các môn còn thiếu của tôi`
   - `với 6.5 ielts tôi có đủ điều kiện tiếng anh để ra trường không`
   - `môn trí tuệ nhân tạo kì này lịch học như nào`
   - `thầy Trần Hoàng Việt kì này dạy những môn gì`

### Cách 2 - Bootstrap local resources nhanh

Nếu muốn sẵn bộ PDF demo trong local runtime:

```powershell
New-Item -ItemType Directory -Force data\\resources\\pdfs | Out-Null
Copy-Item references\\SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf data\\resources\\pdfs\\
Copy-Item references\\PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf data\\resources\\pdfs\\
Copy-Item references\\QuyDinh_KhoaLuanTotNghiep_BoMonCNPM_2026-01.pdf data\\resources\\pdfs\\
```

Sau đó bấm refresh resource trong UI hoặc restart backend / MCP.

## 9. API chính

- `POST /ask`: route hỏi đáp chính
- `POST /upload_pdf`, `POST /upload_pdfs`: upload transcript
- `GET /files`: danh sách file transcript của session/user
- `DELETE /session`: xóa lịch sử session hiện tại
- `GET /history`: lịch sử hỏi đáp
- `GET /api/programs`: danh sách chương trình đào tạo
- `GET /api/resources`: danh sách resource local
- `POST /api/resources/pdf|pdfs|html|htmls|url`: thêm resource
- `DELETE /api/resources/{resource_id}`: xóa resource

## 10. Chạy test

### Unit tests

```powershell
python -m pytest tests/unit -q
```

### Integration tests

```powershell
python -m pytest tests/integration -q
```

### Frontend E2E

```powershell
cd frontend
npx playwright test
```

## 11. Lỗi thường gặp

### `GEMINI_API_KEY missing`

- Kiểm tra `.env`
- Restart backend và MCP sau khi sửa env

### Backend không gọi được MCP

- Kiểm tra `MCP_SERVER_URL`
- Kiểm tra terminal MCP có đang chạy cổng `8000`

### OCR / PDF extraction lỗi

- Kiểm tra `tesseract --version`
- Kiểm tra đã cài language `vie`, `eng`

### Upload file thành công nhưng advisor trả lời thiếu dữ liệu

- Kiểm tra đã tick đúng transcript trong session hiện tại
- Kiểm tra đã chọn đúng chương trình đào tạo
- Nếu đổi file transcript, nên tạo session mới để tránh state cũ

## 12. Ghi chú vận hành

- `data/cache/`: chunk cache, embedding cache
- `data/session_cache/`: meta session local
- `data/resources/`: resource theo scope global / user / session
- `data/memory.db`, `data/structured_schedule.db`: local runtime state

Nếu không muốn commit dữ liệu runtime, giữ nguyên `data/` trong `.gitignore`. Thư mục `references/` mới là nơi để đưa PDF tham khảo vào repo nộp bài.
