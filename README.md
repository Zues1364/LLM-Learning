# LLM Learning - RAG Academic Advisor

He thong nay xay dung chatbot hoc vu cho sinh vien UET theo kieu RAG + deterministic tools. Bot co the doc bang diem PDF, chuong trinh dao tao, so tay hoc vu, thoi khoa bieu va tra loi cac cau hoi nhu mon con thieu, GPA du kien, lich hoc, dieu kien ngoai ngu, lich mo lop va tra cuu theo giao vien.

## 1. Cau truc repository

Yeu cau toi thieu cua repo da duoc giu o root:

```text
LLM Learning/
|- src/                # Ma nguon backend, MCP server, agent va utility
|- references/         # PDF tham khao va tai lieu demo
`- README.md           # Huong dan cai dat, chay va demo
```

Cac thu muc phu dang dung trong project:

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
|- tests/              # Unit + integration tests chinh
|- test/               # Script debug / deep checks
|- scripts/            # Utility scripts
|- sql/                # Schema / migration SQL
|- data/               # Runtime data local: cache, memory, uploaded files
|- references/         # Tai lieu PDF de tham khao va demo
|- docs/, doc/         # Tai lieu mo ta va file noi bo
`- README.md
```

## 2. Thanh phan chinh

- `src/app.py`: FastAPI backend, route `/ask`, upload file, session, resource APIs.
- `src/mcp_server/server.py`: MCP server, deterministic tools, advisor pipeline, schedule tools, transcript analysis.
- `src/agents.py`: planner / answer generation.
- `frontend/src/App.jsx`: giao dien chat, session, file upload, chuong trinh dao tao.
- `src/resource_loader.py`: tai local resources, sync scope local/user/session.
- `src/utils.py`: PDF extraction, OCR, chunking, embeddings.

## 3. Tai lieu trong `/references`

Thu muc `references/` chua bo PDF tham khao duoc dung de demo va kiem thu:

- `SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf`
- `PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf`
- `QuyDinh_KhoaLuanTotNghiep_BoMonCNPM_2026-01.pdf`
- `2504.11094v2.pdf`

Luu y:

- `references/` la bo tai lieu tham khao de nop repo.
- `data/` la du lieu runtime local. Code dang doc cache, memory va uploaded file tu day.
- Neu muon dung nhanh bo PDF tham khao de demo local, ban co the upload qua UI hoac copy thu cong vao `data/resources/pdfs/`.

## 4. Yeu cau moi truong

### Bat buoc

- Python `3.11` khuyen nghi, toi thieu `3.10`
- Node.js `18+`
- npm `9+`
- Tesseract OCR co language data `vie` va `eng`

### Kiem tra nhanh

```powershell
python --version
node --version
npm --version
tesseract --version
```

## 5. Cai dat moi truong

### 5.1. Clone repo

```powershell
git clone https://github.com/Zues1364/LLM-Learning.git
cd "LLM Learning"
```

### 5.2. Tao virtual environment va cai Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu can chay them mot so script phu, co the cai editable package:

```powershell
python -m pip install -e .
```

### 5.3. Cai frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## 6. Cau hinh `.env`

Copy file mau:

```powershell
Copy-Item .env.example .env
```

Toi thieu can dien:

```env
APP_ENV=development
APP_DATA_DIR=data
GEMINI_API_KEY=your_gemini_key
MCP_SERVER_URL=http://127.0.0.1:8000
VITE_API_BASE=http://127.0.0.1:9000
```

Neu muon chay day du storage / auth / Postgres memory:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_DB_URL=
SUPABASE_STORAGE_BUCKET=rag-files
APP_SESSION_SECRET=change-me-before-production
GOOGLE_OAUTH_CLIENT_ID=
GOOGLE_OAUTH_CLIENT_SECRET=
```

Danh sach bien day du nam trong file `.env.example`.

## 7. Chay code local

Mo 3 terminal trong root repo.

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

Mac dinh:

- Frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- Backend docs: [http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)
- MCP server: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 8. Demo end-to-end

### Cach 1 - Demo bang UI

1. Mo frontend tai [http://127.0.0.1:5173](http://127.0.0.1:5173)
2. Vao `Quan ly Tai nguyen`
3. Upload cac file trong `references/` hoac trong bo resource rieng cua ban
4. Quay lai man hinh chat
5. Chon chuong trinh dao tao
6. Upload 1 hoac nhieu file bang diem PDF
7. Tick cac file can dung cho session hien tai
8. Thu cac cau hoi:
   - `toi con thieu nhung mon nao theo chuong trinh dao tao`
   - `toi can ban lap giup toi lich hoc dua tren cac mon con thieu cua toi`
   - `voi 6.5 ielts toi co du dieu kien tieng anh de ra truong khong`
   - `mon tri tue nhan tao ki nay lich hoc nhu nao`
   - `thay Tran Hoang Viet ki nay day nhung mon gi`

### Cach 2 - Bootstrap local resources nhanh

Neu muon san bo PDF demo trong local runtime:

```powershell
New-Item -ItemType Directory -Force data\\resources\\pdfs | Out-Null
Copy-Item references\\SO_TAY_HOC_VU_KY_I_NAM_2023-2024.pdf data\\resources\\pdfs\\
Copy-Item references\\PHU_LUC_THOI_KHOA_BIEU_HKII_2025-2026_DU_LIEU_CAP_NHAT_DEN_22012026_.xlsx_-_Sheet1.pdf data\\resources\\pdfs\\
Copy-Item references\\QuyDinh_KhoaLuanTotNghiep_BoMonCNPM_2026-01.pdf data\\resources\\pdfs\\
```

Sau do bam refresh resource trong UI hoac restart backend / MCP.

## 9. API chinh

- `POST /ask`: route hoi dap chinh
- `POST /upload_pdf`, `POST /upload_pdfs`: upload transcript
- `GET /files`: danh sach file transcript cua session/user
- `DELETE /session`: xoa lich su session hien tai
- `GET /history`: lich su hoi dap
- `GET /api/programs`: danh sach chuong trinh dao tao
- `GET /api/resources`: danh sach resource local
- `POST /api/resources/pdf|pdfs|html|htmls|url`: them resource
- `DELETE /api/resources/{resource_id}`: xoa resource

## 10. Chay test

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

## 11. Loi thuong gap

### `GEMINI_API_KEY missing`

- Kiem tra `.env`
- Restart backend va MCP sau khi sua env

### Backend khong goi duoc MCP

- Kiem tra `MCP_SERVER_URL`
- Kiem tra terminal MCP co dang chay cong `8000`

### OCR / PDF extraction loi

- Kiem tra `tesseract --version`
- Kiem tra da cai language `vie`, `eng`

### Upload file thanh cong nhung advisor tra loi thieu du lieu

- Kiem tra da tick dung transcript trong session hien tai
- Kiem tra da chon dung chuong trinh dao tao
- Neu doi file transcript, nen tao session moi de tranh state cu

## 12. Ghi chu van hanh

- `data/cache/`: chunk cache, embedding cache
- `data/session_cache/`: meta session local
- `data/resources/`: resource theo scope global / user / session
- `data/memory.db`, `data/structured_schedule.db`: local runtime state

Neu khong muon commit du lieu runtime, giu nguyen `data/` trong `.gitignore`. Thu muc `references/` moi la noi de dua PDF tham khao vao repo nop bai.
