# Public Deployment Guide

Target architecture:
- Frontend: Vercel
- Backend API: Railway service `backend`
- MCP server: Railway service `mcp`
- Data plane: Supabase (Postgres + pgvector + Storage bucket)

## 1) Provision Supabase

1. Create a Supabase project.
1. In SQL Editor, run [`sql/supabase_schema.sql`](sql/supabase_schema.sql).
1. In SQL Editor, run:

```sql
create extension if not exists vector;
```

1. Create private bucket: `rag-files`.
1. Copy these values:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_URL` (Postgres connection string with write access)

## 2) Deploy MCP on Railway

Service name: `mcp`

Start command:

```bash
python -m uvicorn src.mcp_server.server:app --host 0.0.0.0 --port $PORT
```

Required env:
- `APP_ENV=production`
- `APP_DATA_DIR=data`
- `MCP_REQUIRE_API_KEY=true`
- `MCP_API_KEY=<strong-random-secret>`
- `SUPABASE_URL=<...>`
- `SUPABASE_SERVICE_ROLE_KEY=<...>`
- `SUPABASE_DB_URL=<...>`
- `SUPABASE_STORAGE_BUCKET=rag-files`
- `GEMINI_API_KEY=<...>`

Health checks:
- `GET /health` must return `{"status":"ok"}`
- `GET /ready` must return `{"status":"ready", ...}`

## 3) Deploy Backend on Railway

Service name: `backend`

Start command:

```bash
python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port $PORT
```

Required env:
- `APP_ENV=production`
- `APP_DATA_DIR=data`
- `APP_SESSION_SECRET=<strong-random-secret>`
- `APP_COOKIE_SECURE=true`
- `APP_COOKIE_SAMESITE=none`
- `CORS_ALLOW_ORIGINS=<your-vercel-domain>`
- `MCP_SERVER_URL=<railway-private-url-of-mcp>`
- `MCP_REQUIRE_API_KEY=true`
- `MCP_API_KEY=<same-secret-as-mcp>`
- `SUPABASE_URL=<...>`
- `SUPABASE_SERVICE_ROLE_KEY=<...>`
- `SUPABASE_DB_URL=<...>`
- `SUPABASE_STORAGE_BUCKET=rag-files`
- `GOOGLE_OAUTH_CLIENT_ID=<...>`
- `GOOGLE_OAUTH_CLIENT_SECRET=<...>`
- `APP_OAUTH_REDIRECT_URI=https://<backend-public-domain>/api/auth/google/callback`
- `MAIL_OAUTH_REDIRECT_URI=https://<backend-public-domain>/api/mail/connect/callback`
- `GEMINI_API_KEY=<...>`

Health checks:
- `GET /healthz`
- `GET /readyz`

`/readyz` validates:
- Memory DB availability (Postgres when `SUPABASE_DB_URL` is set)
- Supabase Storage bucket reachability
- MCP connectivity (when `MCP_SERVER_URL` is set)

## 4) Deploy Frontend on Vercel

Project root: `frontend/`

Build:

```bash
npm run build
```

Output directory:
- `dist`

Required env:
- `VITE_API_BASE=https://<backend-public-domain>`

Notes:
- Browser requests use cookies (`credentials: include`), so backend CORS origin must match your Vercel domain exactly.
- For default Vercel/Railway domains, keep:
  - `APP_COOKIE_SECURE=true`
  - `APP_COOKIE_SAMESITE=none`

## 5) OAuth Redirect URIs (Google Console)

Add these backend callback URIs:
- `https://<backend-public-domain>/api/auth/google/callback`
- `https://<backend-public-domain>/api/mail/connect/callback`

## 6) Migrate existing local data to Supabase

One-off migration script:

```bash
python scripts/migrate_local_to_supabase.py \
  --data-dir data \
  --sqlite-db data/memory.db \
  --reset-postgres \
  --verify-retrieval \
  --report-json reports/migration_report.json
```

What the script migrates:
- SQLite tables -> Supabase Postgres:
  - `history`, `conversation_state`, `file_summaries`
  - `chat_sessions`, `chat_messages`, `legacy_chat_migrations`
- Runtime files -> Supabase Storage bucket `rag-files`:
  - `data/pdfs` -> `sessions/global/pdfs/...`
  - `data/resources/...` -> scoped `resources/...`
  - `data/cache/...` -> `cache/...`
- Vector cache -> pgvector:
  - Reads `data/cache/*.pkl` + `*_embeddings.npy`
  - Inserts into `vector_documents` + `vector_embeddings`

## 7) Pre-Go-Live Gates

Backend tests:

```bash
pytest -q tests/unit/test_persistent_memory.py tests/integration/test_app_ask.py tests/integration/test_mail_updates_api.py tests/unit/test_mail_agent_intent_classification.py
```

Frontend build:

```bash
cd frontend
npm install
npm run build
```

Manual browser E2E on deployed domains:
- Google login success.
- Account-scoped chat isolation:
  - Logout hides account sessions.
  - Login account B cannot see account A sessions.
- IELTS 6.5 query cites handbook source.
- Ca 1/Ca 2 query cites timetable source.
- Poll mail works without OAuth refresh errors.

Migration verification:
- Compare row counts from migration report.
- Confirm historical sessions/messages are accessible after cutover.
- Re-run standard retrieval queries and compare top-k source overlap.

## 8) Rollback order (if needed)

1. Frontend rollback (Vercel deployment rollback).
1. Backend rollback (Railway previous release).
1. MCP rollback (Railway previous release).
1. Keep Supabase data read-only during incident triage.
