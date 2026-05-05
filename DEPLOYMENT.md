# Deployment Notes

This project is still optimized for local/private deployment. Do not expose the
MCP server directly to the public Internet.

## Required runtime state

Mount or persist this directory:

```text
data/
```

It contains SQLite databases, uploaded transcripts, local resources, cache files,
and per-session metadata. If the container is recreated without this volume,
chat/resource state is lost.

## Local container run

1. Copy env template:

```powershell
Copy-Item .env.example .env
```

2. Fill secrets in `.env`:

```text
GEMINI_API_KEY=
GOOGLE_OAUTH_CLIENT_ID=
GOOGLE_OAUTH_CLIENT_SECRET=
APP_SESSION_SECRET=
```

3. Start backend and MCP:

```powershell
docker compose up --build
```

4. Start frontend locally:

```powershell
cd frontend
$env:VITE_API_BASE="http://127.0.0.1:9000"
npm install
npm run dev
```

## Production checklist

- Put backend behind HTTPS and set `APP_COOKIE_SECURE=true`.
- Set a strong random `APP_SESSION_SECRET`.
- Keep `MCP_SERVER_URL` on a private network, for example `http://mcp:8000`.
- Bind MCP to private/internal network only. The sample compose binds to
  `127.0.0.1` for local safety.
- Set `CORS_ALLOW_ORIGINS` to the real frontend origin only.
- Persist `data/` using a Docker volume, host mount, managed disk, or migrate to
  DB/object storage.
- Rotate OAuth/API keys if `.env` has ever been shared.
- Configure Google OAuth redirect URIs for the deployed backend:
  `/api/auth/google/callback` and `/api/mail/connect/callback`.

## Health endpoints

- Backend liveness: `GET /healthz`
- Backend readiness: `GET /readyz`

Readiness checks SQLite access and required runtime directories.
