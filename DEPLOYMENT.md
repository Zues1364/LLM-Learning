# Deployment Notes

This project is still optimized for local/private deployment. Do not expose the
MCP server directly to the public Internet.

## Required runtime state

Mount or persist the runtime data directory. Locally the default is:

```text
data/
```

It contains SQLite databases, uploaded transcripts, local resources, cache files,
and per-session metadata. If the container is recreated without this volume,
chat/resource state is lost.

The backend and MCP server read the runtime root from:

```text
APP_DATA_DIR=data
```

For Docker/container deployment, mount one persistent volume to that path for
both backend and MCP. The sample compose file uses:

```text
APP_DATA_DIR=/app/data
./data:/app/data
```

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
- For public/staging deployments, enable MCP API-key protection:
  `MCP_REQUIRE_API_KEY=true` and set the same strong `MCP_API_KEY` for backend
  and MCP services. The backend forwards it as `X-MCP-API-Key`.
- Set `CORS_ALLOW_ORIGINS` to the real frontend origin only.
- Set `APP_ENV=production`.
- Set `VITE_API_BASE` to the deployed backend HTTPS origin.
- Persist `data/` using a Docker volume, host mount, managed disk, or migrate to
  DB/object storage.
- Rotate OAuth/API keys if `.env` has ever been shared.
- Configure Google OAuth redirect URIs for the deployed backend:
  `/api/auth/google/callback` and `/api/mail/connect/callback`.

## Health endpoints

- Backend liveness: `GET /healthz`
- Backend readiness: `GET /readyz`

Readiness checks SQLite access and required runtime directories.
