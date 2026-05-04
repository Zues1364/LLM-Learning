import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


class _DummyMailService:
    def __init__(self):
        self._whitelist = {}

    def get_status(self, session_id):
        owner_type = "user" if session_id == "auth-session" else "session"
        return {
            "session_id": session_id,
            "connected": True,
            "candidate_counts": {"pending": 1},
            "owner_type": owner_type,
            "mode": owner_type,
        }

    def begin_oauth(self, session_id, redirect_uri):
        return {"auth_url": "https://accounts.google.com/o/oauth2/v2/auth?state=x", "state": "x"}

    def complete_oauth(self, session_id, state, code, redirect_uri):
        return {"session_id": session_id, "email": "student@uet.edu.vn", "connected": True}

    def complete_oauth_from_state(self, state, code, redirect_uri):
        return {"email": "student@uet.edu.vn", "connected": True}

    def disconnect(self, session_id):
        return {"session_id": session_id, "connected": False}

    def get_whitelist(self, session_id):
        return self._whitelist.get(session_id, [])

    def set_whitelist(self, session_id, senders):
        self._whitelist[session_id] = list(senders or [])
        return self._whitelist[session_id]

    def poll_session(self, session_id):
        return {"session_id": session_id, "processed_count": 2}

    def list_candidates(self, session_id, status=None):
        return [{"id": "c1", "status": status or "pending", "session_id": session_id}]

    def apply_candidate(self, session_id, candidate_id):
        return {"id": candidate_id, "status": "applied", "session_id": session_id}

    def reject_candidate(self, session_id, candidate_id, reason=None):
        return {"id": candidate_id, "status": "rejected", "reason": reason, "session_id": session_id}

    def list_connected_sessions(self):
        return []

    def cleanup_old_candidates(self):
        return 0


def test_mail_status_and_whitelist_endpoints(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "mail_agent_service", _DummyMailService())
    client = TestClient(app_mod.app)

    status = client.get("/api/mail/status", params={"session_id": "s-1"})
    assert status.status_code == 200
    assert status.json()["connected"] is True

    set_wl = client.post(
        "/api/mail/whitelist",
        json={"session_id": "s-1", "senders": ["daotao@uet.edu.vn"]},
    )
    assert set_wl.status_code == 200
    assert set_wl.json()["senders"] == ["daotao@uet.edu.vn"]

    get_wl = client.get("/api/mail/whitelist", params={"session_id": "s-1"})
    assert get_wl.status_code == 200
    assert get_wl.json()["senders"] == ["daotao@uet.edu.vn"]


def test_mail_apply_candidate_triggers_session_scan(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "mail_agent_service", _DummyMailService())

    invoked = {}

    class _DummyMCPClient:
        def invoke(self, tool, args):
            invoked["tool"] = tool
            invoked["args"] = args
            return {"result": "ok"}

    monkeypatch.setattr(app_mod, "mcp_client", _DummyMCPClient())
    client = TestClient(app_mod.app)

    resp = client.post(
        "/api/mail/candidates/c42/apply",
        json={"session_id": "mail-session-01"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate"]["status"] == "applied"
    assert invoked["tool"] == "scan_resources"
    assert invoked["args"]["session_id"] == "mail-session-01"


def test_mail_status_keeps_additive_owner_fields(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))
    monkeypatch.setattr(app_mod, "mail_agent_service", _DummyMailService())
    client = TestClient(app_mod.app)

    status = client.get("/api/mail/status", params={"session_id": "auth-session"})
    assert status.status_code == 200
    body = status.json()
    assert body["owner_type"] == "user"
    assert body["mode"] == "user"


def test_mail_poll_returns_reconnect_message_for_invalid_refresh_token(monkeypatch):
    app_mod = importlib.reload(importlib.import_module("app"))

    class _ExpiredMailService(_DummyMailService):
        def poll_owner(self, owner_ctx, max_messages=20):
            raise app_mod.MailOAuthRefreshError(
                "Kết nối Gmail đã hết hạn hoặc refresh token bị Google thu hồi. Vui lòng kết nối Gmail lại.",
                invalid_grant=True,
                detail={"error": "invalid_grant"},
            )

    monkeypatch.setattr(app_mod, "mail_agent_service", _ExpiredMailService())
    client = TestClient(app_mod.app)

    resp = client.post("/api/mail/poll", json={"session_id": "s-1"})

    assert resp.status_code == 401
    assert "Vui lòng kết nối Gmail lại" in resp.json()["detail"]
