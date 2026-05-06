from fastapi.testclient import TestClient

from mcp_client.client import MCPClient
from mcp_server import server as mcp_server


def test_mcp_rejects_missing_api_key_when_required():
    old_require = mcp_server.MCP_REQUIRE_API_KEY
    old_key = mcp_server.MCP_API_KEY
    try:
        mcp_server.MCP_REQUIRE_API_KEY = True
        mcp_server.MCP_API_KEY = "test-secret"
        client = TestClient(mcp_server.app)

        response = client.get("/mcp/discover")

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid MCP API key"
    finally:
        mcp_server.MCP_REQUIRE_API_KEY = old_require
        mcp_server.MCP_API_KEY = old_key


def test_mcp_accepts_valid_api_key_when_required():
    old_require = mcp_server.MCP_REQUIRE_API_KEY
    old_key = mcp_server.MCP_API_KEY
    try:
        mcp_server.MCP_REQUIRE_API_KEY = True
        mcp_server.MCP_API_KEY = "test-secret"
        client = TestClient(mcp_server.app)

        response = client.get("/mcp/discover", headers={"X-MCP-API-Key": "test-secret"})

        assert response.status_code == 200
        assert "tools" in response.json()
    finally:
        mcp_server.MCP_REQUIRE_API_KEY = old_require
        mcp_server.MCP_API_KEY = old_key


def test_mcp_client_sends_api_key_header(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"tools": ["retrieve_chunks"]}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setenv("MCP_API_KEY", "client-secret")
    monkeypatch.setenv("MCP_SERVER_URL", "http://mcp:8000")
    monkeypatch.setattr("mcp_client.client.requests.get", fake_get)

    tools = MCPClient().discover(timeout=3)

    assert tools == ["retrieve_chunks"]
    assert captured["url"] == "http://mcp:8000/mcp/discover"
    assert captured["headers"] == {"X-MCP-API-Key": "client-secret"}
    assert captured["timeout"] == 3
