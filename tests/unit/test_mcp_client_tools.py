import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_client.client as client_mod  # noqa: E402
import mcp_tools  # noqa: E402
import pytest  # noqa: E402


class DummyResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_mcp_client_discover_calls_correct_url(monkeypatch):
    called = {}

    def fake_get(url):
        called["url"] = url
        return DummyResponse({"tools": ["a", "b"]})

    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    cli = client_mod.MCPClient(server_url="http://example.com")
    tools = cli.discover()

    assert called["url"].endswith("/mcp/discover")
    assert tools == ["a", "b"]


def test_mcp_client_invoke_posts_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None):
        captured["url"] = url
        captured["json"] = json
        return DummyResponse({"result": "ok"})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    cli = client_mod.MCPClient(server_url="http://example.com")
    res = cli.invoke("tool_name", {"foo": "bar"})

    assert captured["url"].endswith("/mcp/invoke")
    assert captured["json"] == {"tool": "tool_name", "args": {"foo": "bar"}}
    assert res == "ok"


def test_mcp_client_invoke_raises_on_http_error(monkeypatch):
    def fake_post(url, json=None):
        return DummyResponse({"result": "fail"}, status=500)

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    cli = client_mod.MCPClient(server_url="http://example.com")
    with pytest.raises(RuntimeError):
        cli.invoke("tool", {})


def test_mcp_tools_wrappers_call_invoke(monkeypatch):
    calls = []

    class DummyClient:
        def invoke(self, tool, args):
            calls.append((tool, args))
            # Tools that join list outputs
            if tool in {"retrieve_chunks", "compare_pdfs", "get_file_summaries", "memory_get", "web_search_tool"}:
                return [f"{tool}-ok"]
            return f"{tool}-ok"

    dummy = DummyClient()

    assert mcp_tools.tool_retrieve("q", 5, ["f1"], client=dummy) == "retrieve_chunks-ok"
    assert mcp_tools.tool_compare_pdfs("q", ["f1", "f2"], 5, client=dummy) == "compare_pdfs-ok"
    assert mcp_tools.tool_get_file_summaries(["f1"], client=dummy) == "get_file_summaries-ok"
    assert mcp_tools.tool_memory_get("s", 5, client=dummy) == "memory_get-ok"
    assert mcp_tools.tool_memory_add("s", "q", "a", 1, client=dummy) == "memory_add-ok"
    assert mcp_tools.tool_web_search("q", 3, client=dummy) == "web_search_tool-ok"

    tools_called = [c[0] for c in calls]
    assert set(tools_called) == {
        "retrieve_chunks",
        "compare_pdfs",
        "get_file_summaries",
        "memory_get",
        "memory_add",
        "web_search_tool",
    }
