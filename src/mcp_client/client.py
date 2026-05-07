import requests
from typing import Any, Dict, List

from env_loader import read_str_env

class MCPClient:
    """Python MCP-Client (HTTP)."""
    def __init__(self, server_url: str | None = None):
        default_url = read_str_env("MCP_SERVER_URL", "http://localhost:8000")
        self.base = (server_url or default_url).rstrip("/")
        self.api_key = read_str_env("MCP_API_KEY")

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"X-MCP-API-Key": self.api_key}

    def discover(self, timeout: float | None = None) -> List[str]:
        r = requests.get(f"{self.base}/mcp/discover", headers=self._headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()["tools"]

    def invoke(self, tool: str, args: Dict[str, Any], timeout: float | None = None) -> Any:
        payload = {"tool": tool, "args": args}
        r = requests.post(f"{self.base}/mcp/invoke", json=payload, headers=self._headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()["result"]
