import os
import requests
from typing import Any, Dict, List

class MCPClient:
    """Python MCP-Client (HTTP)."""
    def __init__(self, server_url: str | None = None):
        default_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        self.base = (server_url or default_url).rstrip("/")

    def discover(self, timeout: float | None = None) -> List[str]:
        r = requests.get(f"{self.base}/mcp/discover", timeout=timeout)
        r.raise_for_status()
        return r.json()["tools"]

    def invoke(self, tool: str, args: Dict[str, Any], timeout: float | None = None) -> Any:
        payload = {"tool": tool, "args": args}
        r = requests.post(f"{self.base}/mcp/invoke", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["result"]
