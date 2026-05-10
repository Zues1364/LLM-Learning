import time
from typing import Any, Dict, List

import requests

from env_loader import read_str_env

class MCPClient:
    """Python MCP-Client (HTTP)."""
    def __init__(self, server_url: str | None = None):
        default_url = read_str_env("MCP_SERVER_URL", "http://localhost:8000")
        raw_base = (server_url or default_url).rstrip("/")
        fallback_urls_raw = read_str_env("MCP_SERVER_FALLBACK_URLS", "")
        fallback_urls = [
            url.strip().rstrip("/")
            for url in fallback_urls_raw.split(",")
            if str(url).strip()
        ]
        self.base_candidates = list(dict.fromkeys([raw_base, *fallback_urls]))
        self.base = self.base_candidates[0]
        self.api_key = read_str_env("MCP_API_KEY")
        try:
            retries = int(read_str_env("MCP_RETRY_ATTEMPTS", "3"))
        except ValueError:
            retries = 3
        self.retry_attempts = max(1, retries)
        try:
            backoff = float(read_str_env("MCP_RETRY_BACKOFF_SEC", "0.75"))
        except ValueError:
            backoff = 0.75
        self.retry_backoff_sec = max(0.0, backoff)

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"X-MCP-API-Key": self.api_key}

    def _request_with_retry(
        self,
        method: str,
        path: str,
        timeout: float | None = None,
        json_payload: Dict[str, Any] | None = None,
    ) -> requests.Response:
        retriable_status = {502, 503, 504}
        last_exc: Exception | None = None
        method_value = str(method or "").upper()

        for attempt in range(self.retry_attempts):
            for base in self.base_candidates:
                url = f"{base}{path}"
                try:
                    if method_value == "GET":
                        resp = requests.get(url, headers=self._headers(), timeout=timeout)
                    else:
                        resp = requests.post(
                            url,
                            json=json_payload,
                            headers=self._headers(),
                            timeout=timeout,
                        )
                    resp.raise_for_status()
                    self.base = base
                    return resp
                except requests.HTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status not in retriable_status:
                        raise
                    last_exc = exc
                except requests.RequestException as exc:
                    last_exc = exc
            if attempt < self.retry_attempts - 1 and self.retry_backoff_sec > 0:
                time.sleep(self.retry_backoff_sec * (2**attempt))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("MCP request failed without specific error.")

    def discover(self, timeout: float | None = None) -> List[str]:
        r = self._request_with_retry("GET", "/mcp/discover", timeout=timeout)
        return r.json()["tools"]

    def invoke(self, tool: str, args: Dict[str, Any], timeout: float | None = None) -> Any:
        payload = {"tool": tool, "args": args}
        r = self._request_with_retry("POST", "/mcp/invoke", timeout=timeout, json_payload=payload)
        return r.json()["result"]
