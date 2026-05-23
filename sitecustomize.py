"""Normalize Railway-injected Uvicorn arguments before CLI parsing."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence


def _target_app(args: Sequence[str]) -> str | None:
    for arg in args:
        if arg in {"app:app", "src.mcp_server.server:app"}:
            return arg
    return None


def _default_port_for(args: Sequence[str]) -> str | None:
    target = _target_app(args)
    if target == "src.mcp_server.server:app":
        return "8000"
    if target == "app:app":
        return "9000"
    return None


def normalize_railway_uvicorn_args(
    args: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Let legacy Railway start commands using literal $PORT keep working."""

    env = env or os.environ
    normalized = list(args)
    port = env.get("PORT") or _default_port_for(normalized)

    if port:
        for index, arg in enumerate(normalized):
            if arg == "$PORT":
                normalized[index] = port
            elif arg == "--port=$PORT":
                normalized[index] = f"--port={port}"

    return normalized


sys.argv[:] = normalize_railway_uvicorn_args(sys.argv)
