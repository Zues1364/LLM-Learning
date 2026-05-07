from sitecustomize import normalize_railway_uvicorn_args


def test_replaces_literal_port_with_railway_port() -> None:
    args = [
        "python",
        "-m",
        "uvicorn",
        "app:app",
        "--app-dir",
        "src",
        "--host",
        "0.0.0.0",
        "--port",
        "$PORT",
    ]

    normalized = normalize_railway_uvicorn_args(args, {"PORT": "12345"})

    assert normalized[-1] == "12345"
    assert normalized[normalized.index("--host") + 1] == "0.0.0.0"


def test_replaces_inline_literal_port() -> None:
    args = ["python", "-m", "uvicorn", "app:app", "--port=$PORT"]

    normalized = normalize_railway_uvicorn_args(args, {"PORT": "12345"})

    assert normalized[-1] == "--port=12345"


def test_mcp_uses_ipv6_host_for_railway_private_networking() -> None:
    args = [
        "python",
        "-m",
        "uvicorn",
        "src.mcp_server.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "$PORT",
    ]

    normalized = normalize_railway_uvicorn_args(args, {"PORT": "8000"})

    assert normalized[normalized.index("--host") + 1] == "::"
    assert normalized[-1] == "8000"


def test_mcp_default_port_when_port_is_missing() -> None:
    args = ["python", "-m", "uvicorn", "src.mcp_server.server:app", "--port", "$PORT"]

    normalized = normalize_railway_uvicorn_args(args, {})

    assert normalized[-1] == "8000"
