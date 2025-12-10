import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import mcp_server.server as server  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def test_compare_pdfs_requires_two_files():
    with pytest.raises(HTTPException):
        server.compare_pdfs("q", file_ids=["only_one"], top_k=2)


def test_get_file_summaries_requires_file_ids():
    with pytest.raises(HTTPException):
        server.get_file_summaries([])


def test_retrieve_chunks_without_file_ids_returns_empty(monkeypatch):
    monkeypatch.setattr(server, "_store", None)
    res = server.retrieve_chunks("q", top_k=3, file_ids=[])
    assert res == []
