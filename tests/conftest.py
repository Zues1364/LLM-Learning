import json
import os
import types
from pathlib import Path

import pytest
import requests

# Ensure src on path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import utils  # noqa: E402


# Test modules import the app/MCP modules at collection time. Keep the default
# pytest environment local-only so .env cloud credentials are not loaded into
# tests and cannot mutate a real Supabase project. Individual tests can still
# opt in with monkeypatch.setenv when they specifically exercise readiness.
for _cloud_var in (
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_DB_URL",
    "SUPABASE_STORAGE_BUCKET",
):
    os.environ[_cloud_var] = ""


@pytest.fixture(autouse=True)
def block_external_requests(monkeypatch):
    """
    Prevent accidental real HTTP calls. Tests can override per-case.
    """
    def _blocked(*args, **kwargs):
        raise RuntimeError("External HTTP call blocked in tests")

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    yield


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """
    Avoid loading real embedding models.
    """
    class DummySentenceTransformer:
        def __init__(self, model_name=None):
            self.model_name = model_name

        def encode(self, texts, show_progress_bar=False):
            return [[0.0, 1.0, 0.0] for _ in texts]

    monkeypatch.setattr(utils, "SentenceTransformer", DummySentenceTransformer)
    yield


@pytest.fixture(autouse=True)
def patch_partition_pdf(monkeypatch):
    """
    Stub unstructured partition to avoid heavy OCR.
    """
    class DummyElement:
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text

    def _fake_partition_pdf(*args, **kwargs):
        return [DummyElement("chunk-1"), DummyElement("chunk-2")]

    if hasattr(utils, "partition_pdf"):
        monkeypatch.setattr(utils, "partition_pdf", _fake_partition_pdf)
    yield


@pytest.fixture
def temp_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(utils, "CACHE_DIR", cache_dir)
    return cache_dir
