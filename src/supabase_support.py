import os
import tempfile
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def supabase_enabled() -> bool:
    return bool(_env("SUPABASE_URL") and _env("SUPABASE_SERVICE_ROLE_KEY"))


def pgvector_enabled() -> bool:
    return bool(_env("SUPABASE_DB_URL"))


def _normalize_storage_key(key: str) -> str:
    cleaned = str(key or "").replace("\\", "/").strip()
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    parts = []
    for part in cleaned.split("/"):
        if not part:
            continue
        normalized = unicodedata.normalize("NFKD", part)
        normalized = normalized.replace(chr(0x0110), "D").replace(chr(0x0111), "d")
        ascii_part = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_part = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_part).strip("._-")
        parts.append(ascii_part or "_")
    return "/".join(parts)


@dataclass
class BlobObject:
    key: str
    size: int


class SupabaseBlobStore:
    """
    Minimal Supabase Storage adapter.
    Uses service role key and a private bucket.
    """

    def __init__(self, bucket: Optional[str] = None):
        self.bucket = bucket or _env("SUPABASE_STORAGE_BUCKET", "rag-files")
        self.enabled = supabase_enabled()
        self._client = None
        if self.enabled:
            self._client = _create_supabase_client()

    def ensure_bucket(self) -> None:
        if not self.enabled:
            return
        client = self._client
        assert client is not None
        try:
            existing = client.storage.list_buckets()
            names = {str(item.get("name") or "") for item in (existing or [])}
            if self.bucket in names:
                return
            client.storage.create_bucket(self.bucket, {"public": False})
        except Exception:
            # Safe to ignore when bucket exists or account denies create bucket.
            pass

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        if not self.enabled:
            return
        object_key = _normalize_storage_key(key)
        client = self._client
        assert client is not None
        options = {"upsert": "true", "content-type": content_type}
        attempts = max(1, int(_env("SUPABASE_STORAGE_UPLOAD_RETRIES", "3") or "3"))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                client.storage.from_(self.bucket).upload(object_key, data, file_options=options)
                return
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(min(2 * attempt, 8))
                self._client = _create_supabase_client()
                client = self._client
        assert last_error is not None
        raise last_error

    def upload_file(self, key: str, file_path: Path, content_type: str = "application/octet-stream") -> None:
        data = file_path.read_bytes()
        self.upload_bytes(key, data, content_type=content_type)

    def download_bytes(self, key: str) -> bytes:
        if not self.enabled:
            raise RuntimeError("Supabase storage is not enabled.")
        object_key = _normalize_storage_key(key)
        client = self._client
        assert client is not None
        payload = client.storage.from_(self.bucket).download(object_key)
        if payload is None:
            raise FileNotFoundError(f"Storage object not found: {object_key}")
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, memoryview):
            return bytes(payload)
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return bytes(payload)

    def download_to_path(self, key: str, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(self.download_bytes(key))
        return target_path

    def download_to_temp(self, key: str, suffix: Optional[str] = None) -> Path:
        object_key = _normalize_storage_key(key)
        file_suffix = suffix or Path(object_key).suffix or ".bin"
        fd, temp_name = tempfile.mkstemp(prefix="rag_blob_", suffix=file_suffix)
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.write_bytes(self.download_bytes(object_key))
        return temp_path

    def list_objects(self, prefix: str = "") -> List[BlobObject]:
        if not self.enabled:
            return []
        object_prefix = _normalize_storage_key(prefix)
        # Recursive listing from storage.objects via Postgres for deterministic sync.
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                if object_prefix:
                    cur.execute(
                        """
                        SELECT name, COALESCE((metadata->>'size')::bigint, 0)
                        FROM storage.objects
                        WHERE bucket_id = %s AND name LIKE %s
                        ORDER BY name ASC
                        """,
                        (self.bucket, f"{object_prefix}%"),
                    )
                else:
                    cur.execute(
                        """
                        SELECT name, COALESCE((metadata->>'size')::bigint, 0)
                        FROM storage.objects
                        WHERE bucket_id = %s
                        ORDER BY name ASC
                        """,
                        (self.bucket,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [BlobObject(key=str(row[0]), size=int(row[1] or 0)) for row in rows]

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        object_key = _normalize_storage_key(key)
        parent = str(Path(object_key).parent).replace("\\", "/")
        parent = "" if parent == "." else parent
        name = Path(object_key).name
        for obj in self.list_objects(parent):
            if Path(obj.key).name == name:
                return True
        return False

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        object_key = _normalize_storage_key(key)
        client = self._client
        assert client is not None
        client.storage.from_(self.bucket).remove([object_key])


def _create_supabase_client():
    # Import lazily so local mode does not require supabase package at import-time.
    from supabase import create_client

    return create_client(_env("SUPABASE_URL"), _env("SUPABASE_SERVICE_ROLE_KEY"))


def get_pg_connection():
    dsn = _env("SUPABASE_DB_URL")
    if not dsn:
        raise RuntimeError("SUPABASE_DB_URL is required for Postgres/pgvector mode.")
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def check_postgres_ready() -> Optional[str]:
    if not pgvector_enabled():
        return "disabled"
    try:
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
        finally:
            conn.close()
        return None
    except Exception as exc:
        return str(exc)


def check_blob_ready(blob_store: SupabaseBlobStore) -> Optional[str]:
    if not blob_store.enabled:
        return "disabled"
    try:
        blob_store.ensure_bucket()
        _ = blob_store.list_objects("")
        return None
    except Exception as exc:
        return str(exc)
