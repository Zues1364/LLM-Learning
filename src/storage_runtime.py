import logging
from pathlib import Path
from typing import Optional

from runtime_paths import CACHE_DIR, DATA_DIR, PDF_DIR, RESOURCE_DIR
from supabase_support import SupabaseBlobStore, supabase_enabled

logger = logging.getLogger(__name__)

_blob_store: Optional[SupabaseBlobStore] = None


def get_blob_store() -> SupabaseBlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = SupabaseBlobStore()
        _blob_store.ensure_bucket()
    return _blob_store


def blob_mode_enabled() -> bool:
    return supabase_enabled()


def build_transcript_key(file_name: str, session_id: Optional[str] = None) -> str:
    sid = str(session_id or "global").strip() or "global"
    return f"sessions/{sid}/pdfs/{Path(file_name).name}"


def build_resource_key(
    file_name: str,
    resource_type: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    safe_name = Path(file_name).name
    kind = "pdf" if resource_type.lower() == "pdf" else "html"
    if user_id:
        safe_user = str(user_id).strip().replace("/", "_")
        return f"resources/users/{safe_user}/{kind}/{safe_name}"
    if session_id:
        safe_session = str(session_id).strip().replace("/", "_")
        return f"resources/{safe_session}/{kind}/{safe_name}"
    return f"resources/global/{kind}/{safe_name}"


def build_resource_config_key(session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    if user_id:
        safe_user = str(user_id).strip().replace("/", "_")
        return f"resources/users/{safe_user}/config.json"
    if session_id:
        safe_session = str(session_id).strip().replace("/", "_")
        return f"resources/{safe_session}/config.json"
    return "resources/global/config.json"


def local_path_from_key(object_key: str) -> Path:
    key = str(object_key or "").strip().replace("\\", "/")
    parts = [part for part in key.split("/") if part]
    if not parts:
        return DATA_DIR / "_unknown"

    if key == "resources/global/config.json":
        return RESOURCE_DIR / "config.json"
    if len(parts) == 4 and parts[0] == "resources" and parts[1] == "users" and parts[3] == "config.json":
        return RESOURCE_DIR / "users" / parts[2] / "config.json"
    if len(parts) == 3 and parts[0] == "resources" and parts[2] == "config.json":
        return RESOURCE_DIR / "sessions" / parts[1] / "config.json"

    if len(parts) >= 4 and parts[0] == "sessions" and parts[2] == "pdfs":
        # sessions/<sid>/pdfs/<file>
        if parts[1] == "global":
            return PDF_DIR / parts[3]
        return DATA_DIR / "sessions" / parts[1] / "pdfs" / "/".join(parts[3:])

    if len(parts) >= 4 and parts[0] == "resources" and parts[1] == "users":
        # resources/users/<uid>/(pdf|html)/<file>
        user_id = parts[2]
        if parts[3] == "pdf":
            return RESOURCE_DIR / "users" / user_id / "pdfs" / "/".join(parts[4:])
        if parts[3] == "html":
            return RESOURCE_DIR / "users" / user_id / "html" / "/".join(parts[4:])

    if len(parts) >= 4 and parts[0] == "resources" and parts[1] == "global":
        if parts[2] == "pdf":
            return RESOURCE_DIR / "pdfs" / "/".join(parts[3:])
        if parts[2] == "html":
            return RESOURCE_DIR / "html" / "/".join(parts[3:])

    if len(parts) >= 4 and parts[0] == "resources":
        # resources/<sid>/(pdf|html)/<file>
        sid = parts[1]
        if parts[2] == "pdf":
            return RESOURCE_DIR / "sessions" / sid / "pdfs" / "/".join(parts[3:])
        if parts[2] == "html":
            return RESOURCE_DIR / "sessions" / sid / "html" / "/".join(parts[3:])

    if parts[0] == "cache":
        return CACHE_DIR / "/".join(parts[1:])

    # Fallback for legacy shape
    return DATA_DIR / key


def sync_blob_to_local() -> None:
    if not blob_mode_enabled():
        return
    store = get_blob_store()
    objects = store.list_objects("")
    for item in objects:
        key = item.key
        local_path = local_path_from_key(key)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            store.download_to_path(key, local_path)
        except Exception as exc:
            logger.warning("Failed syncing storage object %s -> %s: %s", key, local_path, exc)


def upload_local_file_to_blob(local_path: Path, object_key: str, content_type: str = "application/octet-stream") -> None:
    if not blob_mode_enabled():
        return
    store = get_blob_store()
    store.upload_file(object_key, local_path, content_type=content_type)


def delete_blob_key(object_key: str) -> None:
    if not blob_mode_enabled():
        return
    store = get_blob_store()
    try:
        store.delete(object_key)
    except Exception as exc:
        logger.warning("Failed deleting storage object %s: %s", object_key, exc)
