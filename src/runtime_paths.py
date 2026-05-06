import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_runtime_dir(env_name: str, default_path: Path) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    path = Path(raw).expanduser() if raw else default_path
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = _resolve_runtime_dir("APP_DATA_DIR", BASE_DIR / "data")
CACHE_DIR = DATA_DIR / "cache"
PDF_DIR = DATA_DIR / "pdfs"
RESOURCE_DIR = DATA_DIR / "resources"
RESOURCE_PDF_DIR = RESOURCE_DIR / "pdfs"
RESOURCE_HTML_DIR = RESOURCE_DIR / "html"
SESSION_CACHE_DIR = DATA_DIR / "session_cache"
MEMORY_DB = DATA_DIR / "memory.db"


def ensure_runtime_dirs() -> None:
    for path in (
        DATA_DIR,
        CACHE_DIR,
        PDF_DIR,
        RESOURCE_DIR,
        RESOURCE_PDF_DIR,
        RESOURCE_HTML_DIR,
        SESSION_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
