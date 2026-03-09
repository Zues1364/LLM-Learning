import os
from pathlib import Path


def normalize_api_keys() -> None:
    """
    Canonicalize Gemini credentials to GEMINI_API_KEY only.
    - If only GOOGLE_API_KEY exists, migrate it to GEMINI_API_KEY.
    - If both exist, prefer GEMINI_API_KEY.
    - Always remove GOOGLE_API_KEY to avoid SDK ambiguity.
    """
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()

    if not gemini_key and google_key:
        os.environ["GEMINI_API_KEY"] = google_key
        gemini_key = google_key

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    os.environ.pop("GOOGLE_API_KEY", None)


def load_env(env_path: str | Path | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ if not already set."""
    path = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        normalize_api_keys()
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

    normalize_api_keys()
