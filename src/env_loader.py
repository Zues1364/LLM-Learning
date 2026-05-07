import os
from pathlib import Path


_TRUE_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_ENV_VALUES = {"0", "false", "f", "no", "n", "off"}


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def read_str_env(env_name: str, default: str = "") -> str:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    value = strip_wrapping_quotes(str(raw))
    return value or default


def read_bool_env(env_name: str, default: bool = False) -> bool:
    raw = read_str_env(env_name, "").lower()
    if not raw:
        return default
    if raw in _TRUE_ENV_VALUES:
        return True
    if raw in _FALSE_ENV_VALUES:
        return False
    return default


def normalize_api_keys() -> None:
    """
    Canonicalize Gemini credentials to GEMINI_API_KEY only.
    - If only GOOGLE_API_KEY exists, migrate it to GEMINI_API_KEY.
    - If both exist, prefer GEMINI_API_KEY.
    - Always remove GOOGLE_API_KEY to avoid SDK ambiguity.
    """
    gemini_key = read_str_env("GEMINI_API_KEY")
    google_key = read_str_env("GOOGLE_API_KEY")

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
        value = strip_wrapping_quotes(value)
        if key and key not in os.environ:
            os.environ[key] = value

    normalize_api_keys()
