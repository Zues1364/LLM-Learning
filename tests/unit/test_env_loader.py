import os

from env_loader import load_env, read_bool_env, read_str_env


def test_read_str_env_strips_wrapped_quotes(monkeypatch):
    monkeypatch.setenv("SERVICE_URL", "'https://service.example.com'")

    assert read_str_env("SERVICE_URL") == "https://service.example.com"


def test_read_bool_env_accepts_wrapped_quotes(monkeypatch):
    monkeypatch.setenv("FEATURE_FLAG", '"true"')

    assert read_bool_env("FEATURE_FLAG") is True


def test_read_bool_env_accepts_false_values(monkeypatch):
    monkeypatch.setenv("FEATURE_FLAG", "'off'")

    assert read_bool_env("FEATURE_FLAG", default=True) is False


def test_read_bool_env_uses_default_for_missing_or_unknown(monkeypatch):
    monkeypatch.delenv("FEATURE_FLAG", raising=False)
    assert read_bool_env("FEATURE_FLAG", default=True) is True

    monkeypatch.setenv("FEATURE_FLAG", "maybe")
    assert read_bool_env("FEATURE_FLAG", default=False) is False


def test_load_env_strips_wrapping_quotes_without_overriding_existing(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'APP_COOKIE_SECURE="true"',
                "CORS_ALLOW_ORIGINS='https://example.com'",
                "APP_ENV=production",
            ]
        )
    )
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("APP_COOKIE_SECURE", raising=False)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    load_env(env_file)

    assert os.environ["APP_COOKIE_SECURE"] == "true"
    assert os.environ["CORS_ALLOW_ORIGINS"] == "https://example.com"
    assert os.environ["APP_ENV"] == "development"
