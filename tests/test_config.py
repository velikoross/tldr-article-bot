import pytest

from bot.config import Settings


def test_from_env(monkeypatch):
    monkeypatch.setenv("ADMIN_IDS", "1, 2")
    monkeypatch.setenv("ALLOWED_IDS", "3;4 5")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_REASONING", "Low")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"provider": {"allow_fallbacks": false}}')
    s = Settings.from_env()
    assert s.admin_ids == {1, 2} and s.allowed_ids == {3, 4, 5}
    assert s.base_url == "https://api.anthropic.com"
    assert s.llm_reasoning == "low"
    assert s.llm_max_tokens == 8000
    assert s.llm_extra_body == {"provider": {"allow_fallbacks": False}}


def test_custom_base_url_and_bad_values(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1/")
    assert Settings.from_env().base_url == "http://ollama:11434/v1"
    monkeypatch.setenv("LLM_REASONING", "extreme")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("LLM_REASONING", "")
    monkeypatch.setenv("ADMIN_IDS", "abc")
    with pytest.raises(ValueError):
        Settings.from_env()
