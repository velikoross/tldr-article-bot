"""Настройки бота из переменных окружения (.env)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

REASONING_LEVELS = ("", "off", "low", "medium", "high")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должно быть целым числом, сейчас: {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должно быть числом, сейчас: {raw!r}") from exc


def _ids(name: str) -> frozenset[int]:
    raw = os.getenv(name, "")
    result = set()
    for part in raw.replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
        except ValueError as exc:
            raise ValueError(f"{name}: неверный Telegram ID {part!r}") from exc
    return frozenset(result)


@dataclass(frozen=True)
class Settings:
    bot_token: str = ""
    admin_ids: frozenset[int] = frozenset()
    allowed_ids: frozenset[int] = frozenset()
    daily_limit_per_user: int = 0

    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash"
    llm_base_url: str = ""
    llm_max_tokens: int = 8000
    llm_reasoning: str = ""
    llm_temperature: float = 0.3
    llm_timeout: float = 180.0
    llm_context_tokens: int = 32000
    llm_concurrency: int = 3
    llm_instructions: str = ""
    llm_extra_body: dict = field(default_factory=dict)

    default_language: str = "ru"
    default_translation_language: str = "en"
    full_translation_format: str = "md"  # md | telegraph
    full_translation_max_chars: int = 200_000

    fetch_timeout: float = 20.0
    fetch_max_bytes: int = 5 * 1024 * 1024
    fetch_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
    max_links_per_message: int = 5
    min_text_length: int = 500
    min_article_chars: int = 300
    cache_ttl_days: int = 30
    db_path: str = "/data/bot.db"
    log_level: str = "INFO"

    @property
    def base_url(self) -> str:
        url = self.llm_base_url or DEFAULT_BASE_URLS.get(self.llm_provider, "")
        return url.rstrip("/")

    @property
    def input_chars_budget(self) -> int:
        """Сколько символов текста статьи помещается в один запрос.

        Оценка грубая: ~3 символа на токен, минус ответ и служебная часть промпта.
        """
        tokens = self.llm_context_tokens - self.llm_max_tokens - 2000
        return max(tokens, 2000) * 3

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower() or "openrouter"
        if provider in ("openai_compatible", "openai-compatible", "compatible"):
            provider = "openai"
        if provider not in DEFAULT_BASE_URLS:
            raise ValueError(
                f"LLM_PROVIDER={provider!r} не поддерживается: используйте openrouter, openai или anthropic"
            )
        reasoning = os.getenv("LLM_REASONING", "").strip().lower()
        if reasoning not in REASONING_LEVELS:
            raise ValueError("LLM_REASONING: допустимо off, low, medium, high или пусто")
        extra_raw = os.getenv("LLM_EXTRA_BODY", "").strip()
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY должен быть JSON-объектом: {exc}") from exc
        full_format = os.getenv("FULL_TRANSLATION_FORMAT", "md").strip().lower() or "md"
        if full_format not in ("md", "telegraph"):
            raise ValueError("FULL_TRANSLATION_FORMAT: md или telegraph")
        defaults = cls()
        return cls(
            bot_token=os.getenv("BOT_TOKEN", "").strip(),
            admin_ids=_ids("ADMIN_IDS"),
            allowed_ids=_ids("ALLOWED_IDS"),
            daily_limit_per_user=_int("DAILY_LIMIT_PER_USER", 0),
            llm_provider=provider,
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip() or defaults.llm_model,
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_max_tokens=max(_int("LLM_MAX_TOKENS", 8000), 256),
            llm_reasoning=reasoning,
            llm_temperature=_float("LLM_TEMPERATURE", 0.3),
            llm_timeout=_float("LLM_TIMEOUT", 180.0),
            llm_context_tokens=_int("LLM_CONTEXT_TOKENS", 32000),
            llm_concurrency=max(_int("LLM_CONCURRENCY", 3), 1),
            llm_instructions=os.getenv("LLM_INSTRUCTIONS", "").strip(),
            llm_extra_body=extra if isinstance(extra, dict) else {},
            default_language=os.getenv("DEFAULT_LANGUAGE", "ru").strip().lower() or "ru",
            default_translation_language=(
                os.getenv("DEFAULT_TRANSLATION_LANGUAGE", "en").strip().lower() or "en"
            ),
            full_translation_format=full_format,
            full_translation_max_chars=_int("FULL_TRANSLATION_MAX_CHARS", 200_000),
            fetch_timeout=_float("FETCH_TIMEOUT", 20.0),
            fetch_max_bytes=_int("FETCH_MAX_BYTES", 5 * 1024 * 1024),
            fetch_user_agent=os.getenv("FETCH_USER_AGENT", "").strip() or defaults.fetch_user_agent,
            max_links_per_message=max(_int("MAX_LINKS_PER_MESSAGE", 5), 1),
            min_text_length=_int("MIN_TEXT_LENGTH", 500),
            cache_ttl_days=_int("CACHE_TTL_DAYS", 30),
            db_path=os.getenv("DB_PATH", "/data/bot.db").strip() or "/data/bot.db",
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )
