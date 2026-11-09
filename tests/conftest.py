from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.config import Settings
from bot.db import Database
from bot.fetcher import FetchResult
from bot.llm.base import LLMClient, LLMResult
from bot.service import ArticleService
from bot.urls import normalize_url

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def summary_json(**overrides) -> str:
    data = {
        "title": "Как мы ускорили CI в четыре раза",
        "lang": "ru",
        "verdict": "read",
        "verdict_reason": "Практический разбор с цифрами.",
        "kind": "practice",
        "audience": "DevOps-инженерам: приёмы кэширования.",
        "warnings": [],
        "summary": ["Сборка шла 40 минут.", "Кэш зависимостей сэкономил 15 минут.", "Тесты разбили на 4 шарда."],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


class FakeLLM(LLMClient):
    """Заглушка нейросети: отвечает по очереди из списка или функцией."""

    provider = "fake"

    def __init__(self, responses=None, handler=None, **kwargs):
        kwargs.setdefault("api_key", "test")
        kwargs.setdefault("model", "fake-model")
        kwargs.setdefault("base_url", "http://fake")
        kwargs.setdefault("retries", 0)
        super().__init__(**kwargs)
        self.responses = list(responses or [])
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    async def _request(self, system: str, user: str, max_tokens: int) -> LLMResult:
        self.calls.append((system, user))
        if self.handler:
            value = self.handler(system, user)
        else:
            value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, LLMResult):
            return value
        return LLMResult(text=value, finish_reason="stop")


class FakeFetcher:
    def __init__(self, pages: dict[str, tuple[bytes, str]] | None = None):
        self.pages = pages or {}
        self.calls: list[str] = []

    async def fetch(self, url: str) -> FetchResult:
        self.calls.append(url)
        pages = {normalize_url(k): v for k, v in self.pages.items()}
        content, kind = pages[normalize_url(url)]
        return FetchResult(url=url, content=content, content_type=kind, encoding="utf-8", status=200)

    async def close(self):
        pass


@pytest.fixture
def settings() -> Settings:
    return Settings(
        bot_token="x",
        admin_ids=frozenset({1}),
        allowed_ids=frozenset({2}),
        db_path=":memory:",
        llm_context_tokens=32000,
        llm_max_tokens=8000,
    )


@pytest.fixture
async def db():
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def make_service(db, settings):
    def factory(llm, fetcher=None, **overrides):
        s = settings
        if overrides:
            from dataclasses import replace

            s = replace(settings, **overrides)
        return ArticleService(s, db, fetcher or FakeFetcher(), llm)

    return factory
