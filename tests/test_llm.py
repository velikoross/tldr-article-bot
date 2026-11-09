import json

import httpx
import pytest

from bot.llm import (
    LLMEmptyResponse,
    LLMProviderError,
    LLMResult,
    LLMTransientError,
    LLMTruncated,
)
from bot.llm.anthropic import AnthropicClient
from bot.llm.openai_compat import OpenAICompatClient
from bot.summarizer import BadModelOutput, Summarizer, extract_json, parse_summary

from .conftest import FakeLLM, summary_json


# --- разбор ответа модели ---

def test_parse_plain_json():
    data = parse_summary(summary_json())
    assert data["verdict"] == "read"
    assert data["kind"] == "practice"
    assert len(data["summary"]) == 3


def test_parse_json_in_fences_with_chatter_and_think():
    text = "<think>hmm, let me think {not json}</think>Вот ответ:\n```json\n" + summary_json() + "\n```\nГотово!"
    assert parse_summary(text)["title"] == "Как мы ускорили CI в четыре раза"


def test_parse_json_followed_by_braces():
    text = summary_json() + "\n\nP.S. {сноска}"
    assert parse_summary(text)["verdict"] == "read"


def test_parse_normalizes_fields():
    raw = json.dumps({
        "verdict": "Пролистать",
        "kind": "something-odd",
        "warnings": ["Реклама курса в конце", {"type": "outdated", "text": "Данные 2019 года"}],
        "summary": "• первый пункт\n• второй пункт\n" + "\n".join(f"- p{i}" for i in range(10)),
    }, ensure_ascii=False)
    data = parse_summary(raw)
    assert data["verdict"] == "skim"
    assert data["kind"] == "other"
    assert data["warnings"][0] == {"type": "other", "text": "Реклама курса в конце"}
    assert data["warnings"][1]["type"] == "outdated"
    assert data["summary"][0] == "первый пункт"
    assert len(data["summary"]) == 8


def test_parse_rejects_non_json_and_missing_summary():
    with pytest.raises(ValueError):
        parse_summary("Извините, я не могу помочь")
    with pytest.raises(ValueError):
        parse_summary('{"verdict": "read"}')
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")


async def test_non_json_answer_is_retried_once():
    llm = FakeLLM(["Конечно! Вот выжимка: статья хорошая.", summary_json()])
    s = Summarizer(llm, input_budget=50_000)
    result = await s.complete_json("sys", "user", parse_summary)
    assert result["verdict"] == "read"
    assert len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1][1]


async def test_non_json_twice_gives_clear_error():
    llm = FakeLLM(["не json", "опять не json", summary_json()])
    s = Summarizer(llm, input_budget=50_000)
    with pytest.raises(BadModelOutput, match="JSON"):
        await s.complete_json("sys", "user", parse_summary)
    assert len(llm.calls) == 2  # ровно один повтор


# --- пустые и обрезанные ответы ---

async def test_empty_answer_is_clear_error():
    llm = FakeLLM([LLMResult(text="   ", finish_reason="stop", reasoning_chars=5000)])
    with pytest.raises(LLMEmptyResponse, match="LLM_MAX_TOKENS"):
        await llm.complete("s", "u")


async def test_truncated_answer_is_clear_error():
    llm = FakeLLM([LLMResult(text='{"summary": ["нача', finish_reason="length")], max_tokens=8000)
    with pytest.raises(LLMTruncated, match="8000"):
        await llm.complete("s", "u")


async def test_truncated_answer_is_not_retried_as_json():
    llm = FakeLLM([LLMResult(text='{"summary": ["нача', finish_reason="length")])
    s = Summarizer(llm, input_budget=50_000)
    with pytest.raises(LLMTruncated):
        await s.complete_json("s", "u", parse_summary)
    assert len(llm.calls) == 1


async def test_max_tokens_never_below_setting():
    seen = {}

    class Probe(FakeLLM):
        async def _request(self, system, user, max_tokens):
            seen["max_tokens"] = max_tokens
            return LLMResult(text="ok", finish_reason="stop")

    llm = Probe(max_tokens=8000)
    await llm.complete("s", "u", max_tokens=500)
    assert seen["max_tokens"] == 8000


# --- HTTP-клиенты (сеть — заглушка httpx.MockTransport) ---

def openrouter(handler, **kwargs) -> OpenAICompatClient:
    kwargs.setdefault("reasoning", "low")
    return OpenAICompatClient(
        openrouter=True, api_key="k", model="vendor/model", base_url="https://openrouter.ai/api/v1",
        max_tokens=8000, retries=2, transport=httpx.MockTransport(handler), **kwargs,
    )


async def test_openrouter_request_body_and_reasoning():
    captured = {}

    def handler(request: httpx.Request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "vendor/model",
            "choices": [{"message": {"content": "привет", "reasoning": "думаю..."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    llm = openrouter(handler)
    result = await llm.complete("sys", "user")
    assert result.text == "привет"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["reasoning"] == {"effort": "low", "exclude": True}
    assert captured["body"]["max_tokens"] == 8000
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_reasoning_off_and_default():
    off = openrouter(lambda r: None, reasoning="off").build_body("s", "u", 8000)
    assert off["reasoning"] == {"enabled": False}
    default = openrouter(lambda r: None, reasoning="").build_body("s", "u", 8000)
    assert "reasoning" not in default


def test_openai_official_uses_max_completion_tokens():
    llm = OpenAICompatClient(api_key="k", model="gpt-x", base_url="https://api.openai.com/v1", reasoning="high")
    body = llm.build_body("s", "u", 9000)
    assert body["max_completion_tokens"] == 9000
    assert body["reasoning_effort"] == "high"
    compat = OpenAICompatClient(api_key="", model="deepseek-chat", base_url="https://api.deepseek.com/v1")
    assert compat.build_body("s", "u", 9000)["max_tokens"] == 9000


async def test_openrouter_4xx_shows_provider_text_and_is_not_repeated(caplog):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": {
            "message": "No endpoints found matching your data policy",
            "code": 404,
            "metadata": {"raw": "Enable 'Allow training' in privacy settings"},
        }})

    llm = openrouter(handler)
    with caplog.at_level("ERROR", logger="bot.llm"):
        with pytest.raises(LLMProviderError) as err:
            await llm.complete("sys", "same request")
        with pytest.raises(LLMProviderError):
            await llm.complete("sys", "same request")
    assert "No endpoints found matching your data policy" in str(err.value)
    assert "Allow training" in str(err.value)
    assert err.value.status == 404
    assert len(calls) == 1  # без повторов, второй раз — из памяти
    assert sum("No endpoints found" in r.message for r in caplog.records) == 1  # в лог один раз

    llm.reset_rejections()  # /llmtest после исправления настроек
    with pytest.raises(LLMProviderError):
        await llm.complete("sys", "same request")
    assert len(calls) == 2


async def test_error_inside_200_response():
    def handler(request):
        return httpx.Response(200, json={"error": {"message": "Provider returned error", "code": 400}})

    with pytest.raises(LLMProviderError, match="Provider returned error"):
        await openrouter(handler).complete("s", "u")


async def test_5xx_is_retried(monkeypatch):
    import bot.llm.base as base

    async def no_sleep(_):
        return None

    monkeypatch.setattr(base.asyncio, "sleep", no_sleep)
    responses = [httpx.Response(502, json={"error": {"message": "bad gateway"}}),
                 httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})]
    llm = openrouter(lambda r: responses.pop(0))
    assert (await llm.complete("s", "u")).text == "ok"


async def test_5xx_gives_up_after_retries(monkeypatch):
    import bot.llm.base as base

    async def no_sleep(_):
        return None

    monkeypatch.setattr(base.asyncio, "sleep", no_sleep)
    calls = []

    def handler(r):
        calls.append(1)
        return httpx.Response(503, text="overloaded")

    with pytest.raises(LLMTransientError):
        await openrouter(handler).complete("s", "u")
    assert len(calls) == 3


async def test_openrouter_null_content_with_reasoning_is_empty_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "reasoning": "a" * 9000},
                                                      "finish_reason": "stop"}]})

    with pytest.raises(LLMEmptyResponse, match="рассуждения"):
        await openrouter(handler).complete("s", "u")


async def test_anthropic_client():
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "claude-test",
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "ответ"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        })

    llm = AnthropicClient(api_key="ak", model="claude-test", base_url="https://api.anthropic.com",
                          reasoning="low", transport=httpx.MockTransport(handler))
    result = await llm.complete("system text", "user text")
    assert result.text == "ответ"
    assert captured["headers"]["x-api-key"] == "ak"
    assert captured["headers"]["anthropic-version"]
    assert captured["body"]["system"] == "system text"
    assert captured["body"]["thinking"]["type"] == "enabled"
    assert captured["body"]["max_tokens"] > captured["body"]["thinking"]["budget_tokens"]


async def test_anthropic_max_tokens_stop_is_truncation():
    def handler(request):
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{\"a\":"}], "stop_reason": "max_tokens"})

    llm = AnthropicClient(api_key="ak", model="m", base_url="https://api.anthropic.com",
                          transport=httpx.MockTransport(handler))
    with pytest.raises(LLMTruncated):
        await llm.complete("s", "u")


async def test_anthropic_error_message():
    def handler(request):
        return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                      "message": "model: not found"}})

    llm = AnthropicClient(api_key="ak", model="m", base_url="https://api.anthropic.com",
                          transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="model: not found"):
        await llm.complete("s", "u")
