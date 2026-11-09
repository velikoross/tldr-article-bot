"""OpenRouter, OpenAI и любой OpenAI-совместимый API (DeepSeek, Ollama, vLLM…)."""

from __future__ import annotations

from .base import LLMClient, LLMResult


class OpenAICompatClient(LLMClient):
    provider = "openai"

    def __init__(self, *, openrouter: bool = False, app_title: str = "TLDR Article Bot",
                 app_url: str = "https://github.com/velikoross/tldr-article-bot", **kwargs):
        super().__init__(**kwargs)
        self.openrouter = openrouter
        self.provider = "openrouter" if openrouter else "openai"
        self.app_title = app_title
        self.app_url = app_url

    @property
    def _official_openai(self) -> bool:
        return "api.openai.com" in self.base_url

    def build_body(self, system: str, user: str, max_tokens: int) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._official_openai:
            # Новые модели OpenAI принимают только max_completion_tokens.
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
        level = self.reasoning
        if self.openrouter:
            body["temperature"] = self.temperature
            if level == "off":
                body["reasoning"] = {"enabled": False}
            elif level in ("low", "medium", "high"):
                body["reasoning"] = {"effort": level, "exclude": True}
        else:
            if level in ("low", "medium", "high"):
                # Рассуждающим моделям OpenAI температура не нужна (и часто запрещена).
                body["reasoning_effort"] = level
            else:
                body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    async def _request(self, system: str, user: str, max_tokens: int) -> LLMResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.openrouter:
            headers["HTTP-Referer"] = self.app_url
            headers["X-Title"] = self.app_title
        data = await self._post(f"{self.base_url}/chat/completions", headers,
                                self.build_body(system, user, max_tokens))
        return parse_chat_completion(data)


def parse_chat_completion(data: dict) -> LLMResult:
    choices = data.get("choices") or []
    if not choices:
        return LLMResult(text="", finish_reason="", model=str(data.get("model", "")))
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # некоторые провайдеры отдают части
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    finish = str(choice.get("finish_reason") or choice.get("native_finish_reason") or "")
    return LLMResult(
        text=content or "",
        finish_reason=finish,
        model=str(data.get("model", "")),
        usage=data.get("usage") or {},
        reasoning_chars=len(reasoning) if isinstance(reasoning, str) else 0,
    )
