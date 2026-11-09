"""Anthropic Messages API."""

from __future__ import annotations

from .base import LLMClient, LLMResult

THINKING_BUDGET = {"low": 2048, "medium": 6144, "high": 16384}
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def build_body(self, system: str, user: str, max_tokens: int) -> dict:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        budget = THINKING_BUDGET.get(self.reasoning)
        if budget:
            # Лимит ответа должен быть больше бюджета рассуждений, иначе текста не останется.
            body["max_tokens"] = max(max_tokens, budget + 4000)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    async def _request(self, system: str, user: str, max_tokens: int) -> LLMResult:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = await self._post(f"{self.base_url}/v1/messages", headers,
                                self.build_body(system, user, max_tokens))
        return parse_messages(data)


def parse_messages(data: dict) -> LLMResult:
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    thinking = sum(len(b.get("thinking", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "thinking")
    return LLMResult(
        text=text,
        finish_reason=str(data.get("stop_reason") or ""),
        model=str(data.get("model", "")),
        usage=data.get("usage") or {},
        reasoning_chars=thinking,
    )
