from __future__ import annotations

from ..config import Settings
from .anthropic import AnthropicClient
from .base import (
    LLMClient,
    LLMEmptyResponse,
    LLMError,
    LLMProviderError,
    LLMResult,
    LLMTransientError,
    LLMTruncated,
)
from .openai_compat import OpenAICompatClient

__all__ = [
    "LLMClient", "LLMError", "LLMProviderError", "LLMTransientError", "LLMEmptyResponse",
    "LLMTruncated", "LLMResult", "create_client",
]


def create_client(settings: Settings, transport=None) -> LLMClient:
    common = dict(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.base_url,
        max_tokens=settings.llm_max_tokens,
        reasoning=settings.llm_reasoning,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        extra_body=settings.llm_extra_body,
        transport=transport,
    )
    if settings.llm_provider == "anthropic":
        return AnthropicClient(**common)
    return OpenAICompatClient(openrouter=settings.llm_provider == "openrouter", **common)
