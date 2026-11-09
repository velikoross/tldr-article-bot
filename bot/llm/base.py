"""Общий интерфейс клиентов нейросети и ошибки."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("bot.llm")


class LLMError(Exception):
    """Ошибка нейросети; ``str(exc)`` — понятный текст для пользователя."""

    retryable = False


class LLMProviderError(LLMError):
    """Провайдер отклонил запрос (400–404): ключ, модель, политика данных и т. п.

    Повторять такой запрос бессмысленно — показываем текст провайдера как есть.
    """

    def __init__(self, status: int, provider_message: str):
        self.status = status
        self.provider_message = provider_message
        super().__init__(f"Провайдер нейросети отклонил запрос ({status}): {provider_message}")


class LLMTransientError(LLMError):
    """Временная ошибка (429, 5xx, таймаут, сеть) — можно повторить."""

    retryable = True


class LLMEmptyResponse(LLMError):
    def __init__(self, reasoning_used: bool = False):
        hint = (
            " Похоже, модель потратила лимит на рассуждения: увеличьте LLM_MAX_TOKENS "
            "или поставьте LLM_REASONING=off/low."
            if reasoning_used
            else " Попробуйте ещё раз или выберите другую модель."
        )
        super().__init__("Модель вернула пустой ответ." + hint)


class LLMTruncated(LLMError):
    def __init__(self, max_tokens: int, partial: str = ""):
        self.partial = partial
        super().__init__(
            f"Ответ модели обрезан на лимите {max_tokens} токенов. "
            "Увеличьте LLM_MAX_TOKENS или уменьшите LLM_REASONING."
        )


@dataclass
class LLMResult:
    text: str
    finish_reason: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    reasoning_chars: int = 0


_logged_errors: set[str] = set()


def log_once(message: str) -> None:
    """Пишет ошибку провайдера в лог один раз за время работы процесса."""
    key = hashlib.sha1(message.encode("utf-8")).hexdigest()
    if key in _logged_errors:
        return
    _logged_errors.add(key)
    log.error("%s", message)


class LLMClient:
    """Базовый клиент: ретраи временных ошибок и проверка пустых/обрезанных ответов."""

    provider = "base"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        max_tokens: int = 8000,
        reasoning: str = "",
        temperature: float = 0.3,
        timeout: float = 180.0,
        extra_body: dict | None = None,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.retries = retries
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0), transport=transport)
        self._rejected: dict[str, LLMProviderError] = {}

    async def close(self) -> None:
        await self.http.aclose()

    def reset_rejections(self) -> None:
        """Забыть отклонённые запросы (например, после исправления настроек аккаунта)."""
        self._rejected.clear()

    async def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> LLMResult:
        """Один запрос к модели. ``max_tokens`` не может быть меньше LLM_MAX_TOKENS."""
        limit = max(self.max_tokens, max_tokens or 0)
        fingerprint = hashlib.sha1(f"{self.model}\0{system}\0{user}".encode("utf-8")).hexdigest()
        if fingerprint in self._rejected:
            # Тот же запрос уже был отклонён провайдером — не отправляем его снова.
            raise self._rejected[fingerprint]

        attempt = 0
        while True:
            try:
                result = await self._request(system, user, limit)
                break
            except LLMProviderError as exc:
                log_once(f"LLM {self.provider}/{self.model}: {exc}")
                self._rejected[fingerprint] = exc
                if len(self._rejected) > 500:
                    self._rejected.clear()
                raise
            except LLMTransientError as exc:
                attempt += 1
                if attempt > self.retries:
                    raise
                delay = 2 ** attempt
                log.warning("LLM transient error (%s), retry in %ss", exc, delay)
                await asyncio.sleep(delay)

        text = (result.text or "").strip()
        if result.finish_reason in ("length", "max_tokens", "MAX_TOKENS"):
            raise LLMTruncated(limit, text)
        if not text:
            raise LLMEmptyResponse(reasoning_used=result.reasoning_chars > 0 or bool(self.reasoning))
        result.text = text
        return result

    async def _request(self, system: str, user: str, max_tokens: int) -> LLMResult:
        raise NotImplementedError

    # --- общие помощники HTTP ---

    async def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTransientError("Нейросеть не ответила вовремя (таймаут).") from exc
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Нет связи с нейросетью: {exc.__class__.__name__}") from exc

        try:
            data = resp.json()
        except ValueError:
            data = None

        if resp.status_code >= 400:
            message = self._error_message(data) or resp.text[:500] or resp.reason_phrase
            if resp.status_code in (400, 401, 402, 403, 404):
                raise LLMProviderError(resp.status_code, message)
            if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                raise LLMTransientError(f"Нейросеть временно недоступна ({resp.status_code}): {message}")
            raise LLMProviderError(resp.status_code, message)
        if not isinstance(data, dict):
            raise LLMTransientError("Нейросеть вернула не JSON.")
        # OpenRouter иногда присылает ошибку со статусом 200.
        if data.get("error") and not data.get("choices") and not data.get("content"):
            message = self._error_message(data) or "неизвестная ошибка"
            code = data["error"].get("code") if isinstance(data["error"], dict) else None
            if isinstance(code, int) and 400 <= code <= 404:
                raise LLMProviderError(code, message)
            raise LLMTransientError(f"Ошибка нейросети: {message}")
        return data

    @staticmethod
    def _error_message(data) -> str:
        if not isinstance(data, dict):
            return ""
        err = data.get("error")
        if isinstance(err, str):
            return err
        if not isinstance(err, dict):
            return str(data.get("message", ""))
        message = str(err.get("message") or "").strip()
        meta = err.get("metadata") or {}
        if isinstance(meta, dict):
            raw = meta.get("raw")
            provider = meta.get("provider_name")
            if raw:
                raw = raw if isinstance(raw, str) else str(raw)
                if raw not in message:
                    message = f"{message} — {raw[:400]}" if message else raw[:400]
            if provider:
                message = f"{message} (провайдер: {provider})"
        return message
