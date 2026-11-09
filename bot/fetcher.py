"""Скачивание страниц: лимит размера, таймауты и понятные ошибки."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .urls import site_name, strip_fragment

log = logging.getLogger(__name__)

HTML_TYPES = ("text/html", "application/xhtml+xml", "application/xml", "text/xml")
TEXT_TYPES = ("text/plain", "text/markdown")
PDF_TYPES = ("application/pdf", "application/x-pdf")


class FetchError(Exception):
    """Ошибка скачивания; ``str(exc)`` — текст для пользователя."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class FetchResult:
    url: str  # итоговый URL после редиректов
    content: bytes
    content_type: str  # html | pdf | text
    encoding: str | None
    status: int


def _kind_of(content_type: str, url: str, head: bytes) -> str | None:
    ct = content_type.split(";")[0].strip().lower()
    if ct in PDF_TYPES or head.startswith(b"%PDF-"):
        return "pdf"
    if ct in HTML_TYPES or not ct:
        return "html"
    if ct in TEXT_TYPES:
        return "text"
    if ct == "application/octet-stream":
        if url.lower().split("?")[0].endswith(".pdf"):
            return "pdf"
        sniff = head[:512].lstrip().lower()
        if sniff.startswith((b"<!doctype html", b"<html")):
            return "html"
    return None


def http_error_message(status: int, site: str) -> tuple[str, str]:
    if status in (401, 402):
        return "paywall", f"🔒 {site}: страница требует авторизации или подписки ({status})."
    if status == 403:
        return "forbidden", (
            f"⛔ {site}: сайт не пустил бота (403). Часто это защита от ботов "
            "или доступ только из определённых стран. Пришлите текст статьи сообщением."
        )
    if status == 404 or status == 410:
        return "not_found", f"🤷 {site}: страница не найдена ({status}). Проверьте ссылку."
    if status == 429:
        return "rate_limited", f"🐢 {site}: сайт просит подождать (429). Попробуйте позже."
    if status >= 500:
        return "server_error", f"💥 {site}: сайт сейчас не отвечает нормально ({status}). Попробуйте позже."
    return "http_error", f"⚠️ {site}: сайт вернул ошибку {status}."


class Fetcher:
    def __init__(self, timeout: float, max_bytes: int, user_agent: str,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.max_bytes = max_bytes
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            follow_redirects=True,
            max_redirects=10,
            transport=transport,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch(self, url: str) -> FetchResult:
        url = strip_fragment(url)
        site = site_name(url)
        limit_mb = self.max_bytes / 1024 / 1024
        try:
            async with self.client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    kind, message = http_error_message(resp.status_code, site)
                    raise FetchError(kind, message)
                length = resp.headers.get("content-length")
                if length and length.isdigit() and int(length) > self.max_bytes:
                    raise FetchError("too_large", f"📦 {site}: файл больше {limit_mb:g} МБ — не скачиваю.")
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FetchError("too_large", f"📦 {site}: страница больше {limit_mb:g} МБ — не скачиваю.")
                    chunks.append(chunk)
                content = b"".join(chunks)
                final_url = str(resp.url)
                kind = _kind_of(resp.headers.get("content-type", ""), final_url, content[:1024])
                if kind is None:
                    ct = resp.headers.get("content-type", "неизвестный тип").split(";")[0]
                    raise FetchError("unsupported", f"🗂 {site}: это не статья, а файл типа {ct}.")
                return FetchResult(final_url, content, kind, resp.charset_encoding, resp.status_code)
        except FetchError:
            raise
        except httpx.TimeoutException as exc:
            raise FetchError("timeout", f"⌛ {site}: сайт слишком долго отвечает. Попробуйте позже.") from exc
        except httpx.TooManyRedirects as exc:
            raise FetchError("redirects", f"🔄 {site}: слишком много перенаправлений.") from exc
        except httpx.InvalidURL as exc:
            raise FetchError("bad_url", f"🔗 Неверная ссылка: {url}") from exc
        except httpx.HTTPError as exc:
            log.info("fetch failed for %s: %r", url, exc)
            raise FetchError("network", f"🔌 {site}: не удалось подключиться к сайту.") from exc
