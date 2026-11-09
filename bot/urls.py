"""Поиск ссылок в сообщениях и нормализация URL для кэша."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'«»]+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}…»\"'"

# Параметры, которые не влияют на содержимое страницы.
TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "gbraid", "wbraid", "yclid", "ysclid", "msclkid",
    "mc_cid", "mc_eid", "igshid", "igsh", "_hsenc", "_hsmi", "_openstat", "ref_src",
    "ref_url", "spm", "si", "share", "rss", "from_rss", "utm",
}
TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "hmb_", "vero_", "oly_")
# Параметры, которые мусорят только на конкретных сайтах.
HOST_TRACKING = {
    "medium.com": {"source", "sk", "postpublishedtype"},
    "habr.com": {"from", "hl"},
    "vc.ru": {"from"},
}
# Ссылки, которые не являются статьями (служебные ссылки Telegram в пересланных постах).
IGNORED_HOSTS = {"t.me", "telegram.me", "telegram.org", "telegram.dog"}


def _clean_candidate(url: str) -> str:
    url = url.rstrip(_TRAILING)
    # Закрывающая скобка — часть URL, только если есть парная открывающая (как в Википедии).
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    if url.lower().startswith("www."):
        url = "https://" + url
    return url


def is_article_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in IGNORED_HOSTS:
        return False
    return "." in host


def find_urls(text: str | None, entities: list | None = None, extra: list[str] | None = None) -> list[str]:
    """Возвращает уникальные ссылки из текста, entities Telegram и кнопок (в порядке появления)."""
    found: list[str] = []
    text = text or ""
    for ent in entities or []:
        ent_type = getattr(ent, "type", None)
        ent_type = getattr(ent_type, "value", ent_type)
        if ent_type == "text_link" and getattr(ent, "url", None):
            found.append(ent.url)
        elif ent_type == "url":
            value = _utf16_slice(text, ent.offset, ent.length)
            # Telegram распознаёт и ссылки без схемы: habr.com/ru/articles/1
            if not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
                value = "https://" + value
            found.append(value)
    found.extend(m.group(0) for m in URL_RE.finditer(text))
    found.extend(extra or [])

    result: list[str] = []
    seen: set[str] = set()
    for raw in found:
        url = _clean_candidate(raw.strip())
        if not is_article_url(url):
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        result.append(url)
    return result


def _utf16_slice(text: str, offset: int, length: int) -> str:
    encoded = text.encode("utf-16-le")
    return encoded[offset * 2 : (offset + length) * 2].decode("utf-16-le", errors="ignore")


def strip_fragment(url: str) -> str:
    """URL для скачивания: только без якоря (параметры могут быть нужны сайту)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def normalize_url(url: str) -> str:
    """Ключ кэша: без схемы-регистра, www, якоря, utm-меток и прочих трекеров."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and host.count(".") >= 2:
        host = host[2:]  # мобильные версии: m.habr.com → habr.com
    port = parts.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    host_params = set()
    for suffix, params in HOST_TRACKING.items():
        if host == suffix or host.endswith("." + suffix):
            host_params |= params
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        low = key.lower()
        if low in TRACKING_PARAMS or low in host_params or low.startswith(TRACKING_PREFIXES):
            continue
        query.append((key, value))
    query.sort()

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit(("https" if scheme in ("http", "https") else scheme, netloc, path, urlencode(query), ""))


def text_key(text: str) -> str:
    """Ключ кэша для присланного текста без ссылки."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return "text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def site_name(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host
