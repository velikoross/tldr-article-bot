"""HTML-оформление ответов и подсчёт длины сообщения так, как это делает Telegram."""

from __future__ import annotations

import html
import re
from datetime import datetime

from .extractor import Article

TELEGRAM_LIMIT = 4096

VERDICT_LABELS = {
    "read": ("✅", "Читать"),
    "skim": ("👀", "Пролистать"),
    "skip": ("⏭", "Можно пропустить"),
}
KIND_LABELS = {
    "practice": "Практика",
    "overview": "Обзор",
    "opinion": "Мнение",
    "news": "Новость",
    "research": "Исследование",
    "promo": "Реклама продукта",
    "other": "Статья",
}
WARNING_LABELS = {
    "promo": "Реклама",
    "outdated": "Устарело",
    "clickbait": "Кликбейт",
    "retelling": "Пересказ",
    "partial": "Неполный текст",
    "other": "Важно",
}
# Подписи разделов; при переводе выжимки переводятся вместе с ней.
DEFAULT_LABELS = {
    "audience": "Кому и зачем",
    "summary": "Выжимка",
    "warnings": "Обратите внимание",
    "minutes": "мин",
    "partial": "Бот видит только часть статьи (платный доступ) — выжимка неполная.",
    "verdict_names": dict((k, v[1]) for k, v in VERDICT_LABELS.items()),
    "kind_names": dict(KIND_LABELS),
    "warning_names": dict(WARNING_LABELS),
}

EN_LABELS = {
    "audience": "Who it's for",
    "summary": "Summary",
    "warnings": "Heads-up",
    "minutes": "min",
    "partial": "The bot sees only part of the article (paywall), so the summary is incomplete.",
    "verdict_names": {"read": "Read", "skim": "Skim", "skip": "Skip it"},
    "kind_names": {
        "practice": "Hands-on", "overview": "Overview", "opinion": "Opinion", "news": "News",
        "research": "Research", "promo": "Product promo", "other": "Article",
    },
    "warning_names": {
        "promo": "Promo", "outdated": "Outdated", "clickbait": "Clickbait", "retelling": "Rehash",
        "partial": "Incomplete text", "other": "Note",
    },
}
BUILTIN_LABELS = {"ru": DEFAULT_LABELS, "en": EN_LABELS}


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def esc_attr(text: str) -> str:
    return html.escape(text or "", quote=True)


_TAG_RE = re.compile(r"<[^>]+>")


def visible_length(html_text: str) -> int:
    """Длина видимого текста в UTF-16 — именно её Telegram сравнивает с лимитом 4096."""
    plain = html.unescape(_TAG_RE.sub("", html_text))
    return len(plain.encode("utf-16-le")) // 2


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 1, 1)].rstrip()
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:—-") + "…"


def format_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        return value or ""


def _label(labels: dict, group: str, key: str, fallback: dict) -> str:
    value = labels.get(group)
    if isinstance(value, dict) and value.get(key):
        return str(value[key])
    return fallback.get(key, key)


def header_lines(article: Article, title: str, minutes_label: str = "мин") -> list[str]:
    title = _cut(title or article.title or article.site or "Без названия", 200)
    if article.url.startswith("http"):
        first = f'📰 <b><a href="{esc_attr(article.url)}">{esc(title)}</a></b>'
    else:
        first = f"📰 <b>{esc(title)}</b>"
    meta = []
    if article.site:
        meta.append(f"🌐 {esc(article.site)}")
    if article.author:
        meta.append(f"✍️ {esc(_cut(article.author, 60))}")
    if article.published:
        meta.append(f"📅 {esc(format_date(article.published))}")
    meta.append(f"⏱ ~{article.reading_minutes} {esc(minutes_label)}")
    if article.source == "pdf":
        meta.append("📄 PDF")
    return [first, " · ".join(meta)]


def render_summary(article: Article, data: dict, labels: dict | None = None,
                   limit: int = TELEGRAM_LIMIT) -> str:
    """Основной ответ. Гарантированно помещается в одно сообщение."""
    labels = labels or DEFAULT_LABELS
    point_limit, text_limit = 600, 700
    points = list(data.get("summary") or [])
    while True:
        result = _render_summary(article, data, labels, points, point_limit, text_limit)
        if visible_length(result) <= limit:
            return result
        # Сначала укорачиваем пункты, потом убираем последние, потом сокращаем остальное.
        if point_limit > 160:
            point_limit -= 60
        elif len(points) > 3:
            points = points[:-1]
        elif text_limit > 150:
            text_limit -= 100
            point_limit = max(point_limit - 20, 60)
        else:
            point_limit = max(point_limit // 2, 20)
            if point_limit == 20:
                return result if visible_length(result) <= limit else _hard_trim(result, limit)


def _hard_trim(result: str, limit: int) -> str:
    lines = result.split("\n")
    while lines and visible_length("\n".join(lines)) > limit - 1:
        lines.pop()
    return "\n".join(lines) + "\n…"


def _render_summary(article: Article, data: dict, labels: dict, points: list[str],
                    point_limit: int, text_limit: int) -> str:
    verdict = data.get("verdict", "skim")
    emoji = VERDICT_LABELS.get(verdict, VERDICT_LABELS["skim"])[0]
    verdict_name = _label(labels, "verdict_names", verdict, DEFAULT_LABELS["verdict_names"])
    lines = header_lines(article, data.get("title") or article.title, str(labels.get("minutes") or "мин"))
    lines.append("")
    reason = _cut(data.get("verdict_reason", ""), text_limit)
    lines.append(f"{emoji} <b>{esc(verdict_name)}</b>" + (f" — {esc(reason)}" if reason else ""))

    audience = _cut(data.get("audience", ""), text_limit)
    kind = data.get("kind", "other")
    kind_name = _label(labels, "kind_names", kind, DEFAULT_LABELS["kind_names"])
    lines.append("")
    lines.append(f"🎯 <b>{esc(str(labels.get('audience') or DEFAULT_LABELS['audience']))}</b>")
    lines.append(f"<i>{esc(kind_name)}</i>" + (f" · {esc(audience)}" if audience else ""))

    warnings = [w for w in data.get("warnings") or [] if isinstance(w, dict)]
    if article.partial and not any(w.get("type") == "partial" for w in warnings):
        warnings.append({"type": "partial", "text": str(labels.get("partial") or DEFAULT_LABELS["partial"])})
    if warnings:
        lines.append("")
        lines.append(f"⚠️ <b>{esc(str(labels.get('warnings') or DEFAULT_LABELS['warnings']))}</b>")
        for w in warnings:
            name = _label(labels, "warning_names", w.get("type", "other"), DEFAULT_LABELS["warning_names"])
            text = _cut(w.get("text", ""), text_limit)
            lines.append(f"• <b>{esc(name)}</b>" + (f": {esc(text)}" if text else ""))

    lines.append("")
    lines.append(f"📝 <b>{esc(str(labels.get('summary') or DEFAULT_LABELS['summary']))}</b>")
    for point in points:
        lines.append(f"• {esc(_cut(point, point_limit))}")
    return "\n".join(lines)


def render_detailed(article: Article, data: dict) -> list[str]:
    """Подробная выжимка; может занять несколько сообщений."""
    lines = header_lines(article, article.title)
    lines.append("")
    lines.append("🔁 <b>Подробный разбор</b>")
    for sec in data.get("sections", []):
        lines.append("")
        if sec.get("title"):
            lines.append(f"<b>{esc(_cut(sec['title'], 300))}</b>")
        for point in sec.get("points", []):
            lines.append(f"• {esc(_cut(point, 1500))}")
    if data.get("conclusion"):
        lines.append("")
        lines.append(f"💡 <b>Итог:</b> {esc(_cut(data['conclusion'], 2000))}")
    return split_lines(lines)


def split_lines(lines: list[str], limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Собирает строки (каждая — законченный HTML) в сообщения не длиннее лимита."""
    messages: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join(current + [line])
        if current and visible_length(candidate) > limit:
            messages.append("\n".join(current).strip("\n"))
            current = [line] if line else []
        else:
            current.append(line)
    if current:
        messages.append("\n".join(current).strip("\n"))
    return [m for m in messages if m.strip()]


def render_history(items) -> str:
    if not items:
        return "История пуста — пришлите ссылку на статью."
    lines = ["🗂 <b>Последние статьи</b>", ""]
    for i, item in enumerate(items, 1):
        emoji = VERDICT_LABELS.get(item.verdict, ("•", ""))[0]
        title = esc(_cut(item.title or item.url or "Текст из сообщения", 90))
        when = datetime.fromtimestamp(item.created_at).strftime("%d.%m")
        if item.url:
            lines.append(f'{i}. {emoji} <a href="{esc_attr(item.url)}">{title}</a> · {when}')
        else:
            lines.append(f"{i}. {emoji} {title} · {when}")
    return "\n".join(split_lines(lines)[0:1])
