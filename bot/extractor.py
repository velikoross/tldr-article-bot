"""Извлечение основного текста и метаданных из HTML, PDF и простого текста."""

from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime

import lxml.html
import trafilatura
from lxml import etree

from .urls import site_name

log = logging.getLogger(__name__)

WORDS_PER_MINUTE = 200
WORD_RE = re.compile(r"\w+", re.UNICODE)


class ExtractError(Exception):
    """Не удалось получить текст; ``str(exc)`` — текст для пользователя."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Article:
    url: str
    url_norm: str
    title: str = ""
    site: str = ""
    author: str = ""
    published: str = ""  # YYYY-MM-DD, если удалось понять
    text: str = ""
    source: str = "html"  # html | pdf | text | message
    partial: bool = False  # видна только часть (платный доступ)
    lang: str = ""  # язык статьи (определяет модель при выжимке)
    id: int | None = None
    notes: str | None = field(default=None, repr=False)  # кэш заметок по частям длинной статьи

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(self.text))

    @property
    def reading_minutes(self) -> int:
        return max(1, math.ceil(self.words / WORDS_PER_MINUTE))


# --- правила для сайтов с нестандартной вёрсткой ---------------------------

@dataclass(frozen=True)
class SiteRule:
    content: tuple[str, ...]
    title: tuple[str, ...] = ()
    author: tuple[str, ...] = ()
    date: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()


SITE_RULES: dict[str, SiteRule] = {
    "habr.com": SiteRule(
        content=(
            "//div[@id='post-content-body']",
            "//div[contains(@class,'article-formatted-body')]",
            "//div[contains(@class,'tm-article-body')]",
        ),
        title=("//h1[contains(@class,'tm-title')]", "//h1"),
        author=(
            "//*[contains(@class,'tm-article-snippet__author')]//a[contains(@class,'tm-user-info__username')]",
            "//a[contains(@class,'tm-user-info__username')]",
        ),
        date=(
            "//*[contains(@class,'tm-article-datetime-published')]//time/@datetime",
            "//time/@datetime",
        ),
        drop=(
            "//*[contains(@class,'tm-article-poll')]",
            "//*[contains(@class,'tm-article-presenter__footer')]",
        ),
    ),
    "vc.ru": SiteRule(
        content=(
            "//div[contains(@class,'content--full')]",
            "//article[contains(@class,'content')]",
            "//div[contains(@class,'entry__content')]",
        ),
        title=("//h1",),
        author=("//*[contains(@class,'author__name')]", "//*[contains(@class,'content-header-author__name')]"),
        date=("//time/@datetime",),
        drop=(
            "//*[contains(@class,'comments')]",
            "//*[contains(@class,'content-footer')]",
            "//*[contains(@class,'subsite-card')]",
        ),
    ),
    "medium.com": SiteRule(
        content=("//article//section", "//article"),
        title=("//article//h1", "//h1"),
        author=("//*[@data-testid='authorName']", "//a[@rel='author']"),
        date=("//*[@data-testid='storyPublishDate']",),
        drop=("//*[@data-testid='headerClapButton']", "//*[@aria-label='responses']"),
    ),
}
SITE_RULES["dtf.ru"] = SITE_RULES["vc.ru"]


def rule_for(host: str) -> SiteRule | None:
    host = host.lower()
    for suffix, rule in SITE_RULES.items():
        if host == suffix or host.endswith("." + suffix):
            return rule
    return None


# --- платный доступ ----------------------------------------------------------

PAYWALL_PHRASES = (
    "member-only story",
    "subscribe to continue reading",
    "subscribe to read",
    "subscribe to keep reading",
    "this article is for subscribers",
    "this content is for subscribers",
    "become a member to read",
    "to continue reading, please",
    "only available to subscribers",
    "только для подписчиков",
    "доступно по подписке",
    "доступна по подписке",
    "доступен по подписке",
    "оформите подписку, чтобы",
    "чтобы продолжить чтение",
    "читать полностью по подписке",
    "материал доступен подписчикам",
)


def _paywall_signal(raw_html: str, ld: list[dict]) -> bool:
    for item in ld:
        value = item.get("isAccessibleForFree")
        if value is False or str(value).lower() == "false":
            return True
    low = raw_html.lower()
    if re.search(r"class=\"[^\"]*\b(paywall|piano-paywall|subscription-wall)\b", low):
        return True
    return any(phrase in low for phrase in PAYWALL_PHRASES)


# --- вспомогательное ---------------------------------------------------------

def decode_html(content: bytes, encoding: str | None = None) -> str:
    if encoding:
        try:
            return content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            pass
    head = content[:4096].decode("ascii", errors="ignore")
    match = re.search(r"<meta[^>]+charset=[\"']?([\w-]+)", head, re.IGNORECASE)
    if match:
        try:
            return content.decode(match.group(1))
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            from trafilatura.utils import decode_file

            return decode_file(content)
        except Exception:  # noqa: BLE001
            return content.decode("utf-8", errors="replace")


def _json_ld(tree) -> list[dict]:
    items: list[dict] = []
    for node in tree.xpath("//script[@type='application/ld+json']/text()"):
        try:
            data = json.loads(node)
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, dict):
                items.append(cur)
                if "@graph" in cur:
                    stack.append(cur["@graph"])
    return items


def _ld_author(items: list[dict]) -> str:
    for item in items:
        author = item.get("author")
        if isinstance(author, list) and author:
            author = author[0]
        if isinstance(author, dict):
            author = author.get("name")
        if isinstance(author, str) and author.strip():
            return author.strip()
    return ""


def _ld_value(items: list[dict], *keys: str) -> str:
    for item in items:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _meta(tree, *names: str) -> str:
    for name in names:
        values = tree.xpath(
            f"//meta[@property='{name}' or @name='{name}' or @itemprop='{name}']/@content"
        )
        for value in values:
            if value and value.strip():
                return value.strip()
    return ""


def _first_xpath_text(tree, paths: tuple[str, ...]) -> str:
    for path in paths:
        for node in tree.xpath(path):
            text = node if isinstance(node, str) else node.text_content()
            text = re.sub(r"\s+", " ", text or "").strip()
            if text:
                return text
    return ""


def normalize_date(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        return match.group(0)
    for fmt in ("%d.%m.%Y", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "li", "ul", "ol", "pre",
    "blockquote", "table", "tr", "figure", "figcaption", "dl", "dt", "dd", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "br",
}
SKIP_TAGS = {"script", "style", "noscript", "nav", "aside", "footer", "form", "button",
             "svg", "iframe", "template", "select", "input"}


def html_to_text(element) -> str:
    """Простой перевод фрагмента HTML в markdown-подобный текст."""
    blocks: list[str] = []
    buf: list[str] = []

    def flush(prefix: str = "") -> None:
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(buf)).strip()
        buf.clear()
        if text:
            blocks.append(prefix + text)

    def walk(el, prefix: str = "") -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        tag = tag.lower()
        if tag in SKIP_TAGS or not tag:
            if el.tail:
                buf.append(el.tail)
            return
        if tag == "pre":
            flush()
            blocks.append("```\n" + el.text_content().strip("\n") + "\n```")
            if el.tail:
                buf.append(el.tail)
            return
        child_prefix = prefix
        if tag in BLOCK_TAGS:
            flush(prefix)
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                child_prefix = "#" * int(tag[1]) + " "
            elif tag == "li":
                child_prefix = "- "
            elif tag == "blockquote":
                child_prefix = "> "
        if el.text:
            buf.append(el.text)
        for child in el:
            walk(child, child_prefix)
        if tag in BLOCK_TAGS:
            flush(child_prefix)
        if el.tail:
            buf.append(el.tail)

    walk(element)
    flush()
    return clean_text("\n\n".join(blocks))


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trafilatura(html: str, url: str | None) -> str:
    try:
        result = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            include_links=False,
            include_images=False,
        )
    except Exception as exc:  # noqa: BLE001 — trafilatura иногда падает на кривой разметке
        log.warning("trafilatura failed for %s: %r", url, exc)
        return ""
    return clean_text(result or "")


def _readability(html: str) -> tuple[str, str]:
    try:
        from readability import Document

        doc = Document(html)
        summary = doc.summary(html_partial=True)
        title = doc.short_title() or ""
        if not summary:
            return "", title
        return html_to_text(lxml.html.fromstring(summary)), title
    except Exception as exc:  # noqa: BLE001
        log.warning("readability failed: %r", exc)
        return "", ""


def _from_site_rule(tree, rule: SiteRule, url: str) -> str:
    for path in rule.content:
        nodes = tree.xpath(path)
        if not nodes:
            continue
        node = nodes[0]
        for drop in rule.drop:
            for bad in node.xpath("." + drop if drop.startswith("//") else drop):
                parent = bad.getparent()
                if parent is not None:
                    parent.remove(bad)
        fragment = etree.tostring(node, encoding="unicode", method="html")
        text = _trafilatura(f"<html><body><article>{fragment}</article></body></html>", url)
        own = html_to_text(node)
        # trafilatura иногда выкидывает слишком много внутри уже найденного блока
        if len(text) < len(own) * 0.6:
            text = own
        if text:
            return text
    return ""


def _strip_leading_title(text: str, title: str) -> str:
    if not title:
        return text
    first, _, rest = text.partition("\n")
    if first.lstrip("# ").strip().lower() == title.strip().lower():
        return rest.strip()
    return text


def extract_html(raw_html: str, url: str, url_norm: str, min_chars: int = 300) -> Article:
    site = site_name(url)
    if not raw_html.strip():
        raise ExtractError("empty", f"📭 {site}: страница пустая.")
    try:
        tree = lxml.html.fromstring(raw_html)
    except (etree.ParserError, ValueError) as exc:
        raise ExtractError("parse", f"🧩 {site}: не получилось разобрать страницу.") from exc

    ld = _json_ld(tree)
    rule = rule_for(site)

    meta = None
    try:
        meta = trafilatura.extract_metadata(raw_html, default_url=url)
    except Exception as exc:  # noqa: BLE001
        log.debug("metadata failed: %r", exc)

    title = (rule and _first_xpath_text(tree, rule.title)) or ""
    title = title or (meta.title if meta and meta.title else "") or _meta(tree, "og:title", "twitter:title")
    title = title or _ld_value(ld, "headline", "name")
    author = (rule and _first_xpath_text(tree, rule.author)) or ""
    author = author or _ld_author(ld) or (meta.author if meta and meta.author else "")
    author = author or _meta(tree, "author", "article:author", "twitter:creator")
    date = (rule and _first_xpath_text(tree, rule.date)) or ""
    date = date or _ld_value(ld, "datePublished", "dateCreated")
    date = date or _meta(tree, "article:published_time", "datePublished", "pubdate")
    date = date or (meta.date if meta and meta.date else "")

    candidates: list[str] = []
    if rule:
        candidates.append(_from_site_rule(lxml.html.fromstring(raw_html), rule, url))
    candidates.append(_trafilatura(raw_html, url))
    best = next((c for c in candidates if len(c) >= min_chars), "")
    if not best:
        text, r_title = _readability(raw_html)
        candidates.append(text)
        title = title or r_title
        candidates.append(clean_text(_ld_value(ld, "articleBody")))
        best = max(candidates, key=len)

    title = re.sub(r"\s+", " ", title or "").strip()
    paywall = _paywall_signal(raw_html, ld)
    if len(best) < min_chars:
        if paywall:
            raise ExtractError(
                "paywall",
                f"🔒 {site}: статья за платным доступом, бот видит только начало. "
                "Если у вас есть доступ — пришлите текст статьи сообщением.",
            )
        raise ExtractError(
            "no_text",
            f"📭 {site}: не нашёл на странице текста статьи. Возможно, это не статья, "
            "контент подгружается скриптами или нужен вход. Можно прислать текст сообщением.",
        )

    text = _strip_leading_title(best, title)
    if not title:
        first_line = text.split("\n", 1)[0].lstrip("# ").strip()
        title = first_line[:150] or site
    return Article(
        url=url,
        url_norm=url_norm,
        title=title,
        site=site,
        author=re.sub(r"\s+", " ", author).strip()[:120],
        published=normalize_date(date),
        text=text,
        source="html",
        partial=paywall and len(text) < 4000,
    )


def extract_pdf(content: bytes, url: str, url_norm: str, min_chars: int = 300, max_pages: int = 300) -> Article:
    site = site_name(url)
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ExtractError("pdf_encrypted", f"🔐 {site}: PDF защищён паролем.") from exc
        pages = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — отдельные битые страницы пропускаем
                continue
        info = reader.metadata or {}
        title = str(info.get("/Title") or "").strip()
        author = str(info.get("/Author") or "").strip()
        created = str(info.get("/CreationDate") or "")
    except ExtractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractError("pdf_broken", f"📄 {site}: не удалось прочитать PDF.") from exc

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # переносы слов
    text = clean_text(text)
    if len(text) < min_chars:
        raise ExtractError(
            "pdf_no_text",
            f"📄 {site}: в PDF нет текстового слоя (похоже на скан) — распознавание не поддерживается.",
        )
    date = ""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", created)
    if match:
        date = "-".join(match.groups())
    if not title:
        title = text.split("\n", 1)[0][:150]
    return Article(url=url, url_norm=url_norm, title=title, site=site, author=author,
                   published=date, text=text, source="pdf")


def extract_plain(content: str, url: str, url_norm: str, source: str = "text", min_chars: int = 300) -> Article:
    text = clean_text(content)
    if len(text) < min_chars:
        raise ExtractError("no_text", "📭 Текста слишком мало для выжимки.")
    first_line = text.split("\n", 1)[0].lstrip("# ").strip()
    title = first_line[:120] if len(first_line) <= 150 else ""
    site = site_name(url) if url.startswith("http") else ""
    return Article(url=url, url_norm=url_norm, title=title, site=site, text=text, source=source)
