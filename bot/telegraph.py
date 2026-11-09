"""Публикация полного перевода на telegra.ph."""

from __future__ import annotations

import json
import re

import httpx

API = "https://api.telegra.ph"
MAX_PAGE_BYTES = 60_000  # лимит Telegraph — 64 КБ на содержимое страницы


class TelegraphError(Exception):
    pass


_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\(https?://[^)\s]+\)|\*[^*\s][^*]*\*)")


def _inline(text: str) -> list:
    nodes: list = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > pos:
            nodes.append(text[pos : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            nodes.append({"tag": "b", "children": [token[2:-2]]})
        elif token.startswith("`"):
            nodes.append({"tag": "code", "children": [token[1:-1]]})
        elif token.startswith("["):
            label, url = re.match(r"\[([^\]]+)\]\(([^)\s]+)\)", token).groups()
            nodes.append({"tag": "a", "attrs": {"href": url}, "children": [label]})
        else:
            nodes.append({"tag": "i", "children": [token[1:-1]]})
        pos = match.end()
    if pos < len(text):
        nodes.append(text[pos:])
    return nodes or [""]


def markdown_to_nodes(markdown: str) -> list:
    """Упрощённый Markdown → узлы Telegraph (заголовки, списки, цитаты, код, абзацы)."""
    nodes: list = []
    lines = markdown.replace("\r\n", "\n").split("\n")
    i = 0
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            nodes.append({"tag": "p", "children": _inline(" ".join(paragraph).strip())})
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            nodes.append({"tag": "pre", "children": ["\n".join(code)]})
            i += 1
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading:
            flush_paragraph()
            tag = "h3" if len(heading.group(1)) <= 2 else "h4"
            nodes.append({"tag": tag, "children": _inline(heading.group(2))})
            i += 1
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            flush_paragraph()
            ordered = bool(re.match(r"^\d+[.)]", stripped))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+[.)])\s+", lines[i]):
                item = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", lines[i])
                items.append({"tag": "li", "children": _inline(item)})
                i += 1
            nodes.append({"tag": "ol" if ordered else "ul", "children": items})
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            nodes.append({"tag": "blockquote", "children": _inline(" ".join(quote))})
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph()
    return nodes


def split_nodes(nodes: list, max_bytes: int = MAX_PAGE_BYTES) -> list[list]:
    pages: list[list] = [[]]
    size = 0
    for node in nodes:
        node_size = len(json.dumps(node, ensure_ascii=False).encode("utf-8"))
        if pages[-1] and size + node_size > max_bytes:
            pages.append([])
            size = 0
        pages[-1].append(node)
        size += node_size
    return pages


class Telegraph:
    def __init__(self, db, transport: httpx.AsyncBaseTransport | None = None):
        self.db = db
        self.http = httpx.AsyncClient(timeout=30, transport=transport)

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, **params) -> dict:
        try:
            resp = await self.http.post(f"{API}/{method}", data=params)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TelegraphError(f"telegra.ph недоступен: {exc.__class__.__name__}") from exc
        if not data.get("ok"):
            raise TelegraphError(f"telegra.ph: {data.get('error', 'ошибка')}")
        return data["result"]

    async def _token(self) -> str:
        token = await self.db.kv_get("telegraph_token")
        if token:
            return token
        result = await self._call("createAccount", short_name="TLDR bot", author_name="TLDR bot")
        token = result["access_token"]
        await self.db.kv_set("telegraph_token", token)
        return token

    async def publish(self, title: str, markdown: str, source_url: str = "") -> list[str]:
        token = await self._token()
        nodes = markdown_to_nodes(markdown)
        if source_url:
            nodes.insert(0, {"tag": "p", "children": [
                {"tag": "i", "children": ["Оригинал: "]},
                {"tag": "a", "attrs": {"href": source_url}, "children": [source_url]},
            ]})
        pages = split_nodes(nodes)
        urls = []
        title = (title or "Перевод")[:200]
        for index, page in enumerate(pages, 1):
            page_title = title if len(pages) == 1 else f"{title} ({index}/{len(pages)})"[:256]
            result = await self._call(
                "createPage",
                access_token=token,
                title=page_title,
                author_name="TLDR bot",
                content=json.dumps(page, ensure_ascii=False),
                return_content="false",
            )
            urls.append(result["url"])
        return urls
