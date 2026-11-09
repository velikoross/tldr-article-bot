"""Промпты, разбор ответов модели и обработка длинных статей по частям."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date

from .extractor import Article
from .languages import prompt_name
from .llm import LLMClient, LLMError, LLMTruncated

log = logging.getLogger(__name__)

VERDICTS = ("read", "skim", "skip")
KINDS = ("practice", "overview", "opinion", "news", "research", "promo", "other")
WARNING_TYPES = ("promo", "outdated", "clickbait", "retelling", "partial", "other")
MAX_CHUNKS = 40

SYSTEM_BASE = (
    "You are a sharp, honest editor who helps a busy reader decide whether an article is worth "
    "their time. You read carefully, never invent facts that are not in the text, and you say "
    "frankly when an article is advertising disguised as content, clickbait, outdated, or a "
    "retelling of someone else's material."
)


class BadModelOutput(LLMError):
    pass


def system_prompt(instructions: str = "") -> str:
    if instructions.strip():
        return (
            SYSTEM_BASE
            + "\n\nAdditional instructions from the bot owner. Always follow them:\n"
            + instructions.strip()
        )
    return SYSTEM_BASE


# --- разбор JSON -------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (с ```-обёрткой, <think> и пояснениями вокруг)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    candidates.append(cleaned)
    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        end = candidate.rfind("}")
        if end > start:
            try:
                data = json.loads(candidate[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        # Ищем первый сбалансированный объект (если после JSON модель дописала текст с фигурными скобками).
        try:
            data, _ = json.JSONDecoder().raw_decode(candidate[start:])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError("в ответе нет JSON-объекта")


def _str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_str(v) for v in value)
    return re.sub(r"\s+", " ", str(value)).strip()


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [line for line in re.split(r"\n+", value)]
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("text") or item.get("point") or " ".join(_str(v) for v in item.values())
        text = _str(item).lstrip("•-*–— ").strip()
        if text:
            items.append(text)
    return items


VERDICT_ALIASES = {
    "read": "read", "читать": "read", "must read": "read", "✅": "read",
    "skim": "skim", "пролистать": "skim", "skim through": "skim", "👀": "skim",
    "skip": "skip", "пропустить": "skip", "можно пропустить": "skip", "⏭": "skip",
}


def parse_summary(text: str) -> dict:
    data = extract_json(text)
    summary = _str_list(data.get("summary") or data.get("points") or data.get("key_points"))
    if not summary:
        raise ValueError("в JSON нет поля summary")
    verdict = VERDICT_ALIASES.get(_str(data.get("verdict")).lower(), "skim")
    kind = _str(data.get("kind")).lower()
    if kind not in KINDS:
        kind = "other"
    warnings = []
    raw_warnings = data.get("warnings") or []
    if isinstance(raw_warnings, (str, dict)):
        raw_warnings = [raw_warnings]
    for item in raw_warnings if isinstance(raw_warnings, list) else []:
        if isinstance(item, dict):
            wtype = _str(item.get("type")).lower()
            wtext = _str(item.get("text") or item.get("reason"))
        else:
            wtype, wtext = "other", _str(item)
        if not wtext and wtype not in WARNING_TYPES:
            continue
        warnings.append({"type": wtype if wtype in WARNING_TYPES else "other", "text": wtext})
    return {
        "title": _str(data.get("title")),
        "lang": _str(data.get("lang")).lower()[:5],
        "verdict": verdict,
        "verdict_reason": _str(data.get("verdict_reason") or data.get("reason")),
        "kind": kind,
        "audience": _str(data.get("audience")),
        "warnings": warnings[:4],
        "summary": summary[:8],
    }


def parse_detailed(text: str) -> dict:
    data = extract_json(text)
    sections = []
    for sec in data.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        points = _str_list(sec.get("points"))
        title = _str(sec.get("title"))
        if title or points:
            sections.append({"title": title, "points": points})
    if not sections:
        raise ValueError("в JSON нет поля sections")
    return {"sections": sections[:12], "conclusion": _str(data.get("conclusion"))}


# --- разбивка текста ----------------------------------------------------------

def split_text(text: str, size: int) -> list[str]:
    """Режет текст на части не длиннее ``size`` по абзацам, затем по предложениям."""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    for para in re.split(r"\n{2,}", text):
        if len(para) <= size:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?…])\s+", para)
        for sentence in sentences:
            while len(sentence) > size:
                pieces.append(sentence[:size])
                sentence = sentence[size:]
            if sentence:
                pieces.append(sentence)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# --- запросы к модели -----------------------------------------------------------

class Summarizer:
    def __init__(self, llm: LLMClient, *, input_budget: int, concurrency: int = 3,
                 max_tokens: int = 8000):
        self.llm = llm
        self.input_budget = input_budget
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_tokens = max_tokens

    async def _complete(self, system: str, user: str) -> str:
        async with self.semaphore:
            result = await self.llm.complete(system, user)
        return result.text

    async def complete_json(self, system: str, user: str, parser):
        """Запрос с разбором JSON; если модель ответила не JSON — один повтор."""
        text = await self._complete(system, user)
        try:
            return parser(text)
        except ValueError as first_error:
            log.warning("model returned invalid JSON (%s), retrying once", first_error)
        retry_user = (
            user
            + "\n\nIMPORTANT: your previous answer was not valid JSON. Reply with ONE valid JSON "
            "object only, no markdown fences, no comments."
        )
        text = await self._complete(system, retry_user)
        try:
            return parser(text)
        except ValueError as exc:
            raise BadModelOutput(
                "Модель дважды ответила не в нужном формате (ожидался JSON). "
                "Попробуйте ещё раз или выберите другую модель."
            ) from exc

    # --- длинные статьи: заметки по частям ---

    async def condense(self, article: Article, instructions: str = "") -> tuple[str, bool]:
        """Возвращает текст, помещающийся в контекст, и флаг «это заметки по частям»."""
        if len(article.text) <= self.input_budget:
            return article.text, False
        if article.notes:
            return article.notes, True
        text = article.text
        level = 0
        while len(text) > self.input_budget:
            level += 1
            chunks = split_text(text, self.input_budget)
            if len(chunks) > MAX_CHUNKS:
                log.warning("article %s too long: %d chunks, truncating", article.url, len(chunks))
                chunks = chunks[:MAX_CHUNKS]
            log.info("condensing %s: level %d, %d chunks", article.url, level, len(chunks))
            notes = await asyncio.gather(
                *(self._chunk_notes(article, chunk, i + 1, len(chunks), instructions)
                  for i, chunk in enumerate(chunks))
            )
            text = "\n\n".join(f"[Part {i + 1}/{len(chunks)}]\n{n}" for i, n in enumerate(notes))
            if level >= 3:
                text = text[: self.input_budget]
                break
        article.notes = text
        return text, True

    async def _chunk_notes(self, article: Article, chunk: str, index: int, total: int, instructions: str) -> str:
        user = (
            f"This is part {index} of {total} of a long text titled \"{article.title or 'untitled'}\".\n"
            "Write dense notes on THIS part: key ideas, facts, numbers, names, examples, "
            "section headings and conclusions, as a bullet list of up to 15 points. Note if the part "
            "is promotional. Write the notes in the same language as the text. No introduction.\n\n"
            f"TEXT:\n<<<\n{chunk}\n>>>"
        )
        return await self._complete(system_prompt(instructions), user)

    # --- основная выжимка ---

    def _meta_block(self, article: Article, is_notes: bool) -> str:
        lines = []
        if article.url.startswith("http"):
            lines.append(f"- URL: {article.url}")
        if article.title:
            lines.append(f"- Title: {article.title}")
        if article.site:
            lines.append(f"- Site: {article.site}")
        if article.author:
            lines.append(f"- Author: {article.author}")
        if article.published:
            lines.append(f"- Published: {article.published}")
        lines.append(f"- Length: ~{article.words} words")
        if article.source == "message":
            lines.append("- Source: the reader pasted this text directly into the chat")
        if article.source == "pdf":
            lines.append("- Source: PDF document")
        if article.partial:
            lines.append("- NOTE: only the beginning of the article is available (paywall).")
        if is_notes:
            lines.append(
                "- NOTE: the article is long; below are notes made from its consecutive parts, "
                "not the original text."
            )
        return "\n".join(lines)

    async def summarize(self, article: Article, *, language: str, profile: str = "",
                        instructions: str = "") -> dict:
        text, is_notes = await self.condense(article, instructions)
        lang = prompt_name(language)
        profile_text = profile.strip() or (
            "not specified — evaluate the article for a general curious, technically literate reader"
        )
        user = f"""Today is {date.today().isoformat()}.
Reader profile: {profile_text}

Article metadata:
{self._meta_block(article, is_notes)}

Return ONE JSON object (no markdown fences) with exactly these keys:
{{
  "title": "article title in its original language; keep the given title if there is one, otherwise make a short one",
  "lang": "ISO 639-1 code of the article's language",
  "verdict": "read" | "skim" | "skip",
  "verdict_reason": "1-2 sentences explaining the verdict for THIS reader",
  "kind": "practice" | "overview" | "opinion" | "news" | "research" | "promo" | "other",
  "audience": "1-2 sentences: who will benefit from the article and what exactly they can take away",
  "warnings": [{{"type": "promo" | "outdated" | "clickbait" | "retelling" | "partial" | "other", "text": "short explanation"}}],
  "summary": ["5-8 points"]
}}

Rules:
- Write every text value in {lang}, regardless of the article's language. Keep code, product names and established terms as they are.
- verdict "read": substantial and relevant to the reader; "skim": a few useful parts, the rest can be skimmed; "skip": little value for this reader, or it's mostly promo, clickbait or a rehash.
- Take the reader profile into account: topics the reader is not interested in lean towards "skip", topics close to their work lean towards "read" if the content is good. If the profile is not specified, judge general quality and usefulness.
- warnings: an empty list if there are no problems. "promo" — mainly advertises a product, service or course; "outdated" — information is clearly obsolete relative to today; "clickbait" — the headline promises more than the text delivers; "retelling" — a translation or rehash of someone else's material without added value; "partial" — only part of the text was available.
- summary: 5-8 points, each one or two concise sentences with concrete ideas, facts, numbers and conclusions. No filler like "the author talks about".
- Plain text in values, no Markdown or HTML.

ARTICLE:
<<<
{text}
>>>"""
        return await self.complete_json(system_prompt(instructions), user, parse_summary)

    async def detailed(self, article: Article, *, language: str, instructions: str = "") -> dict:
        text, is_notes = await self.condense(article, instructions)
        user = f"""Make a detailed section-by-section breakdown of the article.

Article metadata:
{self._meta_block(article, is_notes)}

Return ONE JSON object (no markdown fences):
{{
  "sections": [{{"title": "section name", "points": ["2-6 concrete points: ideas, facts, numbers, examples"]}}],
  "conclusion": "2-3 sentences: the main takeaway and what the reader can apply"
}}

Rules:
- Follow the structure of the article (its own sections if it has them), 3-10 sections.
- Write every text value in {prompt_name(language)}. Keep code and product names as they are.
- Plain text in values, no Markdown or HTML.

ARTICLE:
<<<
{text}
>>>"""
        return await self.complete_json(system_prompt(instructions), user, parse_detailed)

    # --- переводы ---

    async def translate_json(self, data: dict, *, language: str, instructions: str = "") -> dict:
        source = json.dumps(data, ensure_ascii=False, indent=1)
        user = (
            f"Translate every string value in this JSON into {prompt_name(language)}. Keep the keys, "
            "the structure, the values of \"verdict\", \"kind\", \"type\" and \"lang\", numbers, code, URLs "
            "and proper names unchanged. Return ONE JSON object only.\n\n" + source
        )

        def parser(text: str) -> dict:
            result = extract_json(text)
            missing = [k for k in data if k not in result]
            if missing:
                raise ValueError(f"в переводе нет ключей {missing}")
            return result

        system = "You are a professional translator." + (
            "\n\nAdditional instructions from the bot owner. Always follow them:\n" + instructions.strip()
            if instructions.strip() else ""
        )
        return await self.complete_json(system, user, parser)

    def translation_chunk_size(self) -> int:
        # Перевод примерно равен оригиналу по длине: часть должна влезть и во вход, и в ответ.
        return max(2000, min(self.input_budget, int(self.max_tokens * 1.2)))

    async def translate_text(self, text: str, *, language: str, instructions: str = "") -> str:
        chunks = split_text(text, self.translation_chunk_size())
        system = (
            "You are a professional translator of articles. Translate faithfully and naturally, "
            "preserve Markdown formatting (headings, lists, quotes, tables, code blocks). "
            "Do not translate code. Output only the translation, without comments."
        )
        if instructions.strip():
            system += "\n\nAdditional instructions from the bot owner. Always follow them:\n" + instructions.strip()
        parts = await asyncio.gather(*(self._translate_chunk(system, c, language, 0) for c in chunks))
        return "\n\n".join(parts)

    async def _translate_chunk(self, system: str, chunk: str, language: str, depth: int) -> str:
        user = f"Translate into {prompt_name(language)}:\n\n{chunk}"
        try:
            return await self._complete(system, user)
        except LLMTruncated:
            if depth >= 3 or len(chunk) < 1500:
                raise
            halves = split_text(chunk, len(chunk) // 2 + 1)
            parts = [await self._translate_chunk(system, h, language, depth + 1) for h in halves]
            return "\n\n".join(parts)
