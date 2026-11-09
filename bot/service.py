"""Обработка статьи целиком: ссылка → текст → выжимка, с кэшем и лимитами."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .config import Settings
from .db import Database, UserPrefs
from .extractor import Article, ExtractError, decode_html, extract_html, extract_pdf, extract_plain
from .fetcher import Fetcher
from .formatting import BUILTIN_LABELS, DEFAULT_LABELS, render_detailed, render_summary
from .llm import LLMClient
from .summarizer import Summarizer
from .telegraph import Telegraph, TelegraphError
from .urls import normalize_url, text_key

log = logging.getLogger(__name__)

SUMMARY_TRANSLATABLE = ("title", "verdict_reason", "audience", "warnings", "summary")


class LimitExceeded(Exception):
    pass


class TooLongForTranslation(Exception):
    pass


@dataclass
class SummaryOutcome:
    article: Article
    data: dict
    cache_id: int
    cached: bool
    html: str


@dataclass
class FullTranslation:
    title: str
    language: str
    markdown: str = ""
    filename: str = ""
    telegraph_urls: list[str] | None = None
    note: str = ""


def _hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def guess_language(text: str) -> str:
    sample = text[:5000]
    letters = re.findall(r"[^\W\d_]", sample)
    if not letters:
        return ""
    cyr = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    return "ru" if cyr / len(letters) > 0.4 else "en"


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^\w\-]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return slug[:limit].strip("-") or "article"


class ArticleService:
    def __init__(self, settings: Settings, db: Database, fetcher: Fetcher, llm: LLMClient,
                 telegraph: Telegraph | None = None):
        self.settings = settings
        self.db = db
        self.fetcher = fetcher
        self.llm = llm
        self.telegraph = telegraph
        self.summarizer = Summarizer(
            llm,
            input_budget=settings.input_chars_budget,
            concurrency=settings.llm_concurrency,
            max_tokens=settings.llm_max_tokens,
        )
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def _lock(self, key: str):
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            yield
        if not lock.locked():
            self._locks.pop(key, None)

    # --- настройки ---

    async def prefs(self, user_id: int) -> UserPrefs:
        s = self.settings
        return await self.db.get_user(user_id, s.default_language, s.default_translation_language,
                                      s.full_translation_format)

    async def instructions(self) -> str:
        stored = await self.db.kv_get("llm_instructions")
        return self.settings.llm_instructions if stored is None else stored

    async def set_instructions(self, text: str | None) -> None:
        await self.db.kv_set("llm_instructions", text)

    # --- получение статьи ---

    async def load_article(self, url: str) -> Article:
        norm = normalize_url(url)
        cached = await self.db.get_article(norm)
        if cached:
            return cached
        async with self._lock("fetch:" + norm):
            cached = await self.db.get_article(norm)
            if cached:
                return cached
            result = await self.fetcher.fetch(url)
            min_chars = self.settings.min_article_chars
            if result.content_type == "pdf":
                article = await asyncio.to_thread(extract_pdf, result.content, result.url, norm, min_chars)
            elif result.content_type == "text":
                text = decode_html(result.content, result.encoding)
                article = await asyncio.to_thread(extract_plain, text, result.url, norm, "text", min_chars)
            else:
                html = decode_html(result.content, result.encoding)
                article = await asyncio.to_thread(extract_html, html, result.url, norm, min_chars)
            return await self.db.save_article(article)

    async def article_from_text(self, text: str) -> Article:
        key = text_key(text)
        cached = await self.db.get_article(key)
        if cached:
            return cached
        article = extract_plain(text, "", key, source="message", min_chars=min(self.settings.min_text_length, 200))
        article.title = ""  # заголовок придумает модель
        return await self.db.save_article(article)

    # --- лимиты ---

    async def check_limit(self, user_id: int, is_admin: bool, article_id: int) -> None:
        limit = self.settings.daily_limit_per_user
        if limit <= 0 or is_admin:
            return
        done = await self.db.articles_today(user_id)
        if article_id not in done and len(done) >= limit:
            raise LimitExceeded(
                f"🚦 Дневной лимит — {limit} статей — исчерпан. Повторные ссылки на сегодняшние "
                "статьи по-прежнему работают, новые — завтра."
            )

    # --- выжимка ---

    def summary_key(self, article: Article, prefs: UserPrefs, instructions: str) -> str:
        return f"summary:{article.id}:{prefs.lang}:{_hash(prefs.profile.strip(), instructions.strip())}"

    async def labels_for(self, language: str) -> dict:
        if language in BUILTIN_LABELS:
            return BUILTIN_LABELS[language]
        key = f"labels:{language}"
        cached = await self.db.cache_get(key)
        if cached:
            return cached[1]
        async with self._lock(key):
            cached = await self.db.cache_get(key)
            if cached:
                return cached[1]
            labels = await self.summarizer.translate_json(DEFAULT_LABELS, language=language)
            await self.db.cache_set(key, labels)
            return labels

    async def _save_notes_if_new(self, article: Article, had_notes: bool) -> None:
        if article.notes and not had_notes and article.id:
            await self.db.save_notes(article.id, article.notes)

    async def summarize(self, article: Article, prefs: UserPrefs) -> SummaryOutcome:
        instructions = await self.instructions()
        key = self.summary_key(article, prefs, instructions)
        cached = await self.db.cache_get(key)
        from_cache = cached is not None
        if not cached:
            async with self._lock(key):
                cached = await self.db.cache_get(key)
                from_cache = cached is not None
                if not cached:
                    had_notes = bool(article.notes)
                    data = await self.summarizer.summarize(
                        article, language=prefs.lang, profile=prefs.profile, instructions=instructions
                    )
                    await self._save_notes_if_new(article, had_notes)
                    cache_id = await self.db.cache_set(key, data, article.id)
                    cached = (cache_id, data)
                    await self.db.set_article_info(article.id, lang=data.get("lang", ""),
                                                   title=data.get("title", ""))
                    article.lang = article.lang or data.get("lang", "")
        cache_id, data = cached
        if not article.title and data.get("title"):
            article.title = data["title"]
        html = render_summary(article, data, await self.labels_for(prefs.lang))
        return SummaryOutcome(article, data, cache_id, from_cache, html)

    async def translate_summary(self, cache_id: int, language: str) -> str:
        row = await self.db.cache_get_by_id(cache_id)
        if not row:
            raise KeyError("summary not found")
        key, article_id, data = row
        article = await self.db.get_article_by_id(article_id)
        if not article:
            raise KeyError("article not found")
        tkey = f"tr:{language}:{key}"
        cached = await self.db.cache_get(tkey)
        if cached:
            translated = cached[1]
        else:
            async with self._lock(tkey):
                cached = await self.db.cache_get(tkey)
                if cached:
                    translated = cached[1]
                else:
                    source = {k: data[k] for k in SUMMARY_TRANSLATABLE if k in data}
                    translated = await self.summarizer.translate_json(
                        source, language=language, instructions=await self.instructions()
                    )
                    await self.db.cache_set(tkey, translated, article_id)
        merged = dict(data)
        for k in SUMMARY_TRANSLATABLE:
            if k in translated:
                merged[k] = translated[k]
        # Типы предупреждений берём из оригинала: модель могла их «перевести».
        if isinstance(merged.get("warnings"), list):
            fixed = []
            for i, w in enumerate(merged["warnings"]):
                orig = data.get("warnings", [])
                wtype = orig[i]["type"] if i < len(orig) else "other"
                text = w.get("text", "") if isinstance(w, dict) else str(w)
                fixed.append({"type": wtype, "text": text})
            merged["warnings"] = fixed
        return render_summary(article, merged, await self.labels_for(language))

    async def detailed(self, article_id: int, prefs: UserPrefs) -> list[str]:
        article = await self.db.get_article_by_id(article_id)
        if not article:
            raise KeyError("article not found")
        instructions = await self.instructions()
        key = f"detail:{article_id}:{prefs.lang}:{_hash(instructions.strip())}"
        cached = await self.db.cache_get(key)
        if cached:
            data = cached[1]
        else:
            async with self._lock(key):
                cached = await self.db.cache_get(key)
                if cached:
                    data = cached[1]
                else:
                    had_notes = bool(article.notes)
                    data = await self.summarizer.detailed(article, language=prefs.lang, instructions=instructions)
                    await self._save_notes_if_new(article, had_notes)
                    await self.db.cache_set(key, data, article_id)
        return render_detailed(article, data)

    def full_translation_language(self, article: Article, prefs: UserPrefs) -> str:
        source_lang = (article.lang or guess_language(article.text)).split("-")[0]
        return prefs.tr_lang if source_lang == prefs.lang else prefs.lang

    async def full_translation(self, article_id: int, prefs: UserPrefs) -> FullTranslation:
        article = await self.db.get_article_by_id(article_id)
        if not article:
            raise KeyError("article not found")
        if len(article.text) > self.settings.full_translation_max_chars:
            raise TooLongForTranslation(
                f"📚 Статья слишком длинная для полного перевода: {len(article.text):,} символов "
                f"при лимите {self.settings.full_translation_max_chars:,} (FULL_TRANSLATION_MAX_CHARS)."
                .replace(",", " ")
            )
        language = self.full_translation_language(article, prefs)
        instructions = await self.instructions()
        key = f"full:{article_id}:{language}:{_hash(instructions.strip())}"
        cached = await self.db.cache_get(key)
        if cached:
            markdown = cached[1]["markdown"]
        else:
            async with self._lock(key):
                cached = await self.db.cache_get(key)
                if cached:
                    markdown = cached[1]["markdown"]
                else:
                    markdown = await self.summarizer.translate_text(
                        article.text, language=language, instructions=instructions
                    )
                    await self.db.cache_set(key, {"markdown": markdown}, article_id)

        title = article.title or "Статья"
        header = f"# {title}\n\n"
        if article.url.startswith("http"):
            header += f"Оригинал: {article.url}\n\n"
        result = FullTranslation(
            title=title,
            language=language,
            markdown=header + markdown + "\n",
            filename=f"{slugify(title)}.{language}.md",
        )
        if prefs.full_format == "telegraph" and self.telegraph:
            tg_key = f"telegraph:{key}"
            cached = await self.db.cache_get(tg_key)
            if cached:
                result.telegraph_urls = cached[1]["urls"]
            else:
                try:
                    urls = await self.telegraph.publish(title, markdown, article.url if article.url.startswith("http") else "")
                    await self.db.cache_set(tg_key, {"urls": urls}, article_id)
                    result.telegraph_urls = urls
                except TelegraphError as exc:
                    log.warning("telegraph publish failed: %s", exc)
                    result.note = f"Не получилось опубликовать на telegra.ph ({exc}), отправляю файлом."
        return result


__all__ = [
    "ArticleService", "SummaryOutcome", "FullTranslation", "LimitExceeded", "TooLongForTranslation",
    "ExtractError", "guess_language",
]
