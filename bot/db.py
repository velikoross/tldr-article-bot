"""SQLite-хранилище: пользователи, статьи, кэш ответов нейросети, история."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite

from .extractor import Article

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    profile TEXT NOT NULL DEFAULT '',
    lang TEXT,
    tr_lang TEXT,
    full_format TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_norm TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    site TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    published TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'html',
    partial INTEGER NOT NULL DEFAULT 0,
    lang TEXT NOT NULL DEFAULT '',
    notes TEXT,
    fetched_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    article_id INTEGER,
    value TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    verdict TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    day TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS history_user ON history(user_id, created_at);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class UserPrefs:
    id: int
    profile: str
    lang: str
    tr_lang: str
    full_format: str


@dataclass
class HistoryItem:
    title: str
    url: str
    verdict: str
    created_at: int


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Database:
    def __init__(self, path: str, ttl_days: int = 30):
        self.path = path
        self.ttl = ttl_days * 86400 if ttl_days > 0 else 0
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.executescript(SCHEMA)
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()
            self.conn = None

    def _fresh(self, ts: int) -> bool:
        return not self.ttl or time.time() - ts < self.ttl

    # --- пользователи ---

    async def get_user(self, user_id: int, default_lang: str, default_tr: str, default_format: str) -> UserPrefs:
        cur = await self.conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
        row = await cur.fetchone()
        if not row:
            return UserPrefs(user_id, "", default_lang, default_tr, default_format)
        return UserPrefs(
            user_id,
            row["profile"] or "",
            row["lang"] or default_lang,
            row["tr_lang"] or default_tr,
            row["full_format"] or default_format,
        )

    async def update_user(self, user_id: int, **fields) -> None:
        allowed = {"profile", "lang", "tr_lang", "full_format"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        await self.conn.execute(
            "INSERT OR IGNORE INTO users(id, created_at) VALUES(?, ?)", (user_id, int(time.time()))
        )
        for key, value in fields.items():
            await self.conn.execute(f"UPDATE users SET {key}=? WHERE id=?", (value, user_id))
        await self.conn.commit()

    # --- статьи ---

    async def get_article(self, url_norm: str) -> Article | None:
        cur = await self.conn.execute("SELECT * FROM articles WHERE url_norm=?", (url_norm,))
        row = await cur.fetchone()
        if not row or not self._fresh(row["fetched_at"]):
            return None
        return self._article(row)

    async def get_article_by_id(self, article_id: int) -> Article | None:
        cur = await self.conn.execute("SELECT * FROM articles WHERE id=?", (article_id,))
        row = await cur.fetchone()
        return self._article(row) if row else None

    @staticmethod
    def _article(row) -> Article:
        return Article(
            id=row["id"], url=row["url"], url_norm=row["url_norm"], title=row["title"],
            site=row["site"], author=row["author"], published=row["published"], text=row["text"],
            source=row["source"], partial=bool(row["partial"]), notes=row["notes"], lang=row["lang"] or "",
        )

    async def save_article(self, article: Article) -> Article:
        now = int(time.time())
        cur = await self.conn.execute("SELECT id FROM articles WHERE url_norm=?", (article.url_norm,))
        row = await cur.fetchone()
        values = (article.url, article.title, article.site, article.author, article.published,
                  article.text, article.source, int(article.partial), now)
        if row:
            # Статья обновилась (истёк TTL) — старые выжимки и переводы больше не годятся.
            await self.conn.execute(
                "UPDATE articles SET url=?, title=?, site=?, author=?, published=?, text=?, source=?,"
                " partial=?, fetched_at=?, notes=NULL, lang='' WHERE id=?",
                (*values, row["id"]),
            )
            await self.conn.execute("DELETE FROM cache WHERE article_id=?", (row["id"],))
            article.id = row["id"]
        else:
            cur = await self.conn.execute(
                "INSERT INTO articles(url, title, site, author, published, text, source, partial,"
                " fetched_at, url_norm) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (*values, article.url_norm),
            )
            article.id = cur.lastrowid
        article.notes = None
        await self.conn.commit()
        return article

    async def set_article_info(self, article_id: int, *, lang: str = "", title: str = "") -> None:
        if lang:
            await self.conn.execute("UPDATE articles SET lang=? WHERE id=?", (lang, article_id))
        if title:
            await self.conn.execute(
                "UPDATE articles SET title=? WHERE id=? AND title=''", (title, article_id)
            )
        await self.conn.commit()

    async def save_notes(self, article_id: int, notes: str) -> None:
        await self.conn.execute("UPDATE articles SET notes=? WHERE id=?", (notes, article_id))
        await self.conn.commit()

    # --- кэш ответов нейросети ---

    async def cache_get(self, key: str) -> tuple[int, dict] | None:
        cur = await self.conn.execute("SELECT id, value, created_at FROM cache WHERE key=?", (key,))
        row = await cur.fetchone()
        if not row or not self._fresh(row["created_at"]):
            return None
        return row["id"], json.loads(row["value"])

    async def cache_get_by_id(self, cache_id: int) -> tuple[str, int | None, dict] | None:
        cur = await self.conn.execute("SELECT key, article_id, value FROM cache WHERE id=?", (cache_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return row["key"], row["article_id"], json.loads(row["value"])

    async def cache_set(self, key: str, value: dict, article_id: int | None = None) -> int:
        payload = json.dumps(value, ensure_ascii=False)
        now = int(time.time())
        await self.conn.execute(
            "INSERT INTO cache(key, article_id, value, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at, "
            "article_id=excluded.article_id",
            (key, article_id, payload, now),
        )
        await self.conn.commit()
        cur = await self.conn.execute("SELECT id FROM cache WHERE key=?", (key,))
        return (await cur.fetchone())["id"]

    # --- история и лимиты ---

    async def add_history(self, user_id: int, article: Article, verdict: str) -> None:
        await self.conn.execute(
            "INSERT INTO history(user_id, article_id, verdict, title, url, day, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (user_id, article.id, verdict, article.title, article.url if article.url.startswith("http") else "",
             today(), int(time.time())),
        )
        await self.conn.commit()

    async def history(self, user_id: int, limit: int = 20) -> list[HistoryItem]:
        cur = await self.conn.execute(
            "SELECT title, url, verdict, created_at FROM history WHERE user_id=?"
            " ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        )
        return [HistoryItem(r["title"], r["url"], r["verdict"], r["created_at"]) for r in await cur.fetchall()]

    async def articles_today(self, user_id: int) -> set[int]:
        cur = await self.conn.execute(
            "SELECT DISTINCT article_id FROM history WHERE user_id=? AND day=?", (user_id, today())
        )
        return {r["article_id"] for r in await cur.fetchall()}

    # --- произвольные настройки ---

    async def kv_get(self, key: str) -> str | None:
        cur = await self.conn.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None

    async def kv_set(self, key: str, value: str | None) -> None:
        if value is None:
            await self.conn.execute("DELETE FROM kv WHERE key=?", (key,))
        else:
            await self.conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        await self.conn.commit()

    async def stats(self) -> dict:
        result = {}
        for table in ("users", "articles", "cache", "history"):
            cur = await self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
            result[table] = (await cur.fetchone())["n"]
        return result
