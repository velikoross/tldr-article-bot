"""Точка входа: python -m bot"""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeChat, BotCommandScopeDefault
from dotenv import load_dotenv

from .access import AccessMiddleware
from .config import Settings
from .db import Database
from .fetcher import Fetcher
from .handlers import router
from .llm import create_client
from .service import ArticleService
from .telegraph import Telegraph

log = logging.getLogger("bot")

USER_COMMANDS = [
    BotCommand(command="profile", description="Мои интересы для вердикта"),
    BotCommand(command="settings", description="Языки и формат перевода"),
    BotCommand(command="history", description="Последние 20 статей"),
    BotCommand(command="help", description="Справка"),
]
ADMIN_COMMANDS = USER_COMMANDS + [
    BotCommand(command="llmtest", description="Проверить нейросеть"),
    BotCommand(command="instructions", description="Указания для всех промптов"),
    BotCommand(command="stats", description="Статистика"),
]


async def set_commands(bot: Bot, settings: Settings) -> None:
    try:
        await bot.set_my_commands(USER_COMMANDS, scope=BotCommandScopeDefault())
        for admin_id in settings.admin_ids:
            await bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=admin_id))
    except Exception as exc:  # noqa: BLE001 — меню команд не критично
        log.warning("could not set bot commands: %s", exc)


async def main() -> None:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Ошибка настроек: {exc}", file=sys.stderr)
        sys.exit(2)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not settings.bot_token:
        log.error("BOT_TOKEN не задан — заполните .env")
        sys.exit(2)
    if not settings.llm_api_key and settings.llm_provider in ("openrouter", "anthropic"):
        log.warning("LLM_API_KEY не задан — запросы к нейросети будут отклонены")
    if not settings.admin_ids and not settings.allowed_ids:
        log.warning("ADMIN_IDS и ALLOWED_IDS пусты: бот никого не пустит, но покажет Telegram ID")

    db = Database(settings.db_path, settings.cache_ttl_days)
    await db.connect()
    fetcher = Fetcher(settings.fetch_timeout, settings.fetch_max_bytes, settings.fetch_user_agent)
    llm = create_client(settings)
    telegraph = Telegraph(db)
    service = ArticleService(settings, db, fetcher, llm, telegraph)

    bot = Bot(
        settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML, link_preview_is_disabled=True),
    )
    dp = Dispatcher(settings=settings, service=service)
    access = AccessMiddleware(settings)
    dp.message.outer_middleware(access)
    dp.callback_query.outer_middleware(access)
    dp.include_router(router)

    await set_commands(bot, settings)
    log.info("bot started: provider=%s model=%s", settings.llm_provider, settings.llm_model)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await fetcher.close()
        await llm.close()
        await telegraph.close()
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
