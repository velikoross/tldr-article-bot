"""Доступ к боту: админы, друзья и все остальные."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from .config import Settings

log = logging.getLogger(__name__)


def role_of(user_id: int | None, settings: Settings) -> str | None:
    if user_id is None:
        return None
    if user_id in settings.admin_ids:
        return "admin"
    if user_id in settings.allowed_ids:
        return "friend"
    return None


def stranger_text(user_id: int) -> str:
    return (
        "🔒 Это приватный бот.\n"
        f"Ваш Telegram ID: <code>{user_id}</code>\n"
        "Если вы знакомы с владельцем, отправьте ему этот номер."
    )


class AccessMiddleware(BaseMiddleware):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        role = role_of(user.id if user else None, self.settings)
        if role is None:
            if user is None:
                return None
            log.info("access denied for user %s (@%s)", user.id, getattr(user, "username", None))
            if isinstance(event, CallbackQuery):
                await event.answer("🔒 Это приватный бот.", show_alert=True)
            elif isinstance(event, Message):
                await event.answer(stranger_text(user.id))
            return None
        data["role"] = role
        data["is_admin"] = role == "admin"
        return await handler(event, data)
