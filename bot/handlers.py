"""Команды и сообщения Telegram."""

from __future__ import annotations

import logging
import time

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.keyboard import InlineKeyboardBuilder

from .config import Settings
from .extractor import ExtractError
from .fetcher import FetchError
from .formatting import esc, esc_attr, render_history, visible_length
from .languages import LANGUAGES, menu_name
from .llm import LLMError, LLMProviderError
from .service import ArticleService, LimitExceeded, SummaryOutcome, TooLongForTranslation
from .urls import find_urls

log = logging.getLogger(__name__)
router = Router(name="main")

HELP = """👋 Пришлите ссылку на статью — я прочитаю её, сделаю выжимку и скажу, стоит ли тратить на неё время.

Что умею:
• несколько ссылок в одном сообщении — разберу каждую;
• пересланный пост из канала со ссылкой;
• длинный текст без ссылки — сделаю выжимку текста;
• PDF по ссылке (если в нём есть текстовый слой).

Под ответом: 🌐 перевод выжимки, 📄 полный перевод статьи, 🔁 подробный разбор по разделам.

Команды:
/profile — чем вы занимаетесь и что вам интересно (влияет на вердикт)
/settings — язык выжимки, второй язык, формат полного перевода
/history — последние 20 статей
/help — эта справка"""

ADMIN_HELP = """

Для админа:
/llmtest — проверить подключение к нейросети
/instructions — указания, добавляемые ко всем промптам
/stats — статистика"""


class ProfileForm(StatesGroup):
    waiting = State()


_in_progress: set[tuple[int, str]] = set()


def result_keyboard(outcome: SummaryOutcome) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🌐 Перевести", callback_data=f"tr:{outcome.cache_id}")
    kb.button(text="📄 Полный перевод", callback_data=f"full:{outcome.article.id}")
    kb.button(text="🔁 Подробнее", callback_data=f"more:{outcome.article.id}")
    kb.adjust(2, 1)
    return kb.as_markup()


def error_text(exc: Exception) -> str:
    if isinstance(exc, (FetchError, ExtractError, LimitExceeded, TooLongForTranslation)):
        return esc(str(exc))
    if isinstance(exc, LLMProviderError):
        return (
            f"🤖 {esc(str(exc))}\n\n"
            "Исправьте настройки модели или аккаунта провайдера и проверьте командой /llmtest."
        )
    if isinstance(exc, LLMError):
        return f"🤖 {esc(str(exc))}"
    return "😵 Что-то пошло не так. Подробности в логе бота."


async def safe_edit(message: Message, text: str, reply_markup=None) -> Message:
    try:
        result = await message.edit_text(text, reply_markup=reply_markup)
        return result if isinstance(result, Message) else message
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc):
            return message
        log.warning("edit failed (%s), sending new message", exc)
        return await message.answer(text, reply_markup=reply_markup)


def _message_urls(message: Message) -> list[str]:
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []
    extra: list[str] = []
    markup = message.reply_markup
    if markup and getattr(markup, "inline_keyboard", None):
        for row in markup.inline_keyboard:
            extra.extend(button.url for button in row if getattr(button, "url", None))
    preview = message.link_preview_options
    if preview and preview.url and not preview.is_disabled:
        extra.append(preview.url)
    return find_urls(text, entities, extra)


# --- статьи ----------------------------------------------------------------

async def _process(message: Message, bot: Bot, service: ArticleService, is_admin: bool,
                   *, url: str | None = None, text: str | None = None) -> None:
    user_id = message.from_user.id
    status = await message.reply("⏳ Читаю статью…" if url else "⏳ Читаю текст…")
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            article = await service.load_article(url) if url else await service.article_from_text(text)
            await service.check_limit(user_id, is_admin, article.id)
            prefs = await service.prefs(user_id)
            outcome = await service.summarize(article, prefs)
        await service.db.add_history(user_id, outcome.article, outcome.data.get("verdict", ""))
        await safe_edit(status, outcome.html, reply_markup=result_keyboard(outcome))
    except Exception as exc:  # noqa: BLE001 — любую ошибку показываем понятным текстом
        if not isinstance(exc, (FetchError, ExtractError, LimitExceeded, LLMError)):
            log.exception("failed to process %s", url or "text")
        body = error_text(exc)
        if url and isinstance(exc, LLMError):
            body += f'\n\n<a href="{esc_attr(url)}">{esc(url[:200])}</a>'
        await safe_edit(status, body)


@router.message(StateFilter(ProfileForm.waiting), F.text)
async def profile_received(message: Message, state: FSMContext, service: ArticleService) -> None:
    text = message.text.strip()
    if text.startswith("/"):
        await state.clear()
        await message.answer("Ок, профиль не меняю. Повторите команду, если нужно.")
        return
    await service.db.update_user(message.from_user.id, profile=text[:2000])
    await state.clear()
    await message.answer(f"✅ Профиль сохранён:\n<i>{esc(text[:2000])}</i>")


@router.message(Command("start", "help"))
async def cmd_help(message: Message, is_admin: bool) -> None:
    await message.answer(HELP + (ADMIN_HELP if is_admin else ""))


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.")


@router.message(Command("profile"))
async def cmd_profile(message: Message, command: CommandObject, state: FSMContext,
                      service: ArticleService) -> None:
    args = (command.args or "").strip()
    user_id = message.from_user.id
    if args.lower() in ("clear", "reset", "удалить", "-"):
        await service.db.update_user(user_id, profile="")
        await message.answer("🗑 Профиль удалён — буду оценивать статьи в целом.")
        return
    if args:
        await service.db.update_user(user_id, profile=args[:2000])
        await message.answer(f"✅ Профиль сохранён:\n<i>{esc(args[:2000])}</i>")
        return
    prefs = await service.prefs(user_id)
    current = f"Сейчас:\n<i>{esc(prefs.profile)}</i>\n\n" if prefs.profile else "Профиль пока не задан.\n\n"
    await state.set_state(ProfileForm.waiting)
    await message.answer(
        current
        + "Пришлите одним сообщением, чем вы занимаетесь и что вам интересно, например:\n"
        "<i>бэкенд на Python, DevOps, велоспорт; не интересны крипта и маркетинг</i>\n\n"
        "/profile clear — удалить профиль, /cancel — отмена."
    )


@router.message(Command("history"))
async def cmd_history(message: Message, service: ArticleService) -> None:
    items = await service.db.history(message.from_user.id, 20)
    await message.answer(render_history(items))


# --- настройки ---------------------------------------------------------------

def settings_view(prefs) -> tuple[str, InlineKeyboardMarkup]:
    fmt = "📎 файл .md" if prefs.full_format == "md" else "🔗 страница telegra.ph"
    text = (
        "⚙️ <b>Настройки</b>\n\n"
        f"Язык выжимки: {esc(menu_name(prefs.lang))}\n"
        f"Второй язык (кнопка «Перевести»): {esc(menu_name(prefs.tr_lang))}\n"
        f"Полный перевод: {fmt}\n"
        f"Профиль: {'задан' if prefs.profile else 'не задан'} (/profile)"
    )
    kb = InlineKeyboardBuilder()
    kb.button(text="🗣 Язык выжимки", callback_data="set:lang")
    kb.button(text="🌐 Второй язык", callback_data="set:tr")
    kb.button(text="📄 Формат: " + ("telegra.ph" if prefs.full_format == "md" else ".md"), callback_data="set:fmt")
    kb.adjust(2, 1)
    return text, kb.as_markup()


def languages_keyboard(prefix: str, current: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for code, (name, _) in LANGUAGES.items():
        kb.button(text=("• " if code == current else "") + name, callback_data=f"{prefix}:{code}")
    kb.button(text="← Назад", callback_data="set:back")
    kb.adjust(3)
    return kb.as_markup()


@router.message(Command("settings"))
async def cmd_settings(message: Message, service: ArticleService) -> None:
    text, markup = settings_view(await service.prefs(message.from_user.id))
    await message.answer(text, reply_markup=markup)


@router.callback_query(F.data.startswith("set:") | F.data.startswith("setlang:") | F.data.startswith("settr:"))
async def cb_settings(call: CallbackQuery, service: ArticleService) -> None:
    user_id = call.from_user.id
    action, _, value = call.data.partition(":")
    prefs = await service.prefs(user_id)
    if action == "set" and value == "lang":
        await call.message.edit_text("🗣 На каком языке писать выжимки?",
                                     reply_markup=languages_keyboard("setlang", prefs.lang))
        await call.answer()
        return
    if action == "set" and value == "tr":
        await call.message.edit_text("🌐 На какой язык переводить по кнопке «Перевести»?",
                                     reply_markup=languages_keyboard("settr", prefs.tr_lang))
        await call.answer()
        return
    if action == "set" and value == "fmt":
        await service.db.update_user(user_id, full_format="telegraph" if prefs.full_format == "md" else "md")
    elif action == "setlang" and value in LANGUAGES:
        await service.db.update_user(user_id, lang=value)
    elif action == "settr" and value in LANGUAGES:
        await service.db.update_user(user_id, tr_lang=value)
    text, markup = settings_view(await service.prefs(user_id))
    await safe_edit(call.message, text, reply_markup=markup)
    await call.answer("Сохранено" if action != "set" or value == "fmt" else None)


# --- кнопки под выжимкой ---------------------------------------------------------

async def _guarded(call: CallbackQuery, busy_text: str):
    key = (call.from_user.id, call.data)
    if key in _in_progress:
        await call.answer("Уже делаю, подождите немного…")
        return None
    _in_progress.add(key)
    await call.answer(busy_text)
    return key


@router.callback_query(F.data.startswith("tr:"))
async def cb_translate(call: CallbackQuery, bot: Bot, service: ArticleService) -> None:
    prefs = await service.prefs(call.from_user.id)
    if prefs.tr_lang == prefs.lang:
        await call.answer("Второй язык совпадает с языком выжимки — смените его в /settings.", show_alert=True)
        return
    key = await _guarded(call, "Перевожу…")
    if not key:
        return
    status = await call.message.reply(f"⏳ Перевожу выжимку: {esc(menu_name(prefs.tr_lang))}…")
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=call.message.chat.id):
            html = await service.translate_summary(int(call.data.split(":", 1)[1]), prefs.tr_lang)
        await safe_edit(status, html)
    except KeyError:
        await safe_edit(status, "🤷 Выжимка не найдена в кэше — пришлите ссылку ещё раз.")
    except Exception as exc:  # noqa: BLE001
        if not isinstance(exc, LLMError):
            log.exception("translate failed")
        await safe_edit(status, error_text(exc))
    finally:
        _in_progress.discard(key)


@router.callback_query(F.data.startswith("more:"))
async def cb_more(call: CallbackQuery, bot: Bot, service: ArticleService) -> None:
    key = await _guarded(call, "Готовлю подробный разбор…")
    if not key:
        return
    status = await call.message.reply("⏳ Готовлю подробный разбор…")
    try:
        prefs = await service.prefs(call.from_user.id)
        async with ChatActionSender.typing(bot=bot, chat_id=call.message.chat.id):
            parts = await service.detailed(int(call.data.split(":", 1)[1]), prefs)
        status = await safe_edit(status, parts[0])
        for part in parts[1:]:
            await call.message.answer(part)
    except KeyError:
        await safe_edit(status, "🤷 Статья не найдена в кэше — пришлите ссылку ещё раз.")
    except Exception as exc:  # noqa: BLE001
        if not isinstance(exc, LLMError):
            log.exception("detailed failed")
        await safe_edit(status, error_text(exc))
    finally:
        _in_progress.discard(key)


@router.callback_query(F.data.startswith("full:"))
async def cb_full(call: CallbackQuery, bot: Bot, service: ArticleService) -> None:
    key = await _guarded(call, "Перевожу статью целиком — это может занять пару минут")
    if not key:
        return
    status = await call.message.reply("⏳ Перевожу статью целиком… Это может занять пару минут.")
    try:
        prefs = await service.prefs(call.from_user.id)
        async with ChatActionSender.upload_document(bot=bot, chat_id=call.message.chat.id):
            result = await service.full_translation(int(call.data.split(":", 1)[1]), prefs)
        lang = esc(menu_name(result.language))
        if result.telegraph_urls:
            links = "\n".join(
                f'<a href="{esc_attr(u)}">{esc(result.title[:80])}{"" if len(result.telegraph_urls) == 1 else f" — часть {i}"}</a>'
                for i, u in enumerate(result.telegraph_urls, 1)
            )
            await safe_edit(status, f"📄 Полный перевод ({lang}):\n{links}")
        else:
            caption = f"📄 Полный перевод ({lang}): {esc(result.title[:200])}"
            if result.note:
                caption += f"\n{esc(result.note)}"
            await call.message.answer_document(
                BufferedInputFile(result.markdown.encode("utf-8"), filename=result.filename),
                caption=caption[:1000],
            )
            try:
                await status.delete()
            except TelegramBadRequest:
                pass
    except KeyError:
        await safe_edit(status, "🤷 Статья не найдена в кэше — пришлите ссылку ещё раз.")
    except Exception as exc:  # noqa: BLE001
        if not isinstance(exc, (LLMError, TooLongForTranslation)):
            log.exception("full translation failed")
        await safe_edit(status, error_text(exc))
    finally:
        _in_progress.discard(key)


# --- админ ------------------------------------------------------------------

@router.message(Command("llmtest"))
async def cmd_llmtest(message: Message, is_admin: bool, settings: Settings, service: ArticleService) -> None:
    if not is_admin:
        await message.answer("Команда только для админа.")
        return
    llm = service.llm
    llm.reset_rejections()
    head = (
        f"🧪 <b>Проверка нейросети</b>\n"
        f"Провайдер: <code>{esc(settings.llm_provider)}</code>\n"
        f"Модель: <code>{esc(settings.llm_model)}</code>\n"
        f"URL: <code>{esc(settings.base_url)}</code>\n"
        f"LLM_MAX_TOKENS: {settings.llm_max_tokens}, LLM_REASONING: {esc(settings.llm_reasoning or 'по умолчанию модели')}\n"
    )
    status = await message.answer(head + "\n⏳ Отправляю запрос…")
    started = time.monotonic()
    try:
        result = await llm.complete(
            "You are a connectivity test. Follow the instruction exactly.",
            'Reply with this JSON only: {"ok": true, "answer": "<the word \\"pong\\" in Russian>"}',
        )
        took = time.monotonic() - started
        usage = ", ".join(f"{k}={v}" for k, v in (result.usage or {}).items() if isinstance(v, (int, float)))
        body = (
            f"\n✅ Ответ за {took:.1f} с\n"
            f"Модель в ответе: <code>{esc(result.model or '—')}</code>\n"
            f"finish_reason: <code>{esc(result.finish_reason or '—')}</code>\n"
            + (f"Токены: {esc(usage)}\n" if usage else "")
            + (f"Рассуждения: {result.reasoning_chars} символов\n" if result.reasoning_chars else "")
            + f"\n<pre>{esc(result.text[:1500])}</pre>"
        )
    except Exception as exc:  # noqa: BLE001
        took = time.monotonic() - started
        body = f"\n❌ Ошибка через {took:.1f} с:\n{esc(str(exc)) or esc(repr(exc))}"
    await safe_edit(status, head + body)


@router.message(Command("instructions"))
async def cmd_instructions(message: Message, command: CommandObject, is_admin: bool,
                           settings: Settings, service: ArticleService) -> None:
    if not is_admin:
        await message.answer("Команда только для админа.")
        return
    args = (command.args or "").strip()
    if args.lower() == "reset":
        await service.set_instructions(None)
        await message.answer("↩️ Вернул указания из .env (LLM_INSTRUCTIONS).")
        return
    if args.lower() in ("off", "clear"):
        await service.set_instructions("")
        await message.answer("🧹 Дополнительные указания отключены.")
        return
    if args:
        await service.set_instructions(args[:4000])
        await message.answer(f"✅ Указания сохранены:\n<i>{esc(args[:4000])}</i>")
        return
    current = await service.instructions()
    stored = await service.db.kv_get("llm_instructions")
    source = "из .env (LLM_INSTRUCTIONS)" if stored is None else "заданы командой /instructions"
    shown = f"<i>{esc(current)}</i>" if current else "— нет —"
    await message.answer(
        f"📝 <b>Указания для нейросети</b> ({source}):\n{shown}\n\n"
        "/instructions текст — задать\n/instructions off — отключить\n/instructions reset — вернуть из .env"
    )


@router.message(Command("stats"))
async def cmd_stats(message: Message, is_admin: bool, settings: Settings, service: ArticleService) -> None:
    if not is_admin:
        await message.answer("Команда только для админа.")
        return
    stats = await service.db.stats()
    await message.answer(
        "📊 <b>Статистика</b>\n"
        f"Пользователей с настройками: {stats['users']}\n"
        f"Статей в кэше: {stats['articles']}\n"
        f"Ответов нейросети в кэше: {stats['cache']}\n"
        f"Запросов в истории: {stats['history']}\n"
        f"Модель: <code>{esc(settings.llm_model)}</code>"
    )


# --- обычные сообщения -----------------------------------------------------------

@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await message.answer("Не знаю такой команды. /help — список команд.")


@router.message(F.text | F.caption)
async def on_message(message: Message, bot: Bot, settings: Settings, service: ArticleService,
                     is_admin: bool) -> None:
    urls = _message_urls(message)
    if urls:
        limit = settings.max_links_per_message
        if len(urls) > limit:
            await message.reply(f"В сообщении {len(urls)} ссылок — разберу первые {limit}.")
        for url in urls[:limit]:
            await _process(message, bot, service, is_admin, url=url)
        return
    text = (message.text or message.caption or "").strip()
    if len(text) >= settings.min_text_length:
        await _process(message, bot, service, is_admin, text=text)
        return
    await message.reply(
        "Пришлите ссылку на статью или длинный текст "
        f"(от {settings.min_text_length} символов) — сделаю выжимку. /help — подробнее."
    )


@router.message()
async def other_message(message: Message) -> None:
    if message.document and (message.document.mime_type or "") == "application/pdf":
        await message.reply("Пока умею читать PDF только по ссылке. Пришлите ссылку на файл.")
        return
    await message.reply("Пришлите ссылку на статью или текст. /help — подробнее.")


__all__ = ["router", "result_keyboard", "visible_length"]
