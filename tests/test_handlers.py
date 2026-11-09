from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiogram.types import Message

from bot.fetcher import FetchError
from bot.formatting import visible_length
from bot.handlers import on_message

from .conftest import FakeFetcher, FakeLLM, fixture_html, summary_json


def fake_message(text: str, user_id: int = 2):
    status = Mock()
    status.edit_text = AsyncMock(side_effect=lambda text, reply_markup=None: status)
    message = Mock(spec=Message)
    message.text = text
    message.caption = None
    message.entities = None
    message.caption_entities = None
    message.reply_markup = None
    message.link_preview_options = None
    message.from_user = SimpleNamespace(id=user_id)
    message.chat = SimpleNamespace(id=user_id)
    message.reply = AsyncMock(return_value=status)
    message.answer = AsyncMock()
    return message, status


class FailingFetcher(FakeFetcher):
    async def fetch(self, url):
        if "missing" in url:
            raise FetchError("not_found", "🤷 site.example: страница не найдена (404). Проверьте ссылку.")
        return await super().fetch(url)


async def test_message_with_two_links(make_service, db, settings):
    good = "https://habr.com/ru/articles/900001/"
    fetcher = FailingFetcher({good: (fixture_html("habr_like.html").encode(), "html")})
    service = make_service(FakeLLM([summary_json()]), fetcher)
    message, status = fake_message(f"Почитай {good}?utm_source=tg и https://site.example/missing")
    await on_message(message, AsyncMock(), settings, service, is_admin=False)

    assert [c.args[0] for c in message.reply.call_args_list] == ["⏳ Читаю статью…", "⏳ Читаю статью…"]
    first, second = status.edit_text.call_args_list
    html = first.args[0]
    assert "✅ <b>Читать</b>" in html and visible_length(html) <= 4096
    buttons = [b.text for row in first.kwargs["reply_markup"].inline_keyboard for b in row]
    assert buttons == ["🌐 Перевести", "📄 Полный перевод", "🔁 Подробнее"]
    assert "не найдена (404)" in second.args[0]
    assert len(await db.history(2)) == 1


async def test_long_text_without_links(make_service, db, settings):
    service = make_service(FakeLLM([summary_json(title="Про кэш")]))
    message, status = fake_message("Кэширование ускоряет сборку. " * 30)
    await on_message(message, AsyncMock(), settings, service, is_admin=False)
    assert message.reply.call_args.args[0] == "⏳ Читаю текст…"
    assert "Про кэш" in status.edit_text.call_args.args[0]


async def test_short_text_without_links(make_service, settings):
    service = make_service(FakeLLM([]))
    message, _ = fake_message("привет")
    await on_message(message, AsyncMock(), settings, service, is_admin=False)
    assert "Пришлите ссылку" in message.reply.call_args.args[0]


async def test_llm_error_is_shown_in_message(make_service, settings):
    from bot.llm import LLMProviderError

    service = make_service(FakeLLM([LLMProviderError(403, "Key limit exceeded")]))
    message, status = fake_message("Текст статьи для проверки ошибок. " * 30)
    await on_message(message, AsyncMock(), settings, service, is_admin=False)
    text = status.edit_text.call_args.args[0]
    assert "Key limit exceeded" in text and "/llmtest" in text
