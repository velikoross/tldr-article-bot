from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiogram.types import CallbackQuery, Message

from bot.access import AccessMiddleware, role_of
from bot.service import LimitExceeded

from .conftest import FakeLLM, summary_json


def test_roles(settings):
    assert role_of(1, settings) == "admin"
    assert role_of(2, settings) == "friend"
    assert role_of(3, settings) is None
    assert role_of(None, settings) is None


async def run_middleware(settings, user_id, event):
    handler = AsyncMock(return_value="handled")
    data = {"event_from_user": SimpleNamespace(id=user_id, username="u")}
    result = await AccessMiddleware(settings)(handler, event, data)
    return result, handler, data


async def test_admin_and_friend_pass(settings):
    for uid, admin in ((1, True), (2, False)):
        event = Mock(spec=Message)
        event.answer = AsyncMock()
        result, handler, data = await run_middleware(settings, uid, event)
        assert result == "handled"
        assert data["is_admin"] is admin
        event.answer.assert_not_called()


async def test_stranger_gets_private_notice_with_id(settings):
    event = Mock(spec=Message)
    event.answer = AsyncMock()
    result, handler, _ = await run_middleware(settings, 777000123, event)
    assert result is None
    handler.assert_not_called()
    text = event.answer.call_args.args[0]
    assert "приватный бот" in text
    assert "<code>777000123</code>" in text


async def test_stranger_button_press_is_rejected(settings):
    event = Mock(spec=CallbackQuery)
    event.answer = AsyncMock()
    result, handler, _ = await run_middleware(settings, 999, event)
    assert result is None
    handler.assert_not_called()
    assert event.answer.call_args.kwargs.get("show_alert") is True


async def test_daily_limit_for_friends_not_admins(make_service, db):
    service = make_service(FakeLLM(handler=lambda *_: summary_json()), daily_limit_per_user=2)
    articles = []
    for i in range(3):
        a = await service.article_from_text(f"Текст статьи номер {i}. " * 40)
        articles.append(a)

    for a in articles[:2]:
        await service.check_limit(2, False, a.id)
        out = await service.summarize(a, await service.prefs(2))
        await db.add_history(2, out.article, out.data["verdict"])

    # третья новая статья — сверх лимита
    try:
        await service.check_limit(2, False, articles[2].id)
        raise AssertionError("limit not enforced")
    except LimitExceeded as exc:
        assert "лимит" in str(exc)
    # повтор уже прочитанной сегодня статьи — можно
    await service.check_limit(2, False, articles[0].id)
    # на админа лимит не действует
    for a in articles[:2]:
        await db.add_history(1, a, "read")
    await service.check_limit(1, True, articles[2].id)


async def test_history_per_user(make_service, db):
    service = make_service(FakeLLM(handler=lambda *_: summary_json()))
    a = await service.article_from_text("Статья для истории. " * 40)
    out = await service.summarize(a, await service.prefs(2))
    await db.add_history(2, out.article, out.data["verdict"])
    assert len(await db.history(2)) == 1
    assert await db.history(1) == []
    item = (await db.history(2))[0]
    assert item.verdict == "read"
    assert item.title == "Как мы ускорили CI в четыре раза"


async def test_per_user_preferences(make_service, db):
    service = make_service(FakeLLM([]))
    await db.update_user(2, profile="Python", tr_lang="de")
    p2, p1 = await service.prefs(2), await service.prefs(1)
    assert (p2.profile, p2.tr_lang, p2.lang) == ("Python", "de", "ru")
    assert (p1.profile, p1.tr_lang) == ("", "en")
