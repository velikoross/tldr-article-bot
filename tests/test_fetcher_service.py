import httpx
import pytest

from bot.fetcher import FetchError, Fetcher
from bot.service import TooLongForTranslation
from bot.telegraph import TelegraphError, markdown_to_nodes, split_nodes

from .conftest import FakeFetcher, FakeLLM, fixture_html, summary_json


def fetcher(handler, max_bytes=5 * 1024 * 1024) -> Fetcher:
    return Fetcher(timeout=5, max_bytes=max_bytes, user_agent="test", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status,kind,phrase", [
    (403, "forbidden", "403"),
    (404, "not_found", "не найдена"),
    (402, "paywall", "подписки"),
    (503, "server_error", "503"),
])
async def test_http_errors_are_explained(status, kind, phrase):
    f = fetcher(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(FetchError) as err:
        await f.fetch("https://site.example/a")
    assert err.value.kind == kind
    assert phrase in str(err.value)
    assert "site.example" in str(err.value)


async def test_too_large_by_header_and_by_body():
    f = fetcher(lambda r: httpx.Response(200, headers={"content-length": str(10 * 1024 * 1024),
                                                        "content-type": "text/html"}, content=b"x"))
    with pytest.raises(FetchError) as err:
        await f.fetch("https://site.example/big")
    assert err.value.kind == "too_large" and "5 МБ" in str(err.value)

    f = fetcher(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 2000),
                max_bytes=1000)
    with pytest.raises(FetchError) as err:
        await f.fetch("https://site.example/big")
    assert err.value.kind == "too_large"


async def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError) as err:
        await fetcher(handler).fetch("https://slow.example/a")
    assert err.value.kind == "timeout"


async def test_content_types():
    f = fetcher(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"))
    assert (await f.fetch("https://d.example/x.pdf")).content_type == "pdf"
    f = fetcher(lambda r: httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"))
    with pytest.raises(FetchError) as err:
        await f.fetch("https://d.example/x.png")
    assert err.value.kind == "unsupported"
    f = fetcher(lambda r: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=b"<html>"))
    result = await f.fetch("https://d.example/page#frag")
    assert result.content_type == "html" and result.encoding == "utf-8"
    assert "#frag" not in result.url


async def test_service_full_translation_md(make_service, db):
    url = "https://blog.example/en"
    fetcher_ = FakeFetcher({url: (fixture_html("blog.html").encode(), "html")})

    def handler(system, user):
        if user.startswith("Translate into"):
            return "## Перевод\n\nТекст перевода."
        return summary_json(lang="en")

    llm = FakeLLM(handler=handler)
    service = make_service(llm, fetcher_)
    prefs = await service.prefs(2)
    outcome = await service.summarize(await service.load_article(url), prefs)
    result = await service.full_translation(outcome.article.id, prefs)
    assert result.language == "ru"  # английская статья → на основной язык
    assert result.filename.endswith(".ru.md")
    assert result.markdown.startswith("# Five lessons")
    assert "Оригинал: https://blog.example/en" in result.markdown
    n = len(llm.calls)
    await service.full_translation(outcome.article.id, prefs)
    assert len(llm.calls) == n  # из кэша


async def test_full_translation_of_russian_article_goes_to_second_language(make_service, db):
    url = "https://habr.com/ru/articles/900001/"
    service = make_service(FakeLLM(handler=lambda s, u: "## Translated" if u.startswith("Translate into") else summary_json()),
                           FakeFetcher({url: (fixture_html("habr_like.html").encode(), "html")}))
    prefs = await service.prefs(2)
    outcome = await service.summarize(await service.load_article(url), prefs)
    article = await db.get_article_by_id(outcome.article.id)
    assert article.lang == "ru"
    assert service.full_translation_language(article, prefs) == "en"


async def test_full_translation_too_long(make_service, db):
    service = make_service(FakeLLM(handler=lambda *_: summary_json()), full_translation_max_chars=500)
    article = await service.article_from_text("Очень длинный текст. " * 100)
    with pytest.raises(TooLongForTranslation):
        await service.full_translation(article.id, await service.prefs(1))


async def test_telegraph_failure_falls_back_to_file(make_service, db):
    class BrokenTelegraph:
        async def publish(self, *a, **kw):
            raise TelegraphError("telegra.ph недоступен")

    service = make_service(FakeLLM(handler=lambda s, u: "перевод" if u.startswith("Translate") else summary_json()))
    service.telegraph = BrokenTelegraph()
    await db.update_user(1, full_format="telegraph")
    article = await service.article_from_text("Some English text about caching. " * 40)
    result = await service.full_translation(article.id, await service.prefs(1))
    assert result.telegraph_urls is None
    assert "файлом" in result.note
    assert result.markdown


def test_markdown_to_telegraph_nodes():
    md = "# Title\n\nPara with **bold** and `code` and [link](https://x.example).\n\n- a\n- b\n\n```\nx = 1\n```\n\n> quote"
    nodes = markdown_to_nodes(md)
    tags = [n["tag"] for n in nodes]
    assert tags == ["h3", "p", "ul", "pre", "blockquote"]
    para = nodes[1]["children"]
    assert {"tag": "b", "children": ["bold"]} in para
    assert {"tag": "a", "attrs": {"href": "https://x.example"}, "children": ["link"]} in para
    assert len(split_nodes([{"tag": "p", "children": ["x" * 1000]}] * 200, 60_000)) > 1
