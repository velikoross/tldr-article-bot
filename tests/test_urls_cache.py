from types import SimpleNamespace

from bot.urls import find_urls, normalize_url, text_key

from .conftest import FakeFetcher, FakeLLM, fixture_html, summary_json


def test_normalize_strips_tracking_and_fragment():
    a = normalize_url("https://www.Habr.com/ru/articles/123/?utm_source=tg&utm_medium=social#comments")
    b = normalize_url("http://habr.com/ru/articles/123")
    assert a == b == "https://habr.com/ru/articles/123"


def test_normalize_keeps_meaningful_params_sorted():
    a = normalize_url("https://news.example/item?id=5&page=2&fbclid=abc")
    b = normalize_url("https://news.example/item?page=2&id=5")
    assert a == b == "https://news.example/item?id=5&page=2"


def test_normalize_host_specific_params_and_mobile():
    assert normalize_url("https://medium.com/@a/post-1?source=rss----1") == "https://medium.com/@a/post-1"
    assert normalize_url("https://m.habr.com/ru/post/1/") == "https://habr.com/ru/post/1"
    # параметр source важен на других сайтах
    assert "source=x" in normalize_url("https://site.example/a?source=x")


def test_find_urls_in_text_and_entities():
    text = "Глянь https://blog.example/post?utm_source=x, и ещё (https://news.example/a). А это t.me/channel"
    entities = [SimpleNamespace(type="text_link", url="https://docs.example/guide", offset=0, length=5)]
    urls = find_urls(text, entities)
    assert urls == ["https://docs.example/guide", "https://blog.example/post?utm_source=x", "https://news.example/a"]


def test_find_urls_deduplicates_by_normalized_url():
    urls = find_urls("https://a.example/x https://a.example/x/#top https://a.example/x?utm_campaign=1")
    assert urls == ["https://a.example/x"]


def test_find_urls_utf16_entity_offsets():
    text = "😀 ссылка example.com/page"
    # Telegram считает смещения в UTF-16: эмодзи = 2 единицы
    entities = [SimpleNamespace(type="url", offset=10, length=16)]
    assert find_urls(text, entities) == ["https://example.com/page"]  # ссылка без схемы
    text = "😀 https://e.example/p"
    entities = [SimpleNamespace(type="url", offset=3, length=19)]
    assert find_urls(text, entities) == ["https://e.example/p"]


def test_find_urls_skips_telegram_links_from_forwarded_posts():
    assert find_urls("Подписывайтесь https://t.me/somechannel") == []


def test_text_key_ignores_whitespace():
    assert text_key("a  b\nc") == text_key("a b c")


async def test_repeated_link_is_served_from_cache(make_service, db):
    url = "https://habr.com/ru/articles/900001/"
    fetcher = FakeFetcher({url: (fixture_html("habr_like.html").encode(), "html")})
    llm = FakeLLM([summary_json()])
    service = make_service(llm, fetcher)
    prefs = await service.prefs(2)

    first = await service.summarize(await service.load_article(url), prefs)
    assert not first.cached
    assert "Как мы ускорили CI" in first.html

    # та же статья с utm-меткой и якорем: ни скачивания, ни запроса к нейросети
    again_url = "https://habr.com/ru/articles/900001/?utm_source=telegram#habracut"
    second = await service.summarize(await service.load_article(again_url), prefs)
    assert second.cached
    assert second.html == first.html
    assert len(fetcher.calls) == 1
    assert len(llm.calls) == 1


async def test_profile_change_makes_new_verdict_but_reuses_article(make_service, db):
    url = "https://habr.com/ru/articles/900001/"
    fetcher = FakeFetcher({url: (fixture_html("habr_like.html").encode(), "html")})
    llm = FakeLLM([summary_json(), summary_json(verdict="skip")])
    service = make_service(llm, fetcher)
    article = await service.load_article(url)
    await service.summarize(article, await service.prefs(2))
    await db.update_user(2, profile="фронтенд, дизайн")
    out = await service.summarize(article, await service.prefs(2))
    assert out.data["verdict"] == "skip"
    assert "фронтенд, дизайн" in llm.calls[-1][1]
    assert len(fetcher.calls) == 1


async def test_translation_is_cached_per_language(make_service, db):
    url = "https://habr.com/ru/articles/900001/"
    fetcher = FakeFetcher({url: (fixture_html("habr_like.html").encode(), "html")})
    translated = summary_json(
        title="How we made CI four times faster", verdict_reason="Hands-on.", audience="DevOps folks.",
        summary=["Build took 40 minutes.", "Cache saved 15 minutes.", "Tests in 4 shards."],
    )
    llm = FakeLLM([summary_json(), translated, summary_json(summary=["Der Build dauerte 40 Minuten."])]
                  + ['{"audience": "Für wen", "summary": "Zusammenfassung", "warnings": "Achtung", '
                     '"minutes": "Min", "partial": "Teilweise", "verdict_names": {"read": "Lesen", '
                     '"skim": "Überfliegen", "skip": "Überspringen"}, "kind_names": {"practice": "Praxis"}, '
                     '"warning_names": {"promo": "Werbung"}}'])
    service = make_service(llm, fetcher)
    outcome = await service.summarize(await service.load_article(url), await service.prefs(2))

    en1 = await service.translate_summary(outcome.cache_id, "en")
    en2 = await service.translate_summary(outcome.cache_id, "en")
    assert en1 == en2
    assert "How we made CI four times faster" in en1
    assert "Who it's for" in en1  # подписи разделов тоже на английском
    assert len(llm.calls) == 2

    de = await service.translate_summary(outcome.cache_id, "de")
    assert "Der Build dauerte 40 Minuten." in de
    assert "Lesen" in de
    assert len(llm.calls) == 4  # перевод выжимки + разовый перевод подписей


async def test_text_message_summary(make_service, db):
    llm = FakeLLM([summary_json(title="Заметка о кэше")])
    service = make_service(llm)
    text = "Длинный текст про кэширование. " * 40
    article = await service.article_from_text(text)
    out = await service.summarize(article, await service.prefs(1))
    assert "Заметка о кэше" in out.html
    assert "href" not in out.html.split("\n")[0]  # без ссылки
    again = await service.article_from_text(text)
    assert again.id == article.id
