import pytest

from bot.extractor import Article
from bot.llm import LLMTruncated
from bot.summarizer import Summarizer, split_text

from .conftest import FakeLLM, summary_json


def long_article(paragraphs: int = 60) -> Article:
    text = "\n\n".join(
        f"Раздел {i}. " + f"Факт номер {i}: показатель вырос на {i} процентов. " * 20 for i in range(paragraphs)
    )
    return Article(url="https://long.example/a", url_norm="https://long.example/a", title="Длинная статья",
                   site="long.example", text=text, id=1)


def test_split_text_respects_size_and_keeps_everything():
    article = long_article()
    chunks = split_text(article.text, 5000)
    assert len(chunks) > 1
    assert all(len(c) <= 5000 for c in chunks)
    assert sum(len(c) for c in chunks) >= len(article.text) - 2 * len(chunks) * 2
    assert "Раздел 0." in chunks[0] and "Раздел 59." in chunks[-1]


def test_split_text_cuts_huge_paragraph():
    chunks = split_text("слово " * 5000, 1000)
    assert all(len(c) <= 1000 for c in chunks)


async def test_long_article_is_summarized_by_parts():
    article = long_article()

    def handler(system, user):
        if "This is part" in user:
            part = user.split("This is part ")[1].split(" ")[0]
            return f"- заметка по части {part}"
        return summary_json()

    llm = FakeLLM(handler=handler)
    s = Summarizer(llm, input_budget=10_000)
    result = await s.summarize(article, language="ru")
    map_calls = [u for _, u in llm.calls if "This is part" in u]
    reduce_calls = [u for _, u in llm.calls if "This is part" not in u]
    assert len(map_calls) == len(split_text(article.text, 10_000)) > 1
    assert all(len(u) < 12_000 for u in map_calls)
    assert len(reduce_calls) == 1
    assert "notes made from its consecutive parts" in reduce_calls[0]
    assert "заметка по части 1" in reduce_calls[0]
    assert result["verdict"] == "read"
    assert article.notes  # заметки сохраняются для «Подробнее» и других профилей

    # повторно заметки не пересчитываются
    llm.calls.clear()
    await s.summarize(article, language="ru", profile="другой профиль")
    assert all("This is part" not in u for _, u in llm.calls)


async def test_short_article_goes_in_one_request():
    article = long_article(3)
    llm = FakeLLM([summary_json()])
    await Summarizer(llm, input_budget=100_000).summarize(article, language="ru")
    assert len(llm.calls) == 1
    assert "Раздел 2." in llm.calls[0][1]


async def test_notes_are_stored_in_db(make_service, db):
    def handler(system, user):
        return "- заметка" if "This is part" in user else summary_json()

    service = make_service(FakeLLM(handler=handler), llm_context_tokens=13_000)
    article = long_article()
    article.url_norm = "https://long.example/stored"
    await db.save_article(article)
    await service.summarize(article, await service.prefs(1))
    stored = await db.get_article_by_id(article.id)
    assert stored.notes and "заметка" in stored.notes


async def test_full_translation_splits_truncated_chunk():
    calls = []

    def handler(system, user):
        calls.append(user)
        body = user.split(":\n\n", 1)[1]
        if len(body) > 3000:
            from bot.llm.base import LLMResult

            return LLMResult(text="обрезано", finish_reason="length")
        return "TR[" + body[:10] + "]"

    s = Summarizer(FakeLLM(handler=handler), input_budget=100_000, max_tokens=4000)
    text = "\n\n".join("Paragraph %d. " % i + "word " * 100 for i in range(20))
    result = await s.translate_text(text, language="ru")
    assert result.count("TR[") >= 2
    assert len(calls) > len(split_text(text, s.translation_chunk_size()))


async def test_full_translation_gives_up_on_tiny_truncated_chunk():
    from bot.llm.base import LLMResult

    s = Summarizer(FakeLLM(handler=lambda *_: LLMResult(text="x", finish_reason="length")), input_budget=10_000)
    with pytest.raises(LLMTruncated):
        await s.translate_text("Короткий абзац.", language="en")
