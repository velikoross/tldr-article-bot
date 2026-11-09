from bot.extractor import Article
from bot.formatting import (
    TELEGRAM_LIMIT,
    render_detailed,
    render_history,
    render_summary,
    split_lines,
    visible_length,
)
from bot.db import HistoryItem


def article(**kw) -> Article:
    base = dict(url="https://blog.example/post?a=1&b=2", url_norm="k", title="Заголовок <script>", site="blog.example",
                author="Анна & Ко", published="2026-03-12", text="слово " * 2300)
    base.update(kw)
    return Article(**base)


DATA = {
    "verdict": "skim", "verdict_reason": "Есть пара полезных идей.", "kind": "promo",
    "audience": "Тимлидам.", "summary": ["Первый пункт.", "Второй <b>пункт</b>."],
    "warnings": [{"type": "promo", "text": "Реклама сервиса в конце."}],
}


def test_visible_length_counts_utf16_without_tags():
    assert visible_length("<b>abc</b>") == 3
    assert visible_length("&lt;tag&gt; &amp;") == 7
    assert visible_length("😀") == 2  # эмодзи — две единицы UTF-16
    assert visible_length('<a href="https://very.long/url">x</a>') == 1


def test_summary_layout_and_escaping():
    html = render_summary(article(), DATA)
    lines = html.split("\n")
    assert lines[0] == '📰 <b><a href="https://blog.example/post?a=1&amp;b=2">Заголовок &lt;script&gt;</a></b>'
    assert lines[1] == "🌐 blog.example · ✍️ Анна &amp; Ко · 📅 12.03.2026 · ⏱ ~12 мин"
    assert "👀 <b>Пролистать</b> — Есть пара полезных идей." in html
    assert "<i>Реклама продукта</i> · Тимлидам." in html
    assert "• <b>Реклама</b>: Реклама сервиса в конце." in html
    assert "• Второй &lt;b&gt;пункт&lt;/b&gt;." in html
    assert "<script>" not in html


def test_partial_article_gets_warning():
    html = render_summary(article(partial=True), {**DATA, "warnings": []})
    assert "только часть статьи" in html


def test_huge_summary_fits_in_one_message():
    data = {
        "verdict": "read",
        "verdict_reason": "Очень " * 400,
        "audience": "Всем " * 400,
        "kind": "practice",
        "warnings": [{"type": "other", "text": "Внимание " * 200}] * 4,
        "summary": ["😀 Очень длинный пункт с эмодзи " * 60 for _ in range(8)],
    }
    html = render_summary(article(title="Т" * 500), data)
    assert visible_length(html) <= TELEGRAM_LIMIT
    assert "📝 <b>Выжимка</b>" in html
    assert html.count("• ") >= 3
    # HTML остаётся валидным: теги сбалансированы
    assert html.count("<b>") == html.count("</b>")
    assert html.count("<a ") == html.count("</a>")


def test_normal_summary_is_untouched():
    data = {**DATA, "summary": [f"Пункт {i} с фактами и цифрами." for i in range(8)]}
    html = render_summary(article(), data)
    assert all(f"Пункт {i} с фактами" in html for i in range(8))


def test_detailed_splits_into_messages():
    data = {"sections": [{"title": f"Раздел {i}", "points": ["Подробность " * 50] * 5} for i in range(10)],
            "conclusion": "Итог."}
    messages = render_detailed(article(), data)
    assert len(messages) > 1
    assert all(visible_length(m) <= TELEGRAM_LIMIT for m in messages)
    assert "Итог." in messages[-1]


def test_split_lines():
    lines = ["x" * 3000, "y" * 3000, "z"]
    parts = split_lines(lines)
    assert len(parts) == 2
    assert parts[1].endswith("z")


def test_history_render():
    items = [HistoryItem("Статья & co", "https://a.example/1", "read", 1_700_000_000),
             HistoryItem("Текст", "", "skip", 1_700_000_000)]
    html = render_history(items)
    assert '✅ <a href="https://a.example/1">Статья &amp; co</a>' in html
    assert "⏭ Текст" in html
    assert render_history([]).startswith("История пуста")
