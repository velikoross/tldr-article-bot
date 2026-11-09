import pytest

from bot.extractor import ExtractError, decode_html, extract_html, extract_pdf, extract_plain

from .conftest import fixture_html

HABR_URL = "https://habr.com/ru/articles/900001/"


def make_pdf(lines: list[str], title: str = "Test PDF", author: str = "Tester") -> bytes:
    stream = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title ({title}) /Author ({author}) /CreationDate (D:20240115120000Z) >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_habr_like_page_uses_site_rule():
    article = extract_html(fixture_html("habr_like.html"), HABR_URL, HABR_URL)
    assert article.title == "Как мы ускорили CI в четыре раза"
    assert article.author == "ivan_petrov"
    assert article.published == "2026-03-12"
    assert article.site == "habr.com"
    text = article.text
    assert text.startswith("Год назад сборка")
    assert "## Кэширование зависимостей" in text
    assert "hashFiles" in text  # код сохранился
    assert "Итог: сорок минут превратились в десять" in text
    # мусор вокруг статьи не попал в текст
    for junk in ("Комментарии", "удалённый кэш", "Читают сейчас", "опросе", "Хабы:", "Моя лента", "©"):
        assert junk not in text
    assert 150 < article.words < 250
    assert article.reading_minutes == 1


def test_habr_like_page_without_site_rule_still_skips_comments():
    url = "https://mirror.example/post/1"
    article = extract_html(fixture_html("habr_like.html"), url, url)
    assert "Кэширование зависимостей" in article.text
    assert "удалённый кэш" not in article.text
    assert "Читают сейчас" not in article.text


def test_regular_blog():
    url = "https://crumbs.example/five-lessons"
    article = extract_html(fixture_html("blog.html"), url, url)
    assert article.title == "Five lessons from running a tiny bakery website"
    assert article.author == "Jane Doe"
    assert article.published == "2025-09-02"
    assert article.site == "crumbs.example"
    assert "three plain dropdowns" in article.text
    assert "Back up the database" in article.text
    assert "Recent posts" not in article.text
    assert "Great read" not in article.text
    assert not article.partial


def test_page_without_text():
    with pytest.raises(ExtractError) as err:
        extract_html(fixture_html("empty.html"), "https://app.example/", "https://app.example")
    assert err.value.kind == "no_text"
    assert "app.example" in str(err.value)


def test_paywall_page():
    with pytest.raises(ExtractError) as err:
        extract_html(fixture_html("paywall.html"), "https://times.example/a", "https://times.example/a")
    assert err.value.kind == "paywall"
    assert "платным доступом" in str(err.value)


def test_paywall_preview_with_some_text_is_marked_partial():
    body = "<p>" + "Free preview paragraph about cloud pricing and hidden costs. " * 12 + "</p>"
    html = fixture_html("paywall.html").replace("<p>Cloud providers love free tiers.</p>", body)
    article = extract_html(html, "https://times.example/a", "https://times.example/a")
    assert article.partial
    assert article.author == "John Smith"


def test_empty_html_string():
    with pytest.raises(ExtractError):
        extract_html("   ", "https://x.example", "https://x.example")


def test_pdf_extraction():
    lines = [f"Line {i}: quarterly report shows revenue growth of {i} percent." for i in range(20)]
    article = extract_pdf(make_pdf(lines), "https://docs.example/report.pdf", "k")
    assert article.source == "pdf"
    assert article.title == "Test PDF"
    assert article.author == "Tester"
    assert article.published == "2024-01-15"
    assert "revenue growth of 7 percent" in article.text


def test_pdf_without_text_layer():
    with pytest.raises(ExtractError) as err:
        extract_pdf(make_pdf([]), "https://docs.example/scan.pdf", "k")
    assert err.value.kind == "pdf_no_text"


def test_broken_pdf():
    with pytest.raises(ExtractError) as err:
        extract_pdf(b"%PDF-1.4 garbage", "https://docs.example/bad.pdf", "k")
    assert err.value.kind in ("pdf_broken", "pdf_no_text")


def test_decode_windows_1251_from_meta():
    html = '<html><head><meta charset="windows-1251"></head><body>Привет</body></html>'
    assert "Привет" in decode_html(html.encode("cp1251"))


def test_plain_text_too_short():
    with pytest.raises(ExtractError):
        extract_plain("коротко", "", "text:1", min_chars=300)
