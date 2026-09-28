import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch
import pytest

from src.tools.web import fetch_webpage, web_search, _search_google_news


def test_fetch_webpage_clean_text():
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><style>body { color: red; }</style></head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <header><h1>Site Header</h1></header>
        <main>
            <h2>Main Article Title</h2>
            <p>This is the first paragraph with <b>bold text</b> and <a href="#">a link</a>.</p>
            <p>Second paragraph with &amp; entity &quot;quotes&quot;.</p>
        </main>
        <footer><p>Copyright 2026</p></footer>
        <script>console.log("ignore me");</script>
    </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.headers = MagicMock()
    mock_resp.headers.get.return_value = "text/html; charset=utf-8"
    mock_resp.headers.get_content_charset.return_value = "utf-8"
    mock_resp.read.return_value = html_sample.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = fetch_webpage("https://example.com/article")
        assert "Main Article Title" in res
        assert "This is the first paragraph with bold text and a link." in res
        assert 'Second paragraph with & entity "quotes".' in res
        assert "console.log" not in res
        assert "color: red" not in res
        assert "Site Header" not in res
        assert "Copyright" not in res


def test_fetch_webpage_truncation():
    long_html = "<html><body>" + "<p>Word </p>" * 200 + "</body></html>"

    mock_resp = MagicMock()
    mock_resp.headers = MagicMock()
    mock_resp.headers.get.return_value = "text/html"
    mock_resp.headers.get_content_charset.return_value = "utf-8"
    mock_resp.read.return_value = long_html.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = fetch_webpage("https://example.com", max_chars=100)
        assert len(res) <= 150  # 100 chars + truncation notice
        assert "[Content truncated]" in res


def test_fetch_webpage_empty():
    assert "Please provide a valid URL" in fetch_webpage("")


def test_search_google_news_parsing():
    rss_sample = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
        <title>Google News</title>
        <item>
            <title>AMD Announces Next-Gen Architecture - TechNews</title>
            <pubDate>Sun, 27 Sep 2026 14:00:00 GMT</pubDate>
            <link>https://news.example.com/1</link>
        </item>
        <item>
            <title>Radeon Performance Update - GPUWorld</title>
            <pubDate>Sat, 26 Sep 2026 10:00:00 GMT</pubDate>
            <link>https://news.example.com/2</link>
        </item>
    </channel>
    </rss>"""

    mock_resp = MagicMock()
    mock_resp.read.return_value = rss_sample
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = _search_google_news("AMD news")
        assert res is not None
        assert "AMD Announces Next-Gen Architecture" in res
        assert "Radeon Performance Update" in res


def test_web_search_routes_to_news():
    with patch("src.tools.web._search_google_news", return_value="Recent news headlines:\n- AMD Launch") as mock_news:
        res = web_search("what happened recently with AMD?")
        assert "AMD Launch" in res
        mock_news.assert_called_once()
