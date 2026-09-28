"""Web search and webpage fetching tools."""

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape
from typing import Optional


def _search_google_news(query: str) -> Optional[str]:
    """Fetches real-time news headlines via Google News RSS."""
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")
        if not items:
            return None

        headlines = []
        for item in items[:5]:
            t = item.find("title")
            title = t.text if t is not None else ""
            d = item.find("pubDate")
            date = d.text if d is not None else ""
            if title:
                # Format: Title - Source (Date)
                clean_date = date[:16].strip() if date else ""
                suffix = f" ({clean_date})" if clean_date else ""
                headlines.append(f"- {title}{suffix}")

        if headlines:
            return "Recent news headlines:\n" + "\n".join(headlines)
    except Exception:
        pass
    return None


def fetch_webpage(url: str, max_chars: int = 2500) -> str:
    """Fetches and extracts clean, readable text from any webpage URL.

    Args:
        url: The web address to fetch (e.g. 'https://en.wikipedia.org/wiki/AMD' or 'theverge.com/article').
        max_chars: Maximum characters of text to return (default: 2500).
    """
    clean_url = (url or "").strip()
    if not clean_url:
        return "Please provide a valid URL."
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    try:
        req = urllib.request.Request(
            clean_url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        with urllib.request.urlopen(req, timeout=7) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if any(b in content_type for b in ["pdf", "image", "audio", "video", "octet-stream"]):
                return f"Cannot read binary content from {clean_url} (type: {content_type})."
            raw_data = resp.read(200000)  # Read up to 200KB of HTML
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw_data.decode(charset, errors="replace")

        # Strip scripts, styles, metadata, svgs, forms, navs, headers, footers
        clean = re.sub(
            r"<(script|style|svg|noscript|nav|header|footer|aside|form|meta)[^>]*>.*?</\1>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Convert paragraph/break tags into newlines
        clean = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", clean, flags=re.IGNORECASE)
        # Strip all remaining tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Decode HTML entities
        clean = unescape(clean)
        # Collapse excessive whitespace while preserving paragraph breaks
        lines = []
        for raw_line in clean.splitlines():
            l = re.sub(r"[ \t]+", " ", raw_line).strip()
            l = re.sub(r"\s+([.,;:!?])", r"\1", l)
            if l:
                lines.append(l)
        text = "\n".join(lines)

        if not text:
            return f"No readable text content found at {clean_url}."

        truncated = text[:max_chars].strip()
        if len(text) > max_chars:
            truncated += "...\n[Content truncated]"
        return truncated
    except Exception as e:
        return f"Could not fetch webpage {clean_url}: {e}"


def web_search(query: str) -> str:
    """Searches the web via Google News RSS, Wikipedia REST API, and DuckDuckGo."""
    q = (query or "").strip()
    if not q:
        return "Please provide a search query."

    q_lower = q.lower()
    is_news_query = any(k in q_lower for k in ["news", "happened", "recent", "today", "yesterday", "announced", "latest", "update"])

    # 1. If asking for news/recent events, prioritize Google News RSS
    if is_news_query:
        news_res = _search_google_news(q)
        if news_res:
            return news_res

    # 2. Try Wikipedia summary API for direct concepts / entities / people
    try:
        clean_q = re.sub(r"^(who is|who was|what is|what are|tell me about|define)\s+", "", q, flags=re.IGNORECASE).strip("? ")
        encoded = urllib.parse.quote(clean_q.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ShinAssistant/1.0 (https://github.com/shin/voice-assistant)"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            extract = data.get("extract")
            if extract and len(extract) > 40 and not data.get("type") == "disambiguation":
                return extract[:800]
    except Exception:
        pass

    # 3. Try DuckDuckGo HTML Lite
    try:
        ddg_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
        req = urllib.request.Request(ddg_url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
        })
        with urllib.request.urlopen(req, timeout=4) as resp:
            if resp.status == 200:
                html = resp.read().decode("utf-8", errors="replace")
                snippets = re.findall(r"<a class=\"result__snippet[^>]*>(.*?)</a>", html, re.DOTALL)
                if snippets:
                    clean_snippets = []
                    for s in snippets[:3]:
                        cleaned = re.sub(r"<[^>]+>", "", s)
                        clean_snippets.append(unescape(cleaned).strip())
                    return " ".join(clean_snippets)[:800]
    except Exception:
        pass

    # 4. Fallback to Google News RSS even for general queries
    news_fallback = _search_google_news(q)
    if news_fallback:
        return news_fallback

    return f"No results found for query '{query}'."
