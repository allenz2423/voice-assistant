import json
import urllib.parse
import urllib.request
from typing import Optional

CRYPTO_MAP = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "xmr": "monero",
    "monero": "monero"
}

def get_financial_quote(symbol: str) -> str:
    """Gets live price and change for cryptocurrencies or stock tickers."""
    sym = (symbol or "").strip().lower()
    if not sym:
        return "Please specify a ticker or crypto symbol, such as Bitcoin, Ethereum, or NVDA."

    # 1. Check if it is a known cryptocurrency
    if sym in CRYPTO_MAP:
        coin_id = CRYPTO_MAP[sym]
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
            req = urllib.request.Request(url, headers={"User-Agent": "AdamAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
            info = data.get(coin_id, {})
            price = info.get("usd")
            change = info.get("usd_24h_change", 0.0)
            direction = "up" if change >= 0 else "down"
            return f"{coin_id.title()} is currently trading at {price:,.2f} US dollars, {direction} {abs(change):.1f} percent over the last 24 hours."
        except Exception as e:
            return f"Unable to fetch crypto price for {sym}: {e}"

    # 2. Try Yahoo Finance chart API for equity / stock ticker
    ticker = sym.upper()
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose", price)
        curr = meta.get("currency", "USD")
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        direction = "up" if change_pct >= 0 else "down"
        return f"{ticker} is currently trading at {price:,.2f} {curr}, {direction} {abs(change_pct):.1f} percent today."
    except Exception as e:
        return f"Unable to fetch financial quote for '{symbol}': {e}"
