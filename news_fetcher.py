"""
Optional news layer. Alpaca's Benzinga feed supplies headlines. A headline
never opens a trade by itself: it only nominates symbols that are not already
on the watchlist. Those names still have to pass the forecast.
"""
import re
from html import unescape

import config

_STOCK = re.compile(r"[A-Z][A-Z0-9.]{0,9}\Z")
_articles = None
_error = ""


def normalize_news_symbol(raw: str):
    """Turn an Alpaca news ticker into the symbol the bot trades."""
    if not isinstance(raw, str):
        return None
    symbol = raw.strip().upper()
    if not symbol or any(ch.isspace() for ch in symbol):
        return None
    if "/" in symbol:
        base, _, quote = symbol.partition("/")
        if quote != "USD" or not base.isalnum() or len(base) < 2:
            return None
        return f"{base}/USD"
    # Alpaca tags crypto as BTCUSD. A stock ticker does not end in USD.
    if symbol.endswith("USD") and len(symbol) >= 5:
        base = symbol[:-3]
        if base.isalnum() and len(base) >= 2:
            return f"{base}/USD"
    if _STOCK.fullmatch(symbol):
        return symbol
    return None


def select_news_symbols(articles, watchlist, market_open, max_symbols):
    """Newest symbols first, skipping the watchlist and closed-market stocks."""
    watched = set()
    for raw in watchlist:
        normalized = normalize_news_symbol(raw)
        if normalized:
            watched.add(normalized)

    chosen = []
    seen = set()
    for article in articles:
        for raw in article.get("symbols") or []:
            symbol = normalize_news_symbol(raw)
            if symbol is None or symbol in seen or symbol in watched:
                continue
            seen.add(symbol)
            if "/" not in symbol and not market_open:
                continue
            chosen.append(symbol)
            if len(chosen) >= max_symbols:
                return chosen
    return chosen


def build_news_brief(
    articles,
    watchlist,
    market_open,
    *,
    max_symbols,
    confidence_floor,
    position_scale,
    include_closed=False,
) -> list[dict]:
    """Symbols the scan would add, plus the exact text attached to each forecast.

    include_closed keeps stock names visible while the market is shut. They are
    marked waiting so the dashboard can show them without treating them as sent.
    """
    symbols = select_news_symbols(
        articles, watchlist, market_open or include_closed, max_symbols,
    )
    brief = []
    for symbol in symbols:
        waiting = (not market_open) and ("/" not in symbol)
        brief.append({
            "symbol": symbol,
            "sent": summarize_for_symbol(articles, symbol),
            "confidence_floor": confidence_floor,
            "position_scale": position_scale,
            "status": "waiting" if waiting else "sent",
        })
    return brief


def summarize_for_symbol(articles, symbol: str) -> str:
    """Join up to three headlines. Does not invent a bullish or bearish label."""
    wanted = normalize_news_symbol(symbol) or symbol
    lines = []
    for article in articles:
        tags = [
            normalize_news_symbol(raw) for raw in (article.get("symbols") or [])
        ]
        if wanted not in tags:
            continue
        headline = unescape((article.get("headline") or "").strip())
        summary = unescape((article.get("summary") or "").strip())
        if headline and summary:
            lines.append(f"{headline} — {summary}")
        elif headline or summary:
            lines.append(headline or summary)
        if len(lines) == 3:
            break
    if not lines:
        return f"No recent headlines for {symbol}."
    return " ".join(lines)


def _article_dicts(news_set):
    articles = []
    for article in news_set.data.get("news", []):
        articles.append({
            "headline": article.headline,
            "summary": article.summary,
            "symbols": list(article.symbols or []),
        })
    return articles


def _fetch(symbols=None, limit=50):
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        return [], "Alpaca API key not set"
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        request = NewsRequest(symbols=symbols, limit=limit, sort="desc")
        return _article_dicts(client.get_news(request)), ""
    except Exception as exc:
        return [], str(exc)


def refresh_market_news(limit=50):
    """One market-wide fetch, reused for discovery and per-symbol blurbs."""
    global _articles, _error
    _articles, _error = _fetch(limit=limit)
    return _articles


def news_scan(watchlist, market_open, *, include_closed=False) -> dict:
    """One fetch plus the brief the dashboard and the trading cycle can share."""
    articles = refresh_market_news()
    if _error:
        return {"error": _error, "items": []}
    return {
        "error": "",
        "items": build_news_brief(
            articles,
            watchlist,
            market_open,
            max_symbols=getattr(config, "NEWS_MAX_SYMBOLS", 5),
            confidence_floor=getattr(config, "NEWS_MIN_TRADE_CONFIDENCE", 70),
            position_scale=getattr(config, "NEWS_POSITION_SCALE", 0.5),
            include_closed=include_closed,
        ),
    }


def discover_news_symbols(watchlist, market_open):
    articles = refresh_market_news()
    return select_news_symbols(
        articles,
        watchlist,
        market_open,
        getattr(config, "NEWS_MAX_SYMBOLS", 5),
    )


def get_news_summary(symbol: str) -> str:
    global _articles, _error
    if _articles is None:
        query = symbol.replace("/", "") if "/" in symbol else symbol
        _articles, _error = _fetch(symbols=query, limit=10)
    if _error:
        return f"News fetch failed: {_error}"
    return summarize_for_symbol(_articles, symbol)
