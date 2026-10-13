"""Tests for picking trade symbols out of Alpaca news."""
import news_fetcher


def _article(symbols, headline="Headline", summary="Summary"):
    return {"symbols": symbols, "headline": headline, "summary": summary}


def test_news_symbols_are_normalized_to_bot_tickers():
    assert news_fetcher.normalize_news_symbol("BTCUSD") == "BTC/USD"
    assert news_fetcher.normalize_news_symbol("btc/usd") == "BTC/USD"
    assert news_fetcher.normalize_news_symbol("AAPL") == "AAPL"
    assert news_fetcher.normalize_news_symbol("BRK.B") == "BRK.B"
    assert news_fetcher.normalize_news_symbol("  ") is None
    assert news_fetcher.normalize_news_symbol("NOT A TICKER") is None


def test_news_scan_skips_the_watchlist_and_closed_stocks():
    articles = [
        _article(["AAPL"], "Apple already watched"),
        _article(["NVDA"], "Nvidia is new"),
        _article(["BTCUSD"], "Bitcoin moves"),
        _article(["MSFT"], "Microsoft is new too"),
    ]
    picked = news_fetcher.select_news_symbols(
        articles, watchlist=["AAPL"], market_open=False, max_symbols=5,
    )
    assert picked == ["BTC/USD"]


def test_news_scan_keeps_new_stocks_while_the_market_is_open():
    articles = [
        _article(["NVDA"], "Nvidia"),
        _article(["AAPL"], "Already watched"),
        _article(["BTC/USD"], "Bitcoin"),
    ]
    picked = news_fetcher.select_news_symbols(
        articles, watchlist=["AAPL", "BTC/USD"], market_open=True, max_symbols=5,
    )
    assert picked == ["NVDA"]


def test_news_scan_stops_at_the_cap_in_article_order():
    articles = [
        _article(["NVDA"]),
        _article(["AMD"]),
        _article(["INTC"]),
    ]
    picked = news_fetcher.select_news_symbols(
        articles, watchlist=[], market_open=True, max_symbols=2,
    )
    assert picked == ["NVDA", "AMD"]


def test_summary_uses_headlines_for_the_normalized_symbol():
    articles = [
        _article(["BTCUSD"], "Bitcoin ETF inflows", "Funds added exposure."),
        _article(["AAPL"], "Apple event", "Ignored."),
        _article(["BTC/USD"], "Bitcoin holds", "Price steady overnight."),
    ]
    text = news_fetcher.summarize_for_symbol(articles, "BTC/USD")
    assert "Bitcoin ETF inflows" in text
    assert "Bitcoin holds" in text
    assert "Apple event" not in text


def test_summary_says_when_a_symbol_has_no_headlines():
    text = news_fetcher.summarize_for_symbol([], "NVDA")
    assert text == "No recent headlines for NVDA."


def test_news_brief_records_the_text_sent_to_the_model():
    articles = [_article(["NVDA"], "Nvidia surges", "Chip demand rose.")]
    items = news_fetcher.build_news_brief(
        articles,
        watchlist=["AAPL"],
        market_open=True,
        max_symbols=5,
        confidence_floor=70,
        position_scale=0.5,
    )
    assert len(items) == 1
    assert items[0]["symbol"] == "NVDA"
    assert "Nvidia surges" in items[0]["sent"]
    assert "Chip demand rose." in items[0]["sent"]
    assert items[0]["confidence_floor"] == 70
    assert items[0]["position_scale"] == 0.5
