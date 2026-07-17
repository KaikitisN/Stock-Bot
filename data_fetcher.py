"""Pulls recent price bars from Alpaca for a list of symbols.
Automatically routes crypto (e.g. BTC/USD) to CryptoHistoricalDataClient
and stocks/ETFs to StockHistoricalDataClient.
"""
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import config


def _is_crypto(symbol: str) -> bool:
    """Return True if the symbol looks like a crypto pair (contains '/')."""
    return "/" in symbol


def _split_symbols(symbols):
    """Split a list of symbols into (crypto_list, stock_list)."""
    crypto = [s for s in symbols if _is_crypto(s)]
    stocks = [s for s in symbols if not _is_crypto(s)]
    return crypto, stocks


def _get_stock_client():
    return StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def _get_crypto_client():
    return CryptoHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def _smart_round(value: float) -> float:
    """Round to 2 dp for normal prices; preserve up to 10 significant figures
    for micro-priced assets like SHIB ($0.0000145) that round to $0.00 at 2dp.
    """
    if value == 0:
        return 0.0
    if value >= 0.01:
        return round(value, 2)
    # Find the first significant digit and keep 6 sig figs beyond it
    import math
    magnitude = math.floor(math.log10(abs(value)))
    decimal_places = max(2, -magnitude + 5)
    return round(value, decimal_places)


def fetch_bars(symbols, lookback_days=10, timeframe=TimeFrame.Hour):
    """Fetch OHLCV bars for a mixed list of stock and crypto symbols.
    Returns a combined DataFrame indexed by (symbol, timestamp).
    """
    crypto_syms, stock_syms = _split_symbols(symbols)
    frames = []

    if stock_syms:
        client = _get_stock_client()
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=timeframe,
            start=datetime.utcnow() - timedelta(days=lookback_days),
        )
        frames.append(client.get_stock_bars(req).df)

    if crypto_syms:
        client = _get_crypto_client()
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=timeframe,
            start=datetime.utcnow() - timedelta(days=lookback_days),
        )
        frames.append(client.get_crypto_bars(req).df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)


def compute_indicators(df):
    """Adds SMA-10, SMA-30 and RSI-14 to a single-symbol OHLCV dataframe."""
    df = df.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def get_market_snapshot(symbols, timeframe=TimeFrame.Hour):
    """Returns {symbol: latest indicator row as dict} for prompting the AI."""
    all_bars = fetch_bars(symbols, timeframe=timeframe)
    snapshot = {}
    for sym in symbols:
        try:
            sym_df = all_bars.loc[sym].reset_index()
        except KeyError:
            continue
        sym_df = compute_indicators(sym_df)
        last = sym_df.iloc[-1]
        snapshot[sym] = {
            "close": _smart_round(float(last["close"])),
            "sma_10": _smart_round(float(last["sma_10"])) if pd.notna(last["sma_10"]) else None,
            "sma_30": _smart_round(float(last["sma_30"])) if pd.notna(last["sma_30"]) else None,
            "rsi_14": round(float(last["rsi_14"]), 1) if pd.notna(last["rsi_14"]) else None,
        }
    return snapshot


def get_price_series(symbols, lookback_days=5, timeframe=TimeFrame.Hour):
    """Returns {symbol: list of close prices} for sparkline charts."""
    try:
        all_bars = fetch_bars(symbols, lookback_days=lookback_days, timeframe=timeframe)
    except Exception:
        return {}
    series = {}
    for sym in symbols:
        try:
            sym_df = all_bars.loc[sym].reset_index()
            series[sym] = [round(float(p), 2) for p in sym_df["close"].tolist()]
        except KeyError:
            continue
    return series


def get_mover_stats(symbols, lookback_days=5, timeframe=TimeFrame.Hour):
    """Returns {symbol: {price, change_pct, volume, sparkline}} for market widgets."""
    try:
        all_bars = fetch_bars(symbols, lookback_days=lookback_days, timeframe=timeframe)
    except Exception:
        return {}
    stats = {}
    for sym in symbols:
        try:
            sym_df = all_bars.loc[sym].reset_index()
        except KeyError:
            continue
        if len(sym_df) < 2:
            continue
        closes = sym_df["close"].astype(float)
        current = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else current
        first = float(closes.iloc[0])
        change_pct = ((current - prev) / prev * 100) if prev else 0.0
        change_24h = ((current - first) / first * 100) if first else 0.0
        volume = float(sym_df["volume"].iloc[-1]) if "volume" in sym_df.columns else 0
        stats[sym] = {
            "price": round(current, 2),
            "change_pct": round(change_pct, 2),
            "change_24h_pct": round(change_24h, 2),
            "volume": volume,
            "sparkline": [round(float(p), 2) for p in closes.tolist()[-24:]],
        }
    return stats
