"""Pulls recent price bars from Alpaca for a list of symbols."""
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import config


def get_client():
    return StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def fetch_bars(symbols, lookback_days=10, timeframe=TimeFrame.Hour):
    client = get_client()
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=datetime.utcnow() - timedelta(days=lookback_days),
    )
    bars = client.get_stock_bars(req).df
    return bars


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
            "close": round(float(last["close"]), 2),
            "sma_10": round(float(last["sma_10"]), 2) if pd.notna(last["sma_10"]) else None,
            "sma_30": round(float(last["sma_30"]), 2) if pd.notna(last["sma_30"]) else None,
            "rsi_14": round(float(last["rsi_14"]), 1) if pd.notna(last["rsi_14"]) else None,
        }
    return snapshot
