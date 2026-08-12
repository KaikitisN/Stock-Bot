"""
Hard risk rules expressed as % of account equity.
These OVERRIDE the AI — the model can suggest a trade, but this
module decides the final size and can veto or shrink it.
"""
import json
import os
from datetime import datetime, timezone

import config


def get_account_equity(trading_client):
    account = trading_client.get_account()
    return float(account.equity), float(account.cash)


def get_account_summary(trading_client) -> dict:
    account = trading_client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "last_equity": float(getattr(account, "last_equity", account.equity)),
    }


def get_portfolio_state(trading_client) -> dict:
    """Equity, cash and current market exposure on both sides of the book."""
    account = trading_client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "long_market_value": float(getattr(account, "long_market_value", 0.0) or 0.0),
        "short_market_value": float(getattr(account, "short_market_value", 0.0) or 0.0),
    }


def count_open_positions(trading_client) -> int:
    try:
        return len(trading_client.get_all_positions())
    except Exception:
        return 0


def check_daily_loss_limit(trading_client, day_start_equity, max_daily_loss_pct):
    """Returns True if trading should HALT for the day."""
    account = trading_client.get_account()
    current_equity = float(account.equity)
    if day_start_equity <= 0:
        return False
    drawdown_pct = (day_start_equity - current_equity) / day_start_equity * 100
    return drawdown_pct >= max_daily_loss_pct


def get_day_start_equity(trading_client) -> float:
    """Track equity at the start of each UTC day for daily loss limits."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()

    if os.path.isfile(config.DAY_STATE_FILE):
        try:
            with open(config.DAY_STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("date") == today:
                return float(state["equity"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    _, _ = get_account_equity(trading_client)
    account = trading_client.get_account()
    equity = float(account.equity)
    with open(config.DAY_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"date": today, "equity": equity}, f)
    return equity


def is_trading_halted(trading_client, max_daily_loss_pct: float) -> tuple[bool, str]:
    """Returns (halted, reason)."""
    day_start = get_day_start_equity(trading_client)
    if check_daily_loss_limit(trading_client, day_start, max_daily_loss_pct):
        account = trading_client.get_account()
        current = float(account.equity)
        drawdown = (day_start - current) / day_start * 100 if day_start > 0 else 0
        return True, (
            f"Daily loss limit hit ({drawdown:.1f}% drawdown, "
            f"limit {max_daily_loss_pct}%). Trading halted for today."
        )
    return False, ""


def passes_trend_filter(action: str, market_data: dict) -> tuple[bool, str]:
    """Require signals to align with the prevailing trend."""
    if not config.USE_TREND_FILTER:
        return True, ""

    close = market_data.get("close")
    sma_30 = market_data.get("sma_30")
    rsi = market_data.get("rsi_14")

    if close is None or sma_30 is None:
        return True, ""

    action = action.upper()
    if action == "BUY":
        if close <= sma_30:
            return False, f"Trend filter: price ${close} below SMA-30 ${sma_30:.2f}"
        if rsi is not None and rsi > 70:
            return False, f"Trend filter: RSI {rsi} overbought (>70)"
    elif action == "SELL":
        if close >= sma_30:
            return False, f"Trend filter: price ${close} above SMA-30 ${sma_30:.2f}"
        if rsi is not None and rsi < 30:
            return False, f"Trend filter: RSI {rsi} oversold (<30)"

    return True, ""


def stop_loss_take_profit_prices(entry_price, stop_loss_pct, take_profit_pct, side="BUY"):
    side = side.upper()
    if side == "SELL":
        stop_price = round(entry_price * (1 + stop_loss_pct / 100), 8)
        target_price = round(entry_price * (1 - take_profit_pct / 100), 8)
    else:
        stop_price = round(entry_price * (1 - stop_loss_pct / 100), 8)
        target_price = round(entry_price * (1 + take_profit_pct / 100), 8)
    return stop_price, target_price


def atr_stop_take_profit_prices(entry_price, atr, sizing: dict, side="BUY"):
    """Volatility-scaled stop and target. Returns None when ATR is unusable.

    A flat percentage stop is simultaneously too tight for crypto and too loose
    for low-volatility names; ATR adapts the distance to how much the symbol
    actually moves.
    """
    if not atr or atr <= 0 or entry_price <= 0:
        return None

    stop_distance = sizing["atr_stop_multiple"] * atr
    target_distance = sizing["atr_target_multiple"] * atr

    if stop_distance >= entry_price:
        return None

    side = side.upper()
    if side == "SELL":
        stop_price = round(entry_price + stop_distance, 8)
        target_price = round(entry_price - target_distance, 8)
        if target_price <= 0:
            return None
    else:
        stop_price = round(entry_price - stop_distance, 8)
        target_price = round(entry_price + target_distance, 8)

    return stop_price, target_price


def stop_target_for(entry_price, atr, sizing: dict, risk_cfg: dict, side="BUY"):
    """ATR-based stop and target, falling back to fixed percentages."""
    prices = atr_stop_take_profit_prices(entry_price, atr, sizing, side)
    if prices is not None:
        return prices
    return stop_loss_take_profit_prices(
        entry_price,
        risk_cfg["stop_loss_pct"],
        risk_cfg["take_profit_pct"],
        side,
    )
