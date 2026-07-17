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


def count_open_positions(trading_client) -> int:
    try:
        return len(trading_client.get_all_positions())
    except Exception:
        return 0


def calc_position_size(cash: float, price: float, max_position_pct: float):
    """Returns position size capped at max_position_pct of cash.

    For stocks (price >= $1): returns whole shares (int).
    For crypto with fractional prices (price < $1): returns a fractional
    quantity rounded to 8 decimal places, as Alpaca supports fractional
    crypto orders.
    """
    if price <= 0 or cash <= 0:
        return 0
    max_dollar_amount = cash * (max_position_pct / 100)
    raw_qty = max_dollar_amount / price
    if price >= 1.0:
        return max(int(raw_qty), 0)
    return round(raw_qty, 2)


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
