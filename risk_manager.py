"""
Hard risk rules expressed as % of account equity.
These OVERRIDE the AI — the model can suggest a trade, but this
module decides the final size and can veto or shrink it.
"""


def get_account_equity(trading_client):
    account = trading_client.get_account()
    return float(account.equity), float(account.cash)


def calc_position_size(cash: float, price: float, max_position_pct: float):
    """Returns position size capped at max_position_pct of cash.

    For stocks (price >= $1): returns whole shares (int).
    For crypto with fractional prices (price < $1): returns a fractional
    quantity rounded to 8 decimal places, as Alpaca supports fractional
    crypto orders. This prevents micro-priced assets like SHIB from
    being skipped because int() rounds the huge qty to 0 unexpectedly,
    or because the price rounded to $0.00 at 2dp.
    """
    if price <= 0 or cash <= 0:
        return 0
    max_dollar_amount = cash * (max_position_pct / 100)
    raw_qty = max_dollar_amount / price
    if price >= 1.0:
        # Stocks and high-value crypto: whole units only
        return max(int(raw_qty), 0)
    else:
        # Micro-priced crypto (SHIB, DOGE, etc): fractional units, 2dp
        return round(raw_qty, 2)


def check_daily_loss_limit(trading_client, day_start_equity, max_daily_loss_pct):
    """Returns True if trading should HALT for the day."""
    account = trading_client.get_account()
    current_equity = float(account.equity)
    if day_start_equity <= 0:
        return False
    drawdown_pct = (day_start_equity - current_equity) / day_start_equity * 100
    return drawdown_pct >= max_daily_loss_pct


def stop_loss_take_profit_prices(entry_price, stop_loss_pct, take_profit_pct, side="BUY"):
    side = side.upper()
    if side == "SELL":
        stop_price = round(entry_price * (1 + stop_loss_pct / 100), 8)
        target_price = round(entry_price * (1 - take_profit_pct / 100), 8)
    else:
        stop_price = round(entry_price * (1 - stop_loss_pct / 100), 8)
        target_price = round(entry_price * (1 + take_profit_pct / 100), 8)
    return stop_price, target_price
