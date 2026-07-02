"""
Hard risk rules expressed as % of account equity.
These OVERRIDE the AI — the model can suggest a trade, but this
module decides the final size and can veto or shrink it.
"""


def get_account_equity(trading_client):
    account = trading_client.get_account()
    return float(account.equity), float(account.cash)


def calc_position_size(equity, price, max_position_pct):
    """Returns whole shares capped at max_position_pct of equity."""
    max_dollar_amount = equity * (max_position_pct / 100)
    if price <= 0:
        return 0
    return max(int(max_dollar_amount // price), 0)


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
        stop_price = round(entry_price * (1 + stop_loss_pct / 100), 2)
        target_price = round(entry_price * (1 - take_profit_pct / 100), 2)
    else:
        stop_price = round(entry_price * (1 - stop_loss_pct / 100), 2)
        target_price = round(entry_price * (1 + take_profit_pct / 100), 2)
    return stop_price, target_price
