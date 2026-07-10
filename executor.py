"""Submits orders to Alpaca (paper or live, controlled by ALPACA_PAPER in .env)."""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config


def get_trading_client():
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER)


def _is_crypto(symbol: str) -> bool:
    """Returns True if the symbol is a crypto pair (e.g. BTC/USD, ETH/USD)."""
    return "/" in symbol


def _has_open_position(trading_client, symbol: str) -> bool:
    """Returns True if there is an open position for the given symbol."""
    try:
        pos = trading_client.get_open_position(symbol)
        return pos is not None
    except Exception:
        return False


def submit_bracket_order(trading_client, symbol, qty, side, stop_price, target_price):
    """Handles order submission per asset type and side.

    - Crypto BUY/SELL: plain market order (Alpaca does not support bracket/OTOCO for crypto).
    - Stock BUY: bracket order with stop-loss + take-profit attached.
    - Stock SELL: closes the existing long position via close_position().
      Raises the original exception if the close fails so the caller can log it.
    """
    if _is_crypto(symbol):
        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
        )
        return trading_client.submit_order(order_req)

    if side == "SELL":
        # Close the existing long position.
        # Do NOT catch exceptions here — let them bubble up to the dashboard
        # so the error is visible instead of silently swallowed.
        if not _has_open_position(trading_client, symbol):
            # Nothing to close — skip gracefully.
            return None
        return trading_client.close_position(symbol)

    # Stock BUY: bracket order with stop-loss and take-profit
    order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class="bracket",
        stop_loss=StopLossRequest(stop_price=stop_price),
        take_profit=TakeProfitRequest(limit_price=target_price),
    )
    return trading_client.submit_order(order_req)


def get_open_positions(trading_client):
    return trading_client.get_all_positions()
