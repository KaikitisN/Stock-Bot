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


def liquidate_position(trading_client, symbol: str):
    """Fully liquidates a position in one order.

    For micro-priced crypto (e.g. SHIB) with billions of tokens,
    close_position() can hit per-order qty limits and split into many
    partial orders. Instead we submit a single notional SELL for the
    full market value of the position, which Alpaca handles as one order
    regardless of token count.

    For stocks and normal-priced crypto, falls back to close_position().
    Returns None if no position exists.
    """
    try:
        pos = trading_client.get_open_position(symbol)
    except Exception:
        return None  # No position to close

    market_value = abs(float(pos.market_value))

    if _is_crypto(symbol) and market_value > 0:
        # Use notional sell: sell exactly $market_value worth in one order
        order_req = MarketOrderRequest(
            symbol=symbol,
            notional=round(market_value, 2),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return trading_client.submit_order(order_req)

    # Stocks or zero-value positions: standard close
    return trading_client.close_position(symbol)


def submit_bracket_order(trading_client, symbol, qty, side, stop_price, target_price):
    """Handles order submission per asset type and side.

    BUY:
      - Crypto: plain GTC market order (no bracket support on Alpaca for crypto).
      - Stock:  bracket order with stop-loss + take-profit.

    SELL (both crypto and stocks):
      - Uses liquidate_position() which handles large qty via notional.
      - Returns None silently if no position exists.
      - Raises on any other error so the dashboard can display it.
    """
    if side == "SELL":
        return liquidate_position(trading_client, symbol)

    # BUY path
    if _is_crypto(symbol):
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
    else:
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
