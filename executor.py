"""Submits orders to Alpaca (paper or live, controlled by ALPACA_PAPER in .env)."""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
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


def _has_pending_order(trading_client, symbol: str) -> bool:
    """Returns True if there's already an open/pending order for this symbol."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = trading_client.get_orders(filter=request)
        return len(orders) > 0
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

def _round_price(symbol: str, price: float) -> float:
    """Alpaca requires whole-cent increments for stocks priced >= $1,
    and up to 4 decimal places for sub-$1 stocks. Crypto has its own rules."""
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)

def submit_bracket_order(trading_client, symbol, qty, side, stop_price, target_price):
    if not _is_crypto(symbol):
        stop_price = _round_price(symbol, stop_price)
        target_price = _round_price(symbol, target_price)

    if side == "SELL":
        if _has_open_position(trading_client, symbol):
            return liquidate_position(trading_client, symbol)

        if _is_crypto(symbol):
            raise ValueError(
                f"Cannot short {symbol}: Alpaca does not support short-selling crypto."
            )

        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        return trading_client.submit_order(order_req)

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
