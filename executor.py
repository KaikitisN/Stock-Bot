"""Submits orders to Alpaca (paper or live, controlled by ALPACA_PAPER in .env)."""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    StopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
import config


def get_trading_client():
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER)


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


def _has_open_position(trading_client, symbol: str) -> bool:
    try:
        pos = trading_client.get_open_position(symbol)
        return pos is not None
    except Exception:
        return False


def get_position_side(trading_client, symbol: str) -> str | None:
    """Returns 'long', 'short', or None if no position exists."""
    try:
        pos = trading_client.get_open_position(symbol)
        qty = float(pos.qty)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return None
    except Exception:
        return None


def has_pending_order(trading_client, symbol: str) -> bool:
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = trading_client.get_orders(filter=request)
        return len(orders) > 0
    except Exception:
        return False


def liquidate_position(trading_client, symbol: str):
    """Fully liquidates a position in one order."""
    try:
        pos = trading_client.get_open_position(symbol)
    except Exception:
        return None

    market_value = abs(float(pos.market_value))

    if _is_crypto(symbol) and market_value > 0:
        order_req = MarketOrderRequest(
            symbol=symbol,
            notional=round(market_value, 2),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return trading_client.submit_order(order_req)

    return trading_client.close_position(symbol)


def _round_price(symbol: str, price: float) -> float:
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


def _submit_crypto_stop_loss(trading_client, symbol, qty, stop_price):
    """Attach a GTC stop-loss after a crypto market buy."""
    stop_req = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=_round_price(symbol, stop_price),
    )
    return trading_client.submit_order(stop_req)


def submit_bracket_order(trading_client, symbol, qty, side, stop_price, target_price):
    if not _is_crypto(symbol):
        stop_price = _round_price(symbol, stop_price)
        target_price = _round_price(symbol, target_price)

    side = side.upper()

    if side == "SELL":
        if _has_open_position(trading_client, symbol):
            return liquidate_position(trading_client, symbol)

        if _is_crypto(symbol):
            raise ValueError(
                f"Cannot sell {symbol}: no open position and crypto shorting is not supported."
            )

        if not config.ALLOW_SHORT_SELLING:
            raise ValueError(
                f"SELL signal for {symbol} ignored: no open position to close "
                f"(short selling disabled)."
            )

        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        return trading_client.submit_order(order_req)

    # BUY path
    if _is_crypto(symbol):
        stop_price = _round_price(symbol, stop_price)
        target_price = _round_price(symbol, target_price)
        try:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=target_price),
            )
            return trading_client.submit_order(order_req)
        except Exception:
            buy_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            entry_order = trading_client.submit_order(buy_req)
            try:
                _submit_crypto_stop_loss(trading_client, symbol, qty, stop_price)
            except Exception:
                pass
            return entry_order

    order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        stop_loss=StopLossRequest(stop_price=stop_price),
        take_profit=TakeProfitRequest(limit_price=target_price),
    )
    return trading_client.submit_order(order_req)


def get_open_positions(trading_client):
    return trading_client.get_all_positions()
