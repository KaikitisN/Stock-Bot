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


def submit_bracket_order(trading_client, symbol, qty, side, stop_price, target_price):
    """Places a market order.
    For stocks: uses a bracket order (stop-loss + take-profit attached).
    For crypto: Alpaca does not support bracket/OTOCO orders, so a plain
    market order is submitted instead (stop/target managed externally).
    """
    order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL

    if _is_crypto(symbol):
        # Crypto: simple market order — Alpaca rejects bracket orders for crypto
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
        )
    else:
        # Stocks: full bracket order with stop-loss and take-profit
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )

    return trading_client.submit_order(order_req)


def get_open_positions(trading_client):
    return trading_client.get_all_positions()
