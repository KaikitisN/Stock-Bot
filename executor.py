"""Submits orders to Alpaca (paper or live, controlled by ALPACA_PAPER in .env)."""
import json
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
import config

# Set when a crypto entry filled but its protective stop-limit was rejected.
LAST_PROTECTION_ERROR = ""


def get_trading_client():
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER)


def symbol_key(symbol: str) -> str:
    """Compact identity so UNI/USD and UNIUSD are the same coin."""
    return (symbol or "").replace("/", "").upper()


def is_crypto_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    if "/" in symbol:
        return True
    key = symbol_key(symbol)
    return key.endswith("USD") and len(key) > 3


def slash_symbol(symbol: str) -> str:
    """Alpaca accepts crypto orders as UNI/USD. Positions often come back as UNIUSD."""
    if not symbol or "/" in symbol:
        return symbol
    key = symbol_key(symbol)
    if key.endswith("USD") and len(key) > 3:
        return f"{key[:-3]}/USD"
    return symbol


def _is_crypto(symbol: str) -> bool:
    return is_crypto_symbol(symbol)


def find_position(trading_client, symbol: str):
    """Match a holding without GET /positions/UNI/USD, which 404s on the slash."""
    want = symbol_key(symbol)
    try:
        positions = trading_client.get_all_positions()
    except Exception:
        return None
    for pos in positions:
        if symbol_key(getattr(pos, "symbol", "")) == want:
            return pos
    return None


def _has_open_position(trading_client, symbol: str) -> bool:
    return find_position(trading_client, symbol) is not None


def get_position_side(trading_client, symbol: str) -> str | None:
    """Returns 'long', 'short', or None if no position exists."""
    pos = find_position(trading_client, symbol)
    if pos is None:
        return None
    try:
        qty = float(pos.qty)
    except (TypeError, ValueError):
        return None
    if qty > 0:
        return "long"
    if qty < 0:
        return "short"
    return None


def list_open_orders(trading_client, symbol: str):
    """Open orders for this coin, or None when the broker cannot be read.

    Filtering the request by UNI/USD misses orders Alpaca stored as UNIUSD,
    so the match is done locally on the compact symbol.
    """
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = trading_client.get_orders(filter=request)
    except Exception:
        return None
    want = symbol_key(symbol)
    return [order for order in orders if symbol_key(getattr(order, "symbol", "")) == want]


def has_pending_order(trading_client, symbol: str) -> bool:
    orders = list_open_orders(trading_client, symbol)
    if not orders:
        return False
    return True


def cancel_open_orders(trading_client, symbol: str):
    orders = list_open_orders(trading_client, symbol) or []
    for order in orders:
        trading_client.cancel_order_by_id(order.id)


def liquidate_position(trading_client, symbol: str):
    """Cancel resting protection, then sell the whole position."""
    pos = find_position(trading_client, symbol)
    if pos is None:
        return None

    cancel_open_orders(trading_client, symbol)

    try:
        qty = abs(float(pos.qty))
    except (TypeError, ValueError):
        qty = 0.0
    if qty <= 0:
        return None

    if _is_crypto(symbol) or _is_crypto(getattr(pos, "symbol", "")):
        order_req = MarketOrderRequest(
            symbol=slash_symbol(symbol),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return trading_client.submit_order(order_req)

    return trading_client.close_position(slash_symbol(getattr(pos, "symbol", symbol)))


def protection_error() -> str:
    return LAST_PROTECTION_ERROR


def load_exit_plans() -> dict:
    path = config.EXIT_PLANS_FILE
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_exit_plans(plans: dict):
    path = config.EXIT_PLANS_FILE
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)


def remember_exit_plan(symbol, qty, entry_price, stop_price, target_price, stop_order_id=""):
    plans = load_exit_plans()
    plans[symbol_key(symbol)] = {
        "symbol": slash_symbol(symbol),
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "stop_order_id": stop_order_id or "",
    }
    save_exit_plans(plans)


def _round_price(symbol: str, price: float) -> float:
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


def stop_limit_price(symbol: str, stop_price: float) -> float:
    """Limit sits below a sell stop so the order can fill after the trigger."""
    buffer_pct = getattr(config, "CRYPTO_STOP_LIMIT_BUFFER_PCT", 1.0)
    return _round_price(symbol, stop_price * (1 - buffer_pct / 100.0))


def submit_crypto_stop_limit(trading_client, symbol, qty, stop_price):
    """GTC stop-limit sell. Crypto rejects plain stop orders and brackets."""
    trigger = _round_price(symbol, stop_price)
    limit_price = stop_limit_price(symbol, trigger)
    if limit_price >= trigger:
        limit_price = _round_price(symbol, trigger - (0.01 if trigger >= 1 else 0.0001))
    stop_req = StopLimitOrderRequest(
        symbol=slash_symbol(symbol),
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=trigger,
        limit_price=limit_price,
    )
    return trading_client.submit_order(stop_req)


def submit_bracket_order(
    trading_client, symbol, qty, side, stop_price, target_price, entry_price=None,
):
    global LAST_PROTECTION_ERROR
    LAST_PROTECTION_ERROR = ""
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
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        return trading_client.submit_order(order_req)

    # BUY path. Crypto only accepts simple orders, so a bracket is never sent.
    # The stop is a separate stop-limit; the target is watched by manage_open_exits.
    if _is_crypto(symbol):
        stop_price = _round_price(symbol, stop_price)
        target_price = _round_price(symbol, target_price)
        buy_req = MarketOrderRequest(
            symbol=slash_symbol(symbol),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        entry_order = trading_client.submit_order(buy_req)
        remember_exit_plan(
            symbol,
            qty,
            entry_price if entry_price is not None else stop_price,
            stop_price,
            target_price,
        )
        try:
            stop_order = submit_crypto_stop_limit(
                trading_client, symbol, qty, stop_price,
            )
            plans = load_exit_plans()
            key = symbol_key(symbol)
            if key in plans:
                plans[key]["stop_order_id"] = str(getattr(stop_order, "id", ""))
                save_exit_plans(plans)
        except Exception as exc:
            LAST_PROTECTION_ERROR = str(exc)
        return entry_order

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


def get_open_positions(trading_client):
    return trading_client.get_all_positions()


def get_portfolio_history(trading_client, period: str = "1M"):
    """Fetch daily equity history from Alpaca (period: 1W, 1M, 3M, 1A, all)."""
    req = GetPortfolioHistoryRequest(period=period, timeframe="1D")
    return trading_client.get_portfolio_history(req)
