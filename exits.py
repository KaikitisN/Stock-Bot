"""Watch open crypto positions and give each one a real exit.

Alpaca will not attach a bracket to a crypto order, and it will not hold a
stop and a take-profit on the same coins. Each long therefore gets one
stop-limit at the broker. This module sells at the target, and it sells at
the stop when that stop-limit is missing or can no longer fill.
"""
import math

from executor import (
    is_crypto_symbol,
    list_open_orders,
    liquidate_position,
    load_exit_plans,
    save_exit_plans,
    slash_symbol,
    stop_limit_price,
    submit_crypto_stop_limit,
    symbol_key,
)
from risk_manager import stop_target_for


def _enum_value(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _qty(obj) -> float:
    try:
        qty = abs(float(obj.qty))
    except (TypeError, ValueError, AttributeError):
        return 0.0
    if not math.isfinite(qty):
        return 0.0
    return qty


def _is_sell_stop(order) -> bool:
    side = _enum_value(getattr(order, "side", ""))
    kind = _enum_value(getattr(order, "type", ""))
    return side == "sell" and kind in {"stop", "stop_limit"}


def _covers(order, qty: float) -> bool:
    order_qty = _qty(order)
    if qty <= 0 or order_qty <= 0:
        return False
    return abs(order_qty - qty) / qty <= 0.01


def _limit_of(order) -> float:
    try:
        limit = float(order.limit_price)
    except (TypeError, ValueError, AttributeError):
        return float("inf")
    if not math.isfinite(limit):
        return float("inf")
    return limit


def _quote_for(quotes: dict, symbol: str):
    want = symbol_key(symbol)
    for key, value in (quotes or {}).items():
        if symbol_key(str(key)) == want:
            return value
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mark_price(pos, quote):
    if quote is not None:
        price = _number(quote.get("close"))
        if price is not None and price > 0:
            return price
    price = _number(getattr(pos, "current_price", None))
    if price is not None and price > 0:
        return price
    return None


def _atr(quote):
    if not quote:
        return None
    atr = _number(quote.get("atr_14"))
    if atr is None or atr <= 0:
        return None
    return atr


def _sell(trading_client, symbol: str):
    try:
        order = liquidate_position(trading_client, symbol)
    except Exception as exc:
        return False, str(exc), ""
    if order is None:
        return False, "no position to sell", ""
    return True, "", str(getattr(order, "id", ""))


def _action(symbol, action, reason, qty, price, plan, submitted, order_id=""):
    return {
        "symbol": symbol,
        "action": action,
        "reason": reason,
        "qty": qty,
        "price": price,
        "stop_price": plan.get("stop_price", ""),
        "target_price": plan.get("target_price", ""),
        "submitted": submitted,
        "order_id": order_id,
    }


def manage_open_exits(trading_client, quotes, sizing, risk_cfg) -> list[dict]:
    """Arm, hold, or close every open crypto long. Returns the actions taken."""
    plans = load_exit_plans()
    actions = []
    try:
        positions = list(trading_client.get_all_positions())
    except Exception as exc:
        return [_action("", "HOLD", f"Could not read positions: {exc}", "", "", {}, False)]

    seen = set()
    for pos in positions:
        symbol = getattr(pos, "symbol", "")
        if not is_crypto_symbol(symbol):
            continue
        qty = _qty(pos)
        if qty <= 0:
            continue

        key = symbol_key(symbol)
        seen.add(key)
        quote = _quote_for(quotes, symbol)
        price = _mark_price(pos, quote)
        plan = plans.get(key)
        if plan is None:
            entry = _number(getattr(pos, "avg_entry_price", None))
            if entry is None or entry <= 0:
                continue
            stop_price, target_price = stop_target_for(
                entry, _atr(quote), sizing, risk_cfg, "BUY",
            )
            plan = {
                "symbol": slash_symbol(symbol),
                "qty": qty,
                "entry_price": entry,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_order_id": "",
            }
            plans[key] = plan

        plan["qty"] = qty
        plan["symbol"] = slash_symbol(plan.get("symbol") or symbol)
        display = plan["symbol"]
        stop_price = float(plan["stop_price"])
        target_price = float(plan["target_price"])

        orders = list_open_orders(trading_client, symbol)
        if orders is None:
            actions.append(_action(
                display, "HOLD",
                f"Could not read open orders for {display}; left the position unchanged.",
                qty, price, plan, False,
            ))
            continue

        stops = [order for order in orders if _is_sell_stop(order)]
        covering = next((order for order in stops if _covers(order, qty)), None)

        if price is not None and price >= target_price:
            ok, detail, order_id = _sell(trading_client, display)
            actions.append(_action(
                display, "SELL",
                f"Target {target_price} reached at {price}. Sold {qty} {display}."
                if ok else f"Target {target_price} reached at {price}, sell failed: {detail}",
                qty, price, plan, ok, order_id,
            ))
            if ok:
                plans.pop(key, None)
            continue

        if price is not None and price <= stop_price:
            if covering is not None and price > _limit_of(covering):
                plan["stop_order_id"] = str(getattr(covering, "id", ""))
                continue
            ok, detail, order_id = _sell(trading_client, display)
            actions.append(_action(
                display, "SELL",
                f"Stop {stop_price} breached at {price} with no working stop-limit. Sold {qty} {display}."
                if ok else f"Stop {stop_price} breached at {price}, sell failed: {detail}",
                qty, price, plan, ok, order_id,
            ))
            if ok:
                plans.pop(key, None)
            continue

        if covering is not None:
            plan["stop_order_id"] = str(getattr(covering, "id", ""))
            continue

        if stops:
            for order in stops:
                trading_client.cancel_order_by_id(order.id)
        try:
            stop_order = submit_crypto_stop_limit(
                trading_client, display, qty, stop_price,
            )
            plan["stop_order_id"] = str(getattr(stop_order, "id", ""))
            limit = stop_limit_price(display, stop_price)
            actions.append(_action(
                display, "STOP",
                f"Stop-limit for {qty} {display} placed at {stop_price} (limit {limit}).",
                qty, price, plan, True, plan["stop_order_id"],
            ))
        except Exception as exc:
            actions.append(_action(
                display, "STOP",
                f"Stop-limit for {display} was rejected: {exc}",
                qty, price, plan, False,
            ))

    for key in list(plans):
        if key not in seen:
            plans.pop(key, None)
    save_exit_plans(plans)
    return actions
