"""
Main decision-and-execution loop. Runs once per call; the dashboard
schedules repeated calls at the interval you choose.
"""
import csv
import os
from datetime import datetime

import config
from data_fetcher import get_market_snapshot
from ai_decision import get_decision
from risk_manager import (
    get_account_equity,
    calc_position_size,
    stop_loss_take_profit_prices,
    is_trading_halted,
    passes_trend_filter,
    count_open_positions,
)
from executor import (
    get_trading_client,
    submit_bracket_order,
    get_position_side,
    has_pending_order,
)

os.makedirs(config.LOG_DIR, exist_ok=True)


def is_stock_market_open(trading_client) -> bool:
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception:
        return False


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


def log_row(path, row: dict):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_symbol(
    trading_client,
    symbol,
    market_data,
    provider_name,
    use_news,
    risk_cfg,
    cash,
    *,
    trading_halted=False,
    halt_reason="",
):
    """
    Evaluate one symbol and optionally submit a trade.
    Returns (decision_dict, cash_remaining).
    """
    decision = get_decision(symbol, market_data, provider_name, use_news)
    decision["timestamp"] = datetime.utcnow().isoformat()
    decision["trade_submitted"] = False
    decision["error"] = ""

    min_confidence = getattr(config, "MIN_TRADE_CONFIDENCE", 70)
    action = decision.get("action", "HOLD").upper()
    confidence = decision.get("confidence", 0)

    if action not in ("BUY", "SELL") or confidence < min_confidence:
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    if trading_halted:
        decision["error"] = halt_reason
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    trend_ok, trend_reason = passes_trend_filter(action, market_data)
    if not trend_ok:
        decision["error"] = trend_reason
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    current_side = get_position_side(trading_client, symbol)
    wants_long = action == "BUY"
    wants_short = action == "SELL"

    if (wants_long and current_side == "long") or (wants_short and current_side == "short"):
        decision["error"] = (
            f"Skipped: already holding a {current_side} position in {symbol}."
        )
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    if has_pending_order(trading_client, symbol):
        decision["error"] = (
            f"Skipped: an order for {symbol} is already pending."
        )
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    if action == "BUY" and count_open_positions(trading_client) >= config.MAX_OPEN_POSITIONS:
        decision["error"] = (
            f"Skipped: max open positions ({config.MAX_OPEN_POSITIONS}) reached."
        )
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    # Only skip a SELL with no position if short selling is disabled.
    # When ALLOW_SHORT_SELLING=true, let executor.py handle the short order.
    if action == "SELL" and current_side is None and not config.ALLOW_SHORT_SELLING:
        decision["error"] = "Skipped: SELL signal but no position to close (short selling disabled)."
        log_row(config.DECISIONS_LOG, decision)
        return decision, cash

    price = market_data["close"]

    if action == "BUY":
        qty = calc_position_size(cash, price, risk_cfg["max_position_pct"])
        cost = qty * price
        if qty <= 0 or cost > cash:
            decision["error"] = (
                f"Insufficient cash (need ${cost:.2f}, have ${cash:.2f})"
            )
            log_row(config.DECISIONS_LOG, decision)
            return decision, cash
    else:
        qty = 1

    stop_price, target_price = stop_loss_take_profit_prices(
        price,
        risk_cfg["stop_loss_pct"],
        risk_cfg["take_profit_pct"],
        action,
    )

    try:
        order = submit_bracket_order(
            trading_client, symbol, qty, action, stop_price, target_price
        )
        if order is not None:
            trade_row = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": action,
                "qty": qty,
                "entry_price": price,
                "stop_price": stop_price,
                "target_price": target_price,
                "order_id": str(order.id),
            }
            log_row(config.TRADES_LOG, trade_row)
            if action == "BUY":
                cash -= qty * price
        decision["trade_submitted"] = order is not None
    except Exception as e:
        decision["trade_submitted"] = False
        decision["error"] = str(e)

    log_row(config.DECISIONS_LOG, decision)
    return decision, cash


def run_once(symbols, provider_name, use_news, risk_cfg):
    trading_client = get_trading_client()
    equity, cash = get_account_equity(trading_client)
    market_open = is_stock_market_open(trading_client)

    halted, halt_reason = is_trading_halted(
        trading_client, risk_cfg.get("max_daily_loss_pct", config.DEFAULT_RISK["max_daily_loss_pct"])
    )

    tradable_symbols = [s for s in symbols if _is_crypto(s) or market_open]
    skipped = [s for s in symbols if s not in tradable_symbols]

    snapshot = get_market_snapshot(tradable_symbols)
    results = []

    for symbol in skipped:
        results.append({
            "symbol": symbol,
            "action": "SKIPPED",
            "confidence": 0,
            "reason": "Stock market closed",
            "provider": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "trade_submitted": False,
            "error": "",
        })

    if halted:
        for symbol in tradable_symbols:
            if symbol not in snapshot:
                continue
            results.append({
                "symbol": symbol,
                "action": "HALTED",
                "confidence": 0,
                "reason": halt_reason,
                "provider": provider_name,
                "timestamp": datetime.utcnow().isoformat(),
                "trade_submitted": False,
                "error": halt_reason,
            })
        return results, equity, cash

    for symbol, market_data in snapshot.items():
        decision, cash = process_symbol(
            trading_client,
            symbol,
            market_data,
            provider_name,
            use_news,
            risk_cfg,
            cash,
            trading_halted=halted,
            halt_reason=halt_reason,
        )
        results.append(decision)

    return results, equity, cash
