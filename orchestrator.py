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
    get_portfolio_state,
    stop_target_for,
    is_trading_halted,
    passes_trend_filter,
    count_open_positions,
)
from portfolio_allocator import allocate, exposure_budget
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


DECISION_FIELDS = [
    "timestamp", "symbol", "action", "confidence", "reason", "provider",
    "mu_pct", "sigma_pct", "conviction", "p_up",
    "qty", "notional", "weight_pct", "trade_submitted", "error",
]


def _rotate_legacy_log(path: str):
    """Move aside a decisions log whose header predates DECISION_FIELDS.

    Appending new columns to an old file would misalign every historical row,
    and the dashboard reads it with on_bad_lines="skip".
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    if header == ",".join(DECISION_FIELDS):
        return
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base, ext = os.path.splitext(path)
    os.replace(path, f"{base}_legacy_{stamp}{ext}")


def log_decision(row: dict):
    """Append a decision using a stable column set, ignoring extra keys."""
    _rotate_legacy_log(config.DECISIONS_LOG)
    file_exists = os.path.isfile(config.DECISIONS_LOG)
    with open(config.DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=DECISION_FIELDS, extrasaction="ignore", restval="",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_symbol(
    trading_client,
    symbol,
    market_data,
    provider_name,
    use_news,
    *,
    trading_halted=False,
    halt_reason="",
):
    """Score one symbol and apply every veto. Submits nothing.

    Returns (decision, candidate). candidate is None when the symbol is not
    tradable this cycle; the decision has already been logged in that case.
    """
    decision = get_decision(symbol, market_data, provider_name, use_news)
    decision["timestamp"] = datetime.utcnow().isoformat()
    decision["trade_submitted"] = False
    decision["error"] = ""

    min_confidence = getattr(config, "MIN_TRADE_CONFIDENCE", 70)
    action = decision.get("action", "HOLD").upper()
    confidence = decision.get("confidence", 0)

    def veto(reason):
        decision["error"] = reason
        log_decision(decision)
        return decision, None

    if action not in ("BUY", "SELL") or confidence < min_confidence:
        log_decision(decision)
        return decision, None

    if trading_halted:
        return veto(halt_reason)

    trend_ok, trend_reason = passes_trend_filter(action, market_data)
    if not trend_ok:
        return veto(trend_reason)

    current_side = get_position_side(trading_client, symbol)
    if (action == "BUY" and current_side == "long") or (
        action == "SELL" and current_side == "short"
    ):
        return veto(f"Skipped: already holding a {current_side} position in {symbol}.")

    if has_pending_order(trading_client, symbol):
        return veto(f"Skipped: an order for {symbol} is already pending.")

    if action == "SELL" and current_side is None and not config.ALLOW_SHORT_SELLING:
        return veto(
            "Skipped: SELL signal but no position to close (short selling disabled)."
        )

    # Closing an existing position is a full liquidation, so it bypasses sizing
    # and the exposure budget entirely.
    closing = action == "SELL" and current_side == "long"

    candidate = {
        "symbol": symbol,
        "action": action,
        "price": market_data["close"],
        "atr": market_data.get("atr_14"),
        "conviction": float(decision.get("conviction", 0.0) or 0.0),
        "closing": closing,
        "decision": decision,
    }
    return decision, candidate


def execute_plan(trading_client, plan, risk_cfg, sizing):
    """Submit one funded plan (or one full liquidation) and log the outcome."""
    decision = plan["decision"]
    symbol = plan["symbol"]
    action = plan["action"]
    price = plan["price"]
    qty = plan.get("qty", 0.0)

    stop_price, target_price = stop_target_for(
        price, plan.get("atr"), sizing, risk_cfg, action,
    )

    try:
        order = submit_bracket_order(
            trading_client, symbol, qty, action, stop_price, target_price
        )
        decision["trade_submitted"] = order is not None
        if order is not None:
            decision["qty"] = qty
            decision["notional"] = round(qty * price, 2)
            decision["weight_pct"] = round(plan.get("weight_pct", 0.0), 3)
            log_row(config.TRADES_LOG, {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": action,
                "qty": qty,
                "entry_price": price,
                "stop_price": stop_price,
                "target_price": target_price,
                "order_id": str(order.id),
            })
    except Exception as e:
        decision["trade_submitted"] = False
        decision["error"] = str(e)

    log_decision(decision)
    return decision


def run_once(symbols, provider_name, use_news, risk_cfg):
    trading_client = get_trading_client()
    sizing = config.SIZING
    market_open = is_stock_market_open(trading_client)

    halted, halt_reason = is_trading_halted(
        trading_client,
        risk_cfg.get("max_daily_loss_pct", config.DEFAULT_RISK["max_daily_loss_pct"]),
    )

    tradable_symbols = [s for s in symbols if _is_crypto(s) or market_open]
    results = []

    for symbol in symbols:
        if symbol in tradable_symbols:
            continue
        results.append({
            "symbol": symbol, "action": "SKIPPED", "confidence": 0,
            "reason": "Stock market closed", "provider": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "trade_submitted": False, "error": "",
        })

    snapshot = get_market_snapshot(tradable_symbols)

    # Phase 1: score everything, submit nothing.
    candidates = []
    for symbol, market_data in snapshot.items():
        decision, candidate = evaluate_symbol(
            trading_client, symbol, market_data, provider_name, use_news,
            trading_halted=halted, halt_reason=halt_reason,
        )
        results.append(decision)
        if candidate is not None:
            candidates.append(candidate)

    # Phase 2: liquidations first (they free capital), then ranked entries.
    closing = [c for c in candidates if c["closing"]]
    entries = [c for c in candidates if not c["closing"]]

    for plan in closing:
        execute_plan(trading_client, plan, risk_cfg, sizing)

    if entries:
        state = get_portfolio_state(trading_client)
        budget = exposure_budget(
            equity=state["equity"],
            long_market_value=state["long_market_value"],
            short_market_value=state["short_market_value"],
            cash=state["cash"],
            sizing=sizing,
        )
        plans = allocate(
            entries,
            equity=state["equity"],
            cash=state["cash"],
            budget=budget,
            open_positions=count_open_positions(trading_client),
            max_open_positions=config.MAX_OPEN_POSITIONS,
            sizing=sizing,
        )
        for plan in plans:
            if plan["funded"]:
                execute_plan(trading_client, plan, risk_cfg, sizing)
            else:
                decision = plan["decision"]
                decision["error"] = plan["rejected_reason"]
                log_decision(decision)

    equity, cash = get_account_equity(trading_client)
    return results, equity, cash
