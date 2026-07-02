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
from risk_manager import get_account_equity, calc_position_size, stop_loss_take_profit_prices
from executor import get_trading_client, submit_bracket_order

os.makedirs(config.LOG_DIR, exist_ok=True)


def log_row(path, row: dict):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_once(symbols, provider_name, use_news, risk_cfg):
    trading_client = get_trading_client()
    equity, cash = get_account_equity(trading_client)
    snapshot = get_market_snapshot(symbols)
    results = []

    for symbol, market_data in snapshot.items():
        decision = get_decision(symbol, market_data, provider_name, use_news)
        decision["timestamp"] = datetime.utcnow().isoformat()
        log_row(config.DECISIONS_LOG, decision)

        if decision["action"] in ("BUY", "SELL") and decision.get("confidence", 0) >= 60:
            price = market_data["close"]
            qty = calc_position_size(equity, price, risk_cfg["max_position_pct"])
            if qty > 0:
                stop_price, target_price = stop_loss_take_profit_prices(
                        price,
                        risk_cfg["stop_loss_pct"],
                        risk_cfg["take_profit_pct"],
                        decision["action"],
                    )
                try:
                    order = submit_bracket_order(
                        trading_client, symbol, qty, decision["action"], stop_price, target_price
                    )
                    trade_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "side": decision["action"],
                        "qty": qty,
                        "entry_price": price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "order_id": str(order.id),
                    }
                    log_row(config.TRADES_LOG, trade_row)
                    decision["trade_submitted"] = True
                except Exception as e:
                    decision["trade_submitted"] = False
                    decision["error"] = str(e)
        results.append(decision)
    return results, equity, cash
