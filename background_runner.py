"""
Standalone trading loop — runs independently of the Streamlit dashboard.
Keeps trading even if no browser is connected. Writes live status to
logs/runner_status.json so the dashboard can display current progress
(including which symbol is being analyzed).
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import config
from data_fetcher import get_market_snapshot
from executor import get_trading_client
from orchestrator import is_stock_market_open, process_symbol
from risk_manager import get_account_equity, is_trading_halted
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_DIR}/background_runner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("background_runner")

PROVIDER_NAME = config.DEFAULT_AI_PROVIDER
USE_NEWS = config.USE_NEWS_DEFAULT
RISK_CFG = config.DEFAULT_RISK
RUN_INTERVAL_MINUTES = config.FREQUENCY_OPTIONS[config.DEFAULT_FREQUENCY]

STATUS_PATH = f"{config.LOG_DIR}/runner_status.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_status(status: dict):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def _base_running_status(total: int, started_at: str) -> dict:
    return {
        "state": "running",
        "cycle_started_at": started_at,
        "current_symbol": None,
        "current_index": 0,
        "total_symbols": total,
        "last_completed_symbol": None,
        "last_result": None,
        "next_run_at": None,
        "interval_minutes": RUN_INTERVAL_MINUTES,
        "progress_pct": 0,
    }


def job() -> bool:
    """Run one trading cycle, updating per-symbol progress for the dashboard."""
    symbols = list(config.DEFAULT_SYMBOLS)
    total = len(symbols)
    started_at = utc_now().isoformat()
    write_status(_base_running_status(total, started_at))
    logger.info(f"Running trading cycle for {total} symbols: {symbols}")

    try:
        trading_client = get_trading_client()
        equity, cash = get_account_equity(trading_client)
        halted, halt_reason = is_trading_halted(
            trading_client, RISK_CFG.get("max_daily_loss_pct", 3.0)
        )
        market_open = is_stock_market_open(trading_client)

        tradable = [s for s in symbols if "/" in s or market_open]
        skipped = [s for s in symbols if s not in tradable]

        results = []
        for symbol in skipped:
            decision = {
                "symbol": symbol,
                "action": "SKIPPED",
                "confidence": 0,
                "reason": "Stock market closed",
                "provider": PROVIDER_NAME,
                "timestamp": datetime.utcnow().isoformat(),
                "trade_submitted": False,
                "error": "",
            }
            results.append(decision)
            logger.info(f"  {symbol}: SKIPPED (market closed)")

        if halted:
            logger.warning(halt_reason)
            for i, symbol in enumerate(tradable, start=1):
                write_status({
                    **_base_running_status(total, started_at),
                    "current_symbol": symbol,
                    "current_index": i,
                    "progress_pct": int(i / max(total, 1) * 100),
                    "last_result": {
                        "symbol": symbol,
                        "action": "HALTED",
                        "confidence": 0,
                        "reason": halt_reason,
                    },
                })
                decision = {
                    "symbol": symbol,
                    "action": "HALTED",
                    "confidence": 0,
                    "reason": halt_reason,
                    "provider": PROVIDER_NAME,
                    "timestamp": datetime.utcnow().isoformat(),
                    "trade_submitted": False,
                    "error": halt_reason,
                }
                results.append(decision)
                logger.info(f"  {symbol}: HALTED")
        else:
            write_status({
                **_base_running_status(total, started_at),
                "current_symbol": "fetching market data…",
                "current_index": 0,
                "progress_pct": 0,
            })
            snapshot = get_market_snapshot(tradable)
            snapshot_items = list(snapshot.items())
            work_total = max(len(snapshot_items), 1)

            for i, (symbol, market_data) in enumerate(snapshot_items, start=1):
                write_status({
                    **_base_running_status(total, started_at),
                    "current_symbol": symbol,
                    "current_index": i,
                    "total_symbols": len(snapshot_items),
                    "progress_pct": int((i - 1) / work_total * 100),
                    "last_completed_symbol": results[-1]["symbol"] if results else None,
                    "last_result": results[-1] if results else None,
                })
                logger.info(f"Analyzing {symbol} ({i}/{len(snapshot_items)})...")

                decision, cash = process_symbol(
                    trading_client,
                    symbol,
                    market_data,
                    PROVIDER_NAME,
                    USE_NEWS,
                    RISK_CFG,
                    cash,
                    trading_halted=halted,
                    halt_reason=halt_reason,
                )
                results.append(decision)
                logger.info(
                    f"  {symbol}: {decision.get('action')} "
                    f"(confidence={decision.get('confidence')}) "
                    f"submitted={decision.get('trade_submitted')}"
                )

                write_status({
                    **_base_running_status(total, started_at),
                    "current_symbol": symbol,
                    "current_index": i,
                    "total_symbols": len(snapshot_items),
                    "progress_pct": int(i / work_total * 100),
                    "last_completed_symbol": symbol,
                    "last_result": decision,
                })

        finished_at = utc_now()
        next_at = finished_at + timedelta(minutes=RUN_INTERVAL_MINUTES)
        # Refresh equity after the cycle
        try:
            equity, cash = get_account_equity(trading_client)
        except Exception:
            pass

        write_status({
            "state": "idle",
            "cycle_started_at": None,
            "current_symbol": None,
            "current_index": total,
            "total_symbols": total,
            "last_completed_symbol": results[-1]["symbol"] if results else None,
            "last_result": results[-1] if results else None,
            "cycle_finished_at": finished_at.isoformat(),
            "next_run_at": next_at.isoformat(),
            "interval_minutes": RUN_INTERVAL_MINUTES,
            "progress_pct": 100,
            "equity": equity,
            "cash": cash,
        })
        logger.info(
            f"Cycle complete. Equity=${equity:,.2f}, Cash=${cash:,.2f}. "
            f"Next run at {next_at.isoformat()}"
        )
        return True
    except Exception as e:
        next_at = utc_now() + timedelta(minutes=RUN_INTERVAL_MINUTES)
        logger.exception(f"Cycle failed: {e}")
        write_status({
            "state": "error",
            "error": str(e),
            "next_run_at": next_at.isoformat(),
            "interval_minutes": RUN_INTERVAL_MINUTES,
            "current_symbol": None,
        })
        return False


def wait_until_next_run(next_run_at: datetime):
    """Sleep until next_run_at. Wakes every 30s so SIGTERM stays responsive."""
    remaining = (next_run_at - utc_now()).total_seconds()
    if remaining > 0:
        logger.info(
            f"Sleeping {int(remaining)}s until next cycle "
            f"({next_run_at.strftime('%Y-%m-%d %H:%M:%S %Z')})"
        )
    while True:
        now = utc_now()
        remaining = (next_run_at - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


if __name__ == "__main__":
    logger.info(
        f"Starting background runner. Interval={RUN_INTERVAL_MINUTES} min, "
        f"Provider={PROVIDER_NAME}, Symbols={len(config.DEFAULT_SYMBOLS)}"
    )
    while True:
        job()
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                status = json.load(f)
            next_run_at = datetime.fromisoformat(status["next_run_at"])
            if next_run_at.tzinfo is None:
                next_run_at = next_run_at.replace(tzinfo=timezone.utc)
        except Exception:
            next_run_at = utc_now() + timedelta(minutes=RUN_INTERVAL_MINUTES)
            write_status({
                "state": "idle",
                "next_run_at": next_run_at.isoformat(),
                "interval_minutes": RUN_INTERVAL_MINUTES,
            })
        wait_until_next_run(next_run_at)
