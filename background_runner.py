"""
Standalone trading loop — runs independently of the Streamlit dashboard.
Keeps trading even if no browser is connected. Writes live status to
logs/runner_status.json so the dashboard can display current progress.
"""
import json
import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

import config
from orchestrator import run_once

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_DIR}/background_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("background_runner")

SYMBOLS = config.DEFAULT_SYMBOLS
PROVIDER_NAME = config.DEFAULT_AI_PROVIDER
USE_NEWS = config.USE_NEWS_DEFAULT
RISK_CFG = config.DEFAULT_RISK
RUN_INTERVAL_MINUTES = config.FREQUENCY_OPTIONS[config.DEFAULT_FREQUENCY]

STATUS_PATH = f"{config.LOG_DIR}/runner_status.json"


def write_status(status: dict):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def job():
    total = len(SYMBOLS)
    write_status({
        "state": "running",
        "cycle_started_at": datetime.now(timezone.utc).isoformat(),
        "current_symbol": None,
        "current_index": 0,
        "total_symbols": total,
        "last_completed_symbol": None,
        "last_result": None,
    })
    logger.info(f"Running trading cycle for {total} symbols: {SYMBOLS}")

    try:
        results, equity, cash = run_once(SYMBOLS, PROVIDER_NAME, USE_NEWS, RISK_CFG)
        for r in results:
            logger.info(
                f"  {r.get('symbol')}: {r.get('action')} "
                f"(confidence={r.get('confidence')}) submitted={r.get('trade_submitted')}"
            )

        write_status({
            "state": "idle",
            "cycle_started_at": None,
            "current_symbol": None,
            "current_index": total,
            "total_symbols": total,
            "last_completed_symbol": results[-1]["symbol"] if results else None,
            "last_result": results[-1] if results else None,
            "cycle_finished_at": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "cash": cash,
        })
        logger.info(f"Cycle complete. Equity=${equity:,.2f}, Cash=${cash:,.2f}")
    except Exception as e:
        logger.error(f"Cycle failed: {e}")
        write_status({"state": "error", "error": str(e)})


if __name__ == "__main__":
    logger.info(
        f"Starting background runner. Interval={RUN_INTERVAL_MINUTES} min, "
        f"Provider={PROVIDER_NAME}, Symbols={len(SYMBOLS)}"
    )
    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", minutes=RUN_INTERVAL_MINUTES, next_run_time=None)
    job()
    scheduler.start()
