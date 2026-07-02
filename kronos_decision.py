"""
Kronos local inference module.
A foundation model trained on 1.2 billion OHLCV candlesticks from 45 exchanges.
Runs entirely on your machine — zero API cost, zero internet calls.

Prerequisites:
    git clone https://github.com/shiyu-coder/Kronos.git  (sibling folder)
    pip install -r Kronos/requirements.txt
    Set KRONOS_REPO_PATH in .env (default: ../Kronos)
    Set KRONOS_MODEL_SIZE in .env: mini | small | base | large (default: mini)
"""
import sys
import os
import pandas as pd
import numpy as np

KRONOS_REPO_PATH = os.getenv("KRONOS_REPO_PATH", "../Kronos")
KRONOS_MODEL_SIZE = os.getenv("KRONOS_MODEL_SIZE", "mini")

_predictor = None  # singleton — loaded once, reused every cycle


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    if not os.path.isdir(KRONOS_REPO_PATH):
        raise FileNotFoundError(
            f"Kronos repo not found at '{KRONOS_REPO_PATH}'.\n"
            "Run: git clone https://github.com/shiyu-coder/Kronos.git\n"
            "Then set KRONOS_REPO_PATH in your .env file."
        )

    sys.path.insert(0, os.path.abspath(KRONOS_REPO_PATH))

    try:
        import torch
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        raise ImportError(
            f"Kronos dependencies missing: {e}\n"
            f"Run: pip install -r {KRONOS_REPO_PATH}/requirements.txt"
        )

    model_map = {
        "mini":  "NeoQuasar/Kronos-mini",
        "small": "NeoQuasar/Kronos-small",
        "base":  "NeoQuasar/Kronos-base",
        "large": "NeoQuasar/Kronos-large",
    }
    hub_name = model_map.get(KRONOS_MODEL_SIZE, "NeoQuasar/Kronos-mini")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Kronos] Loading {hub_name} on {device}...")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(hub_name)
    _predictor = KronosPredictor(model, tokenizer, device=device)
    print(f"[Kronos] Model ready.")
    return _predictor


def get_kronos_decision(symbol: str, bars_df: pd.DataFrame) -> dict:
    """
    Takes a DataFrame with columns [open, high, low, close, volume]
    and returns a BUY / SELL / HOLD decision based on Kronos price forecast.
    """
    try:
        predictor = _get_predictor()
    except (FileNotFoundError, ImportError) as e:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": str(e), "provider": "Kronos (Local)",
        }

    input_df = bars_df[["open", "high", "low", "close", "volume"]].tail(200).copy()

    if len(input_df) < 60:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": f"Not enough bars ({len(input_df)} < 60 required).",
            "provider": "Kronos (Local)",
        }

    try:
        predictions = predictor.predict(input_df, pred_len=10, num_samples=50)

        last_close = float(input_df["close"].iloc[-1])
        median_forecast = float(predictions["close"].median())
        p10 = float(predictions["close"].quantile(0.10))   # bearish scenario
        p90 = float(predictions["close"].quantile(0.90))   # bullish scenario
        pct_change = (median_forecast - last_close) / last_close * 100

        if pct_change > 1.5:
            action = "BUY"
            confidence = min(int(50 + pct_change * 10), 95)
        elif pct_change < -1.5:
            action = "SELL"
            confidence = min(int(50 + abs(pct_change) * 10), 95)
        else:
            action = "HOLD"
            confidence = 70

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": (
                f"Kronos ({KRONOS_MODEL_SIZE}) forecasts close at "
                f"${median_forecast:.2f} (now: ${last_close:.2f}, {pct_change:+.2f}%). "
                f"80% range: ${p10:.2f}\u2013${p90:.2f}"
            ),
            "provider": f"Kronos (Local / {KRONOS_MODEL_SIZE})",
        }

    except Exception as e:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": f"Kronos inference error: {e}",
            "provider": "Kronos (Local)",
        }
