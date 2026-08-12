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

import config  # needed for KRONOS_SIGNAL_THRESHOLD_PCT
from forecast_stats import summarize

KRONOS_REPO_PATH = os.getenv("KRONOS_REPO_PATH", "../Kronos")
KRONOS_MODEL_SIZE = os.getenv("KRONOS_MODEL_SIZE", "mini")

_predictor = None  # singleton — loaded once, reused every cycle


def _get_timestamp_series(df: pd.DataFrame):
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index), index=df.index)

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return pd.to_datetime(df[column])

    for column_name in ("timestamp", "timestamps", "datetime", "date", "time"):
        if column_name in df.columns:
            return pd.to_datetime(df[column_name])

    return None


def _make_future_timestamps(x_timestamp: pd.Series, pred_len: int):
    x_timestamp = pd.to_datetime(pd.Index(x_timestamp))
    if len(x_timestamp) < 2:
        freq = pd.Timedelta(hours=1)
    else:
        delta = pd.Series(x_timestamp).diff().dropna().median()
        freq = delta if pd.notna(delta) and delta > pd.Timedelta(0) else pd.Timedelta(hours=1)

    start = x_timestamp[-1] + freq
    return pd.Series(pd.date_range(start=start, periods=pred_len, freq=freq))


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
            "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
        }

    input_df = bars_df[["open", "high", "low", "close", "volume"]].tail(200).copy()
    x_timestamp = _get_timestamp_series(bars_df.tail(len(input_df)))

    if x_timestamp is None:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": "Kronos input is missing timestamps; expected a DatetimeIndex or a timestamp column.",
            "provider": "Kronos (Local)",
            "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
        }

    y_timestamp = _make_future_timestamps(x_timestamp, pred_len=10)

    if len(input_df) < 60:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": f"Not enough bars ({len(input_df)} < 60 required).",
            "provider": "Kronos (Local)",
            "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
        }

    try:
        n_paths = getattr(config, "KRONOS_SAMPLE_PATHS", 30)

        # One series per path with sample_count=1 gives K independent futures in
        # a single batched pass. sample_count > 1 would make Kronos average the
        # paths internally (model/kronos.py:465-467), which is exactly the
        # dispersion we need to measure.
        path_dfs = predictor.predict_batch(
            df_list=[input_df] * n_paths,
            x_timestamp_list=[x_timestamp] * n_paths,
            y_timestamp_list=[y_timestamp] * n_paths,
            pred_len=10,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        last_close = float(input_df["close"].iloc[-1])
        returns = [
            (float(path["close"].iloc[-1]) - last_close) / last_close
            for path in path_dfs
        ]
        stats = summarize(returns)

        if stats is None:
            return {
                "symbol": symbol, "action": "HOLD", "confidence": 0,
                "reason": (
                    f"Too few usable forecast paths ({len(returns)}); "
                    "dispersion is undefined."
                ),
                "provider": f"Kronos (Local / {KRONOS_MODEL_SIZE})",
                "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
            }

        mu_pct = stats["mu"] * 100
        sigma_pct = stats["sigma"] * 100
        signal_threshold = getattr(config, "KRONOS_SIGNAL_THRESHOLD_PCT", 2.5)
        min_confidence = getattr(config, "MIN_TRADE_CONFIDENCE", 70)

        if mu_pct > signal_threshold:
            action = "BUY"
            confidence = round(100 * stats["p_up"])
        elif mu_pct < -signal_threshold:
            action = "SELL"
            confidence = round(100 * (1 - stats["p_up"]))
        else:
            action = "HOLD"
            agreement = round(100 * max(stats["p_up"], 1 - stats["p_up"]))
            confidence = min(agreement, min_confidence - 1)

        expected_close = last_close * (1 + stats["mu"])
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": (
                f"Kronos ({KRONOS_MODEL_SIZE}) expects ${expected_close:.2f} "
                f"(now: ${last_close:.2f}, {mu_pct:+.2f}%) across "
                f"{stats['n_paths']} paths. "
                f"Dispersion {sigma_pct:.2f}%, conviction {stats['conviction']:.2f}, "
                f"{round(100 * stats['p_up'])}% of paths up. "
                f"80% of paths land between {stats['p10'] * 100:+.2f}% and "
                f"{stats['p90'] * 100:+.2f}%. "
                f"Signal threshold: {signal_threshold:.2f}%"
            ),
            "provider": f"Kronos (Local / {KRONOS_MODEL_SIZE})",
            "mu_pct": round(mu_pct, 4),
            "sigma_pct": round(sigma_pct, 4),
            "conviction": round(stats["conviction"], 4),
            "p_up": stats["p_up"],
        }

    except Exception as e:
        return {
            "symbol": symbol, "action": "HOLD", "confidence": 0,
            "reason": f"Kronos inference error: {e}",
            "provider": "Kronos (Local)",
            "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
        }
