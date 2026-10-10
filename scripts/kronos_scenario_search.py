"""
Walk-forward search for Kronos-mini settings on real crypto history.

Inference settings are fixed in ADVANCE. Decision settings (threshold,
confidence, conviction, horizon, long-only vs both, trend filter) are chosen
only on the training window. The last 30 days are touched once, after that
choice, so the reported percentages are out of sample.

Forecasts are cached under logs/kronos_search so a stopped run can resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config  # noqa: E402

config.TORCH_NUM_THREADS = max(1, os.cpu_count() or 1)

from alpaca.data.timeframe import TimeFrame  # noqa: E402
from data_fetcher import fetch_bars  # noqa: E402
from forecast_stats import summarize  # noqa: E402

OUT_DIR = os.path.join(ROOT, "logs", "kronos_search")
BARS_CACHE = os.path.join(OUT_DIR, "hourly_bars.pkl")

SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
]

# Chosen before any score was computed.
INFERENCE_CONFIGS = [
    {"name": "bot_default", "lookback": 200, "temperature": 1.0, "top_p": 0.9},
    {"name": "short_memory", "lookback": 64, "temperature": 1.0, "top_p": 0.9},
    {"name": "long_memory", "lookback": 512, "temperature": 1.0, "top_p": 0.9},
    {"name": "cool_sample", "lookback": 200, "temperature": 0.6, "top_p": 0.8},
    {"name": "hot_sample", "lookback": 200, "temperature": 1.3, "top_p": 0.95},
]

HORIZONS = [4, 8, 10, 12]
THRESHOLDS = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]
MIN_P = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_CONV = [0.0, 0.2, 0.35, 0.5, 0.8, 1.2]
ROUND_TRIP_COST = 0.002  # 0.20% round trip, applied to every filled signal
MIN_TRAIN_TRADES = 40
HOLDOUT_DAYS = 30


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT_DIR, "search.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_frames(days: int) -> dict[str, pd.DataFrame]:
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(BARS_CACHE):
        frames = pd.read_pickle(BARS_CACHE)
        log(f"loaded cached bars for {list(frames)}")
        return frames

    log(f"fetching {days} days of hourly bars for {len(SYMBOLS)} coins")
    raw = fetch_bars(SYMBOLS, lookback_days=days, timeframe=TimeFrame.Hour)
    frames = {}
    for symbol in SYMBOLS:
        try:
            frame = raw.loc[symbol].reset_index()
        except KeyError:
            log(f"missing {symbol}")
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
        frames[symbol] = frame.reset_index(drop=True)
        log(f"{symbol}: {len(frame)} bars {frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]}")
    pd.to_pickle(frames, BARS_CACHE)
    return frames


def jobs_for(frames, lookback: int, pred_len: int, step: int):
    jobs = []
    for symbol, frame in frames.items():
        last = len(frame) - pred_len - 1
        start = lookback - 1
        for origin in range(start, last + 1, step):
            jobs.append((symbol, origin))
    return jobs


def cache_path(name: str) -> str:
    return os.path.join(OUT_DIR, f"forecasts_{name}.npz")


def load_done(path: str):
    if not os.path.isfile(path):
        return set(), []
    data = np.load(path, allow_pickle=True)
    rows = []
    done = set()
    for i in range(len(data["symbol"])):
        row = {
            "symbol": str(data["symbol"][i]),
            "origin": int(data["origin"][i]),
            "timestamp": data["timestamp"][i],
            "path_ret": data["path_ret"][i],
            "realized": data["realized"][i],
            "past_ret": float(data["past_ret"][i]),
            "uptrend": bool(data["uptrend"][i]),
        }
        rows.append(row)
        done.add((row["symbol"], row["origin"]))
    return done, rows


def save_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    payload = dict(
        symbol=np.array([r["symbol"] for r in rows]),
        origin=np.array([r["origin"] for r in rows], dtype=np.int32),
        timestamp=np.array([r["timestamp"] for r in rows]),
        path_ret=np.stack([r["path_ret"] for r in rows]).astype(np.float32),
        realized=np.stack([r["realized"] for r in rows]).astype(np.float32),
        past_ret=np.array([r["past_ret"] for r in rows], dtype=np.float32),
        uptrend=np.array([r["uptrend"] for r in rows], dtype=bool),
    )
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def run_inference(predictor, frames, cfg, pred_len, paths, step, limit, batch_series):
    name = cfg["name"]
    path = cache_path(name)
    done, rows = load_done(path)
    pending = [
        job for job in jobs_for(frames, cfg["lookback"], pred_len, step) if job not in done
    ]
    if limit:
        pending = pending[: max(0, limit - len(rows))]
    log(f"{name}: {len(rows)} cached, {len(pending)} remaining")
    if not pending:
        return rows

    started = time.time()
    cursor = 0
    per_batch = max(1, batch_series // paths)
    while cursor < len(pending):
        chunk = pending[cursor: cursor + per_batch]
        df_list = []
        x_list = []
        y_list = []
        meta = []
        for symbol, origin in chunk:
            frame = frames[symbol]
            window = frame.iloc[origin - cfg["lookback"] + 1: origin + 1]
            future = frame.iloc[origin + 1: origin + 1 + pred_len]
            hist = window[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
            x_ts = pd.to_datetime(window["timestamp"]).reset_index(drop=True)
            y_ts = pd.to_datetime(future["timestamp"]).reset_index(drop=True)
            entry = float(hist["close"].iloc[-1])
            realized = future["close"].astype(float).to_numpy() / entry - 1.0
            past_close = float(frame["close"].iloc[max(0, origin - 24)])
            past_ret = entry / past_close - 1.0 if past_close else 0.0
            sma = float(frame["close"].iloc[origin - 29: origin + 1].astype(float).mean())
            meta.append({
                "symbol": symbol,
                "origin": origin,
                "timestamp": np.datetime64(pd.Timestamp(window["timestamp"].iloc[-1]).to_datetime64()),
                "realized": realized.astype(np.float32),
                "past_ret": past_ret,
                "uptrend": entry > sma,
                "entry": entry,
            })
            for _ in range(paths):
                df_list.append(hist)
                x_list.append(x_ts)
                y_list.append(y_ts)

        forecasts = predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_list,
            y_timestamp_list=y_list,
            pred_len=pred_len,
            T=cfg["temperature"],
            top_p=cfg["top_p"],
            sample_count=1,
            verbose=False,
        )
        for i, item in enumerate(meta):
            path_closes = []
            for p in range(paths):
                closes = forecasts[i * paths + p]["close"].astype(float).to_numpy()
                path_closes.append(closes / item["entry"] - 1.0)
            item["path_ret"] = np.stack(path_closes).astype(np.float32)
            del item["entry"]
            rows.append(item)

        cursor += len(chunk)
        if cursor % (per_batch * 5) < per_batch or cursor >= len(pending):
            save_rows(path, rows)
            elapsed = time.time() - started
            rate = cursor / elapsed
            eta = (len(pending) - cursor) / rate if rate else 0
            log(f"{name}: {len(rows)} forecasts, eta {eta / 60:.1f} min")

    save_rows(path, rows)
    return rows


def _stats_at(path_ret: np.ndarray, horizon: int):
    mus, sigmas, pups, convs = [], [], [], []
    for row in path_ret:
        stats = summarize(row[:, horizon - 1])
        if stats is None:
            mus.append(np.nan)
            sigmas.append(np.nan)
            pups.append(np.nan)
            convs.append(np.nan)
        else:
            mus.append(stats["mu"])
            sigmas.append(stats["sigma"])
            pups.append(stats["p_up"])
            convs.append(stats["conviction"])
    return (
        np.array(mus),
        np.array(sigmas),
        np.array(pups),
        np.array(convs),
    )


def _pack(name, inference, spec, pnls, train: bool):
    if len(pnls) == 0:
        return None
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    ordered = np.asarray(pnls, dtype=float)
    equity = np.cumprod(1.0 + ordered)
    peak = np.maximum.accumulate(equity)
    drawdown = float((equity / peak - 1.0).min()) if len(equity) else 0.0
    return {
        "inference": inference,
        "split": "train" if train else "test",
        "horizon_hours": spec["horizon"],
        "threshold_pct": spec["threshold"],
        "min_confidence": int(round(spec["min_p"] * 100)),
        "min_conviction": spec["min_conv"],
        "side": spec["side"],
        "trend_filter": spec["trend"],
        "trades": int(len(ordered)),
        "win_rate_pct": round(float((ordered > 0).mean() * 100), 2),
        "avg_trade_pct": round(float(ordered.mean() * 100), 3),
        "median_trade_pct": round(float(np.median(ordered) * 100), 3),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "compounded_pct": round(float((equity[-1] - 1.0) * 100), 2),
        "max_drawdown_pct": round(drawdown * 100, 2),
        "label": name,
    }


def score_rows(rows, inference_name: str):
    if not rows:
        return None
    timestamps = pd.to_datetime(np.array([r["timestamp"] for r in rows]))
    cutoff = timestamps.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_mask = np.asarray(timestamps < cutoff)
    test_mask = ~train_mask
    path_ret = np.stack([r["path_ret"] for r in rows])
    realized = np.stack([r["realized"] for r in rows])
    uptrend = np.array([r["uptrend"] for r in rows])
    past_ret = np.array([r["past_ret"] for r in rows])

    best = None
    best_score = -1e18
    current = None
    baselines = []

    for horizon in HORIZONS:
        if horizon > path_ret.shape[-1]:
            continue
        mu, _sigma, p_up, conv = _stats_at(path_ret, horizon)
        future = realized[:, horizon - 1]
        valid = np.isfinite(mu) & np.isfinite(future)

        for trend in (False, True):
            for side in ("long", "both"):
                for threshold in THRESHOLDS:
                    for min_p in MIN_P:
                        for min_conv in MIN_CONV:
                            long_ok = (mu * 100 > threshold) & (p_up >= min_p) & (conv >= min_conv)
                            short_ok = (mu * 100 < -threshold) & ((1 - p_up) >= min_p) & (conv >= min_conv)
                            if trend:
                                long_ok = long_ok & uptrend
                                short_ok = short_ok & ~uptrend
                            if side == "long":
                                short_ok = np.zeros_like(short_ok, dtype=bool)
                            take = (long_ok | short_ok) & valid
                            direction = np.where(long_ok, 1.0, np.where(short_ok, -1.0, 0.0))
                            pnl = direction * future - ROUND_TRIP_COST
                            train_pnl = pnl[take & train_mask]
                            if len(train_pnl) < MIN_TRAIN_TRADES:
                                continue
                            mean = float(train_pnl.mean())
                            std = float(train_pnl.std(ddof=1))
                            t_stat = mean / std * np.sqrt(len(train_pnl)) if std > 1e-9 else -1e18
                            spec = {
                                "horizon": horizon,
                                "threshold": threshold,
                                "min_p": min_p,
                                "min_conv": min_conv,
                                "side": side,
                                "trend": trend,
                            }
                            is_current = (
                                inference_name == "bot_default"
                                and horizon == 10
                                and threshold == 1.5
                                and min_p == 0.70
                                and min_conv == 0.2
                                and side == "long"
                                and trend is True
                            )
                            if is_current:
                                current = {
                                    "spec": spec,
                                    "train": _pack("current_bot", inference_name, spec, train_pnl, True),
                                    "test": _pack(
                                        "current_bot",
                                        inference_name,
                                        spec,
                                        pnl[take & test_mask],
                                        False,
                                    ),
                                }
                            if mean <= 0 or t_stat <= best_score:
                                continue
                            best_score = t_stat
                            best = {
                                "spec": spec,
                                "train_t": round(t_stat, 3),
                                "train": _pack("best_train", inference_name, spec, train_pnl, True),
                                "test": _pack(
                                    "best_train",
                                    inference_name,
                                    spec,
                                    pnl[take & test_mask],
                                    False,
                                ),
                            }

        # Baselines use the same horizon and the same dates, with no Kronos filter.
        always = future[valid & test_mask] - ROUND_TRIP_COST
        mom_dir = np.sign(past_ret)
        mom_dir[mom_dir == 0] = 1.0
        momentum = mom_dir[valid & test_mask] * future[valid & test_mask] - ROUND_TRIP_COST
        baselines.append({
            "horizon_hours": horizon,
            "always_long_avg_pct": round(float(always.mean() * 100), 3) if len(always) else None,
            "always_long_win_pct": round(float((always > 0).mean() * 100), 2) if len(always) else None,
            "momentum_avg_pct": round(float(momentum.mean() * 100), 3) if len(momentum) else None,
            "momentum_win_pct": round(float((momentum > 0).mean() * 100), 2) if len(momentum) else None,
            "test_points": int((valid & test_mask).sum()),
        })

    return {
        "inference": inference_name,
        "forecasts": len(rows),
        "train_forecasts": int(train_mask.sum()),
        "test_forecasts": int(test_mask.sum()),
        "cutoff": str(cutoff),
        "best": best,
        "current_bot_rules": current,
        "baselines_test": baselines,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-series", type=int, default=16)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = load_frames(args.days)
    import kronos_decision

    predictor = kronos_decision._get_predictor()
    configs = INFERENCE_CONFIGS
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]

    board_path = os.path.join(OUT_DIR, "leaderboard.json")
    board = []
    if os.path.isfile(board_path):
        with open(board_path, encoding="utf-8") as fh:
            board = json.load(fh)

    for cfg in configs:
        rows = run_inference(
            predictor,
            frames,
            cfg,
            args.pred_len,
            args.paths,
            args.step,
            args.limit,
            args.batch_series,
        )
        if args.limit:
            log(f"smoke stop after limit={args.limit}")
            return
        result = score_rows(rows, cfg["name"])
        result["settings"] = {
            "lookback": cfg["lookback"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "paths": args.paths,
            "pred_len": args.pred_len,
            "step_hours": args.step,
            "round_trip_cost": ROUND_TRIP_COST,
        }
        board = [row for row in board if row.get("inference") != cfg["name"]]
        board.append(result)
        with open(board_path, "w", encoding="utf-8") as fh:
            json.dump(board, fh, indent=2)
        best = result.get("best") or {}
        test = (best.get("test") or {})
        train = (best.get("train") or {})
        log(
            f"{cfg['name']} train win {train.get('win_rate_pct')}% "
            f"avg {train.get('avg_trade_pct')}% | test trades {test.get('trades')} "
            f"win {test.get('win_rate_pct')}% avg {test.get('avg_trade_pct')}%"
        )

    log(f"wrote {board_path}")


if __name__ == "__main__":
    main()
