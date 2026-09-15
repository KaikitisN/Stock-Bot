"""
Conservative rebalancing: free capital for stronger new entries by closing
weaker open holdings.

Pure planning only — the broker is never touched here. Callers execute the
returned close plans, refresh portfolio state, then run allocate() as usual.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

from position_sizer import size_position


def latest_convictions(path: str) -> dict[str, float]:
    """Latest logged conviction per symbol from the decisions CSV.

    Missing file / column / values → empty or 0.0. Rows are assumed
    chronological (append-only log); the last row for a symbol wins.
    """
    if not path or not os.path.isfile(path):
        return {}

    out: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                symbol = (row.get("symbol") or "").strip()
                if not symbol:
                    continue
                raw = row.get("conviction", "")
                try:
                    out[symbol] = float(raw) if raw not in ("", None) else 0.0
                except (TypeError, ValueError):
                    out[symbol] = 0.0
    except OSError:
        return {}
    return out


def holdings_from_positions(positions, convictions: dict[str, float]) -> list[dict]:
    """Normalize Alpaca positions (or dicts) into rebalance holdings."""
    holdings = []
    for p in positions or []:
        if isinstance(p, dict):
            symbol = str(p.get("symbol") or "")
            qty = float(p.get("qty") or 0)
            price = float(p.get("current_price") or p.get("price") or 0)
            raw_mv = p.get("market_value")
        else:
            symbol = str(getattr(p, "symbol", "") or "")
            qty = float(getattr(p, "qty", 0) or 0)
            price = float(getattr(p, "current_price", 0) or 0)
            raw_mv = getattr(p, "market_value", None)

        if not symbol or qty == 0:
            continue

        market_value = float(raw_mv) if raw_mv is not None else qty * price
        dollars = abs(market_value)
        if dollars <= 0:
            continue

        holdings.append({
            "symbol": symbol,
            "qty": abs(qty),
            "price": price if price > 0 else (dollars / abs(qty) if qty else 0),
            "dollars": dollars,
            "side": "long" if qty > 0 else "short",
            "conviction": float(convictions.get(symbol, 0.0) or 0.0),
        })
    return holdings


def _close_plan(victim: dict, for_entry: dict) -> dict:
    reason = (
        f"Rebalance: closing {victim['symbol']} (conviction "
        f"{victim['conviction']:.2f}) to free capital for {for_entry['symbol']} "
        f"(conviction {for_entry['conviction']:.2f})"
    )
    decision = {
        "symbol": victim["symbol"],
        "action": "SELL",
        "confidence": 0,
        "reason": reason,
        "provider": "rebalance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_submitted": False,
        "error": "",
        "mu_pct": 0.0,
        "sigma_pct": 0.0,
        "conviction": victim["conviction"],
        "p_up": 0.0,
    }
    return {
        "symbol": victim["symbol"],
        "action": "SELL",
        "price": victim["price"],
        "atr": None,
        "conviction": victim["conviction"],
        "closing": True,
        "rebalance": True,
        "dollars": victim["dollars"],
        "qty": victim["qty"],
        "for_symbol": for_entry["symbol"],
        "decision": decision,
    }


def propose_rebalance_closes(
    entries,
    holdings,
    *,
    equity: float,
    cash: float,
    budget: float,
    open_positions: int,
    max_open_positions: int,
    sizing: dict,
    exclude_symbols=None,
) -> list[dict]:
    """Return full-close plans that free enough room for stronger entries.

    Conservative: at most one displacement per entry, and only when
    entry.conviction >= held.conviction + rebalance_conviction_gap and the
    freed dollars would actually make that entry fundable.
    """
    gap = float(sizing.get("rebalance_conviction_gap", 0.3))
    exclude = set(exclude_symbols or ())
    exclude |= {e["symbol"] for e in entries}

    pool = [
        h for h in holdings
        if h["symbol"] not in exclude and h.get("dollars", 0) > 0
    ]
    pool.sort(key=lambda h: (h["conviction"], h["dollars"]))

    remaining_budget = max(budget, 0.0)
    remaining_cash = max(cash, 0.0)
    slots = max(max_open_positions - open_positions, 0)
    claimed: set[str] = set()
    closes: list[dict] = []

    for candidate in sorted(entries, key=lambda c: c["conviction"], reverse=True):
        funded = False
        if slots > 0 and remaining_budget > 0:
            result = size_position(
                price=candidate["price"],
                equity=equity,
                cash=remaining_cash,
                atr=candidate.get("atr"),
                conviction=candidate["conviction"],
                remaining_budget=remaining_budget,
                sizing=sizing,
            )
            if result["qty"] > 0:
                remaining_budget -= result["dollars"]
                remaining_cash -= result["dollars"]
                slots -= 1
                funded = True

        if funded:
            continue

        victim = None
        for h in pool:
            if h["symbol"] in claimed:
                continue
            if candidate["conviction"] >= h["conviction"] + gap:
                victim = h
                break

        if victim is None:
            continue

        trial_budget = remaining_budget + victim["dollars"]
        trial_cash = remaining_cash + victim["dollars"]
        trial_slots = slots + 1
        if trial_slots <= 0:
            continue

        result = size_position(
            price=candidate["price"],
            equity=equity,
            cash=trial_cash,
            atr=candidate.get("atr"),
            conviction=candidate["conviction"],
            remaining_budget=trial_budget,
            sizing=sizing,
        )
        if result["qty"] <= 0:
            continue

        claimed.add(victim["symbol"])
        closes.append(_close_plan(victim, candidate))
        remaining_budget = trial_budget - result["dollars"]
        remaining_cash = trial_cash - result["dollars"]
        slots = trial_slots - 1

    return closes
