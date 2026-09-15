"""Conservative rebalance: displace weak holdings for stronger new entries."""
import csv

import pytest

from rebalance import (
    holdings_from_positions,
    latest_convictions,
    propose_rebalance_closes,
)

SIZING = {
    "target_exposure_pct": 72.0,
    "min_position_pct": 1.5,
    "max_position_pct": 10.0,
    "risk_per_trade_pct": 0.5,
    "atr_stop_multiple": 2.0,
    "atr_target_multiple": 4.0,
    "ir_saturation": 1.0,
    "min_information_ratio": 0.2,
    "rebalance_conviction_gap": 0.3,
}

EQUITY = 100_000.0


def _entry(symbol, conviction, price=100.0, atr=None):
    return {
        "symbol": symbol,
        "action": "BUY",
        "price": price,
        "atr": atr,
        "conviction": conviction,
        "closing": False,
        "decision": {"symbol": symbol, "action": "BUY"},
    }


def _holding(symbol, dollars, conviction, price=100.0):
    return {
        "symbol": symbol,
        "qty": dollars / price,
        "price": price,
        "dollars": dollars,
        "side": "long",
        "conviction": conviction,
    }


def _propose(entries, holdings, **overrides):
    kwargs = dict(
        equity=EQUITY,
        cash=5_000.0,
        budget=0.0,
        open_positions=len(holdings),
        max_open_positions=12,
        sizing=SIZING,
    )
    kwargs.update(overrides)
    return propose_rebalance_closes(entries, holdings, **kwargs)


# --- latest_convictions ---

def test_latest_convictions_takes_last_row_per_symbol(tmp_path):
    path = tmp_path / "decisions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "conviction"])
        w.writeheader()
        w.writerow({"symbol": "AAPL", "conviction": "0.4"})
        w.writerow({"symbol": "AAPL", "conviction": "0.9"})
        w.writerow({"symbol": "NVDA", "conviction": "0.2"})
    assert latest_convictions(str(path)) == {"AAPL": 0.9, "NVDA": 0.2}


def test_latest_convictions_missing_file_is_empty():
    assert latest_convictions("no/such/file.csv") == {}


# --- holdings_from_positions ---

def test_holdings_use_absolute_market_value_and_logged_conviction():
    holdings = holdings_from_positions(
        [{"symbol": "TSLA", "qty": -2, "current_price": 250, "market_value": -500}],
        {"TSLA": 0.55},
    )
    assert holdings == [{
        "symbol": "TSLA",
        "qty": 2.0,
        "price": 250.0,
        "dollars": 500.0,
        "side": "short",
        "conviction": 0.55,
    }]


# --- propose_rebalance_closes ---

def test_no_closes_when_budget_already_funds_the_entry():
    closes = _propose(
        [_entry("LINK/USD", 1.6, price=11.0)],
        [_holding("WEAK", 10_000, 0.3)],
        budget=20_000.0,
        cash=20_000.0,
        open_positions=1,
    )
    assert closes == []


def test_closes_weakest_when_budget_is_full_and_gap_is_met():
    closes = _propose(
        [_entry("LINK/USD", 1.6, price=11.0)],
        [
            _holding("MID", 30_000, 0.8),
            _holding("WEAK", 12_000, 0.2),
        ],
        budget=0.0,
        cash=0.0,
        open_positions=2,
    )
    assert len(closes) == 1
    assert closes[0]["symbol"] == "WEAK"
    assert closes[0]["closing"] is True
    assert closes[0]["rebalance"] is True
    assert closes[0]["for_symbol"] == "LINK/USD"
    assert "Rebalance" in closes[0]["decision"]["reason"]


def test_does_not_close_when_conviction_gap_is_too_small():
    closes = _propose(
        [_entry("LINK/USD", 0.45)],
        [_holding("WEAK", 12_000, 0.3)],
        budget=0.0,
        cash=0.0,
        open_positions=1,
    )
    assert closes == []


def test_does_not_close_a_stronger_or_equal_holding():
    closes = _propose(
        [_entry("LINK/USD", 0.9)],
        [_holding("STRONG", 12_000, 1.2)],
        budget=0.0,
        cash=0.0,
        open_positions=1,
    )
    assert closes == []


def test_skips_symbols_already_excluded_or_entering():
    closes = _propose(
        [_entry("LINK/USD", 1.6), _entry("WEAK", 0.5)],
        [_holding("WEAK", 12_000, 0.2), _holding("OTHER", 12_000, 0.1)],
        budget=0.0,
        cash=0.0,
        open_positions=2,
        exclude_symbols={"OTHER"},
    )
    # WEAK is an entry this cycle; OTHER excluded → nothing to close.
    assert closes == []


def test_at_most_one_close_per_needy_entry():
    closes = _propose(
        [_entry("LINK/USD", 1.6)],
        [
            _holding("A", 5_000, 0.1),
            _holding("B", 5_000, 0.15),
            _holding("C", 5_000, 0.2),
        ],
        budget=0.0,
        cash=0.0,
        open_positions=3,
    )
    assert len(closes) == 1


def test_two_strong_entries_can_each_displace_one_weak():
    # Each weak frees only $8k; max weight is $10k, so leftover after the
    # first displacement cannot fund the second entry without another close.
    closes = _propose(
        [_entry("AAA", 1.5), _entry("BBB", 1.4)],
        [_holding("W1", 8_000, 0.2), _holding("W2", 8_000, 0.25)],
        budget=0.0,
        cash=0.0,
        open_positions=2,
    )
    assert {c["symbol"] for c in closes} == {"W1", "W2"}


def test_does_not_churn_if_freed_capital_still_cannot_fund():
    # Victim is tiny; even after close, floor cannot be met at this price.
    closes = _propose(
        [_entry("BTC/USD", 1.6, price=60_000.0)],
        [_holding("DUST", 50.0, 0.0, price=1.0)],
        budget=0.0,
        cash=0.0,
        open_positions=1,
    )
    assert closes == []
