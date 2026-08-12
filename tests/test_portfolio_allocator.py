import pytest

from portfolio_allocator import allocate, exposure_budget

SIZING = {
    "target_exposure_pct": 65.0,
    "min_position_pct": 2.0,
    "max_position_pct": 12.0,
    "risk_per_trade_pct": 0.5,
    "atr_stop_multiple": 2.0,
    "atr_target_multiple": 4.0,
    "ir_saturation": 1.0,
    "min_information_ratio": 0.2,
}

EQUITY = 100_000.0


def _candidate(symbol, conviction, price=100.0, atr=2.0):
    return {"symbol": symbol, "price": price, "atr": atr, "conviction": conviction}


def _allocate(candidates, **overrides):
    kwargs = dict(
        equity=EQUITY,
        cash=EQUITY,
        budget=65_000.0,
        open_positions=0,
        max_open_positions=12,
        sizing=SIZING,
    )
    kwargs.update(overrides)
    return allocate(candidates, **kwargs)


# --- exposure_budget ---

def test_budget_is_target_minus_current_exposure():
    # 65% of 100k = 65,000; current exposure 6,000 + 4,000 = 10,000.
    budget = exposure_budget(
        equity=EQUITY,
        long_market_value=6_000.0,
        short_market_value=-4_000.0,
        cash=EQUITY,
        sizing=SIZING,
    )
    assert budget == pytest.approx(55_000.0)


def test_shorts_count_toward_exposure_as_absolute_value():
    with_shorts = exposure_budget(
        equity=EQUITY, long_market_value=0.0, short_market_value=-20_000.0,
        cash=EQUITY, sizing=SIZING,
    )
    assert with_shorts == pytest.approx(45_000.0)


def test_budget_is_capped_by_cash_so_margin_is_never_used():
    budget = exposure_budget(
        equity=EQUITY, long_market_value=0.0, short_market_value=0.0,
        cash=1_000.0, sizing=SIZING,
    )
    assert budget == 1_000.0


def test_budget_is_zero_when_already_over_target():
    budget = exposure_budget(
        equity=EQUITY, long_market_value=80_000.0, short_market_value=0.0,
        cash=EQUITY, sizing=SIZING,
    )
    assert budget == 0.0


# --- allocate ---

def test_candidates_are_funded_in_conviction_order():
    plans = _allocate([
        _candidate("LOW", 0.3),
        _candidate("HIGH", 0.9),
        _candidate("MID", 0.6),
    ])
    assert [p["symbol"] for p in plans] == ["HIGH", "MID", "LOW"]
    assert all(p["funded"] for p in plans)


def test_highest_conviction_gets_the_largest_position():
    plans = _allocate([_candidate("LOW", 0.3), _candidate("HIGH", 0.9)])
    by_symbol = {p["symbol"]: p for p in plans}
    assert by_symbol["HIGH"]["dollars"] > by_symbol["LOW"]["dollars"]


def test_budget_exhaustion_leaves_later_candidates_unfunded():
    # Budget funds roughly one 12% position, not three.
    plans = _allocate(
        [_candidate("A", 1.5), _candidate("B", 1.4), _candidate("C", 1.3)],
        budget=12_000.0,
    )
    funded = [p for p in plans if p["funded"]]
    assert [p["symbol"] for p in funded] == ["A"]
    assert "Exposure budget full" in plans[-1]["rejected_reason"]


def test_position_cap_blocks_funding_when_slots_are_gone():
    plans = _allocate(
        [_candidate("A", 0.9), _candidate("B", 0.8)],
        open_positions=12,
        max_open_positions=12,
    )
    assert not any(p["funded"] for p in plans)
    assert "Max open positions" in plans[0]["rejected_reason"]


def test_partial_slots_fund_only_what_fits():
    plans = _allocate(
        [_candidate("A", 0.9), _candidate("B", 0.8), _candidate("C", 0.7)],
        open_positions=11,
        max_open_positions=12,
    )
    assert [p["symbol"] for p in plans if p["funded"]] == ["A"]


def test_funding_decrements_the_budget_for_later_candidates():
    plans = _allocate(
        [_candidate("A", 1.5), _candidate("B", 1.5)],
        budget=20_000.0,
    )
    by_symbol = {p["symbol"]: p for p in plans}
    assert by_symbol["A"]["dollars"] == pytest.approx(12_000.0)
    # Only ~8,000 of budget remains, so B is capped below A.
    assert by_symbol["B"]["dollars"] < by_symbol["A"]["dollars"]


def test_rejected_candidates_are_returned_not_dropped():
    plans = _allocate([_candidate("WEAK", 0.05), _candidate("GOOD", 0.9)])
    symbols = {p["symbol"] for p in plans}
    assert symbols == {"WEAK", "GOOD"}
    weak = next(p for p in plans if p["symbol"] == "WEAK")
    assert not weak["funded"]
    assert weak["rejected_reason"]


def test_empty_candidate_list_returns_empty_plan():
    assert _allocate([]) == []
