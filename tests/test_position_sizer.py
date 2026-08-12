import pytest

from position_sizer import (
    atr_budget_dollars,
    conviction_weight_pct,
    quantity_for,
    size_position,
)

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


def _size(**overrides):
    kwargs = dict(
        price=100.0,
        equity=EQUITY,
        cash=EQUITY,
        atr=2.0,
        conviction=0.6,
        remaining_budget=65_000.0,
        sizing=SIZING,
    )
    kwargs.update(overrides)
    return size_position(**kwargs)


# --- conviction_weight_pct ---

def test_weight_saturates_at_the_cap():
    assert conviction_weight_pct(1.5, SIZING) == 12.0


def test_weight_scales_linearly_below_saturation():
    assert conviction_weight_pct(0.6, SIZING) == pytest.approx(7.2)


def test_weight_never_falls_below_the_floor():
    assert conviction_weight_pct(0.05, SIZING) == 2.0


# --- atr_budget_dollars ---

def test_atr_budget_risks_the_configured_slice_of_equity():
    # stop = 2 * 2.0 = 4.0 on a $100 price = 4%; risking 0.5% of 100k = $500.
    # $500 / 0.04 = $12,500 notional.
    assert atr_budget_dollars(EQUITY, 100.0, 2.0, SIZING) == pytest.approx(12_500.0)


def test_atr_budget_is_none_without_atr():
    assert atr_budget_dollars(EQUITY, 100.0, None, SIZING) is None
    assert atr_budget_dollars(EQUITY, 100.0, 0.0, SIZING) is None


def test_higher_volatility_yields_a_smaller_budget():
    calm = atr_budget_dollars(EQUITY, 100.0, 1.0, SIZING)
    wild = atr_budget_dollars(EQUITY, 100.0, 10.0, SIZING)
    assert wild < calm


# --- quantity_for ---

def test_whole_shares_for_normal_prices():
    assert quantity_for(1_000.0, 99.0) == 10.0


def test_fractional_quantity_below_one_dollar():
    assert quantity_for(1_000.0, 0.5) == 2_000.0


def test_zero_quantity_when_price_exceeds_the_budget():
    assert quantity_for(1_000.0, 64_000.0) == 0.0


# --- size_position ---

def test_conviction_binds_when_it_is_the_tightest_constraint():
    result = _size(conviction=0.6)
    assert result["binding"] == "conviction"
    assert result["weight_pct"] == pytest.approx(7.2)
    assert result["qty"] == 72.0  # 7.2% of 100k = $7,200 / $100


def test_atr_risk_binds_for_a_volatile_name():
    # atr=3 -> stop 6% -> budget $8,333; conviction at cap would be $12,000.
    result = _size(conviction=1.5, atr=3.0)
    assert result["binding"] == "atr_risk"
    assert result["qty"] == 83.0


def test_exposure_budget_binds_when_nearly_full():
    result = _size(conviction=1.5, remaining_budget=3_000.0)
    assert result["binding"] == "exposure_budget"
    assert result["qty"] == 30.0


def test_cash_binds_and_margin_is_never_used():
    result = _size(conviction=1.5, cash=2_500.0)
    assert result["binding"] == "cash"
    assert result["qty"] == 25.0


def test_low_conviction_is_rejected_outright():
    result = _size(conviction=0.1)
    assert result["qty"] == 0
    assert "below minimum" in result["rejected_reason"]


def test_below_floor_is_rejected_rather_than_shrunk():
    """This is the fix for 1-share stub positions."""
    result = _size(conviction=1.5, remaining_budget=500.0)
    assert result["qty"] == 0
    assert "below floor" in result["rejected_reason"]


def test_expensive_crypto_is_rejected_by_the_floor():
    """Known, accepted consequence: BTC cannot be sized without fractional support."""
    result = _size(conviction=1.5, price=64_000.0)
    assert result["qty"] == 0
    assert result["rejected_reason"]


def test_missing_atr_falls_back_to_the_other_constraints():
    result = _size(conviction=0.6, atr=None)
    assert result["binding"] == "conviction"
    assert result["qty"] == 72.0


def test_invalid_price_is_rejected():
    assert _size(price=0.0)["qty"] == 0


def test_reported_dollars_match_the_filled_quantity():
    result = _size(conviction=0.6)
    assert result["dollars"] == pytest.approx(result["qty"] * 100.0)
