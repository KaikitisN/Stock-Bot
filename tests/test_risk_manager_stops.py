import pytest

from risk_manager import atr_stop_take_profit_prices, stop_target_for

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

RISK_CFG = {
    "max_position_pct": 3.0,
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0,
    "max_daily_loss_pct": 3.0,
}


def test_buy_stop_is_below_and_target_above_entry():
    stop, target = atr_stop_take_profit_prices(100.0, 2.0, SIZING, side="BUY")
    assert stop == 96.0   # 100 - 2*2
    assert target == 108.0  # 100 + 4*2


def test_sell_side_is_mirrored():
    stop, target = atr_stop_take_profit_prices(100.0, 2.0, SIZING, side="SELL")
    assert stop == 104.0
    assert target == 92.0


def test_reward_to_risk_follows_the_atr_multiples():
    stop, target = atr_stop_take_profit_prices(100.0, 3.0, SIZING, side="BUY")
    assert (target - 100.0) / (100.0 - stop) == pytest.approx(2.0)


def test_volatile_name_gets_a_wider_stop_than_a_calm_one():
    calm_stop, _ = atr_stop_take_profit_prices(100.0, 0.5, SIZING, side="BUY")
    wild_stop, _ = atr_stop_take_profit_prices(100.0, 8.0, SIZING, side="BUY")
    assert wild_stop < calm_stop


def test_returns_none_without_atr():
    assert atr_stop_take_profit_prices(100.0, None, SIZING) is None
    assert atr_stop_take_profit_prices(100.0, 0.0, SIZING) is None


def test_returns_none_when_atr_would_push_the_stop_to_zero():
    """A 2xATR stop wider than the price itself is not a usable stop."""
    assert atr_stop_take_profit_prices(10.0, 6.0, SIZING, side="BUY") is None


def test_stop_target_for_prefers_atr():
    stop, target = stop_target_for(100.0, 2.0, SIZING, RISK_CFG, side="BUY")
    assert (stop, target) == (96.0, 108.0)


def test_stop_target_for_falls_back_to_percentages():
    stop, target = stop_target_for(100.0, None, SIZING, RISK_CFG, side="BUY")
    assert stop == 97.0   # 3% below
    assert target == 106.0  # 6% above


def test_fallback_also_applies_when_atr_is_unusable():
    stop, target = stop_target_for(10.0, 6.0, SIZING, RISK_CFG, side="BUY")
    assert stop == 9.7
    assert target == 10.6
