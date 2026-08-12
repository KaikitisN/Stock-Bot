import pandas as pd
import pytest

import kronos_decision


class StubPredictor:
    """Returns one forecast DataFrame per requested path, with preset closes."""

    def __init__(self, terminal_closes):
        self.terminal_closes = terminal_closes
        self.last_call = None

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len,
                      T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        self.last_call = {
            "n_series": len(df_list),
            "sample_count": sample_count,
            "pred_len": pred_len,
        }
        frames = []
        for close in self.terminal_closes:
            frames.append(pd.DataFrame({
                "open": [close] * pred_len,
                "high": [close] * pred_len,
                "low": [close] * pred_len,
                "close": [close] * pred_len,
                "volume": [1.0] * pred_len,
                "amount": [1.0] * pred_len,
            }))
        return frames


def _bars(n=120, close=100.0):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": [close] * n,
        "high": [close + 1] * n,
        "low": [close - 1] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
    })


@pytest.fixture
def stub(monkeypatch):
    def _install(terminal_closes):
        predictor = StubPredictor(terminal_closes)
        monkeypatch.setattr(kronos_decision, "_get_predictor", lambda: predictor)
        return predictor
    return _install


def test_sample_count_must_be_one_to_preserve_dispersion(stub):
    """sample_count > 1 makes Kronos average paths internally, destroying sigma."""
    predictor = stub([103.0] * 30)
    kronos_decision.get_kronos_decision("TEST", _bars())
    assert predictor.last_call["sample_count"] == 1


def test_one_series_is_submitted_per_requested_path(stub):
    import config
    predictor = stub([103.0] * config.KRONOS_SAMPLE_PATHS)
    kronos_decision.get_kronos_decision("TEST", _bars())
    assert predictor.last_call["n_series"] == config.KRONOS_SAMPLE_PATHS


def test_agreed_bullish_forecast_is_a_confident_buy(stub):
    stub([104.0, 105.0, 106.0] * 10)  # every path well above the 2.5% threshold
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["action"] == "BUY"
    assert decision["confidence"] == 100
    assert decision["conviction"] > 1.0


def test_scattered_forecast_yields_low_conviction(stub):
    stub([92.0, 103.0, 114.0] * 10)  # same mean, wide spread
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["conviction"] < 0.5


def test_confidence_is_path_agreement_not_forecast_magnitude(stub):
    # 20 of 30 paths up: p_up = 2/3, so confidence must be 67, not a
    # function of the average move.
    stub([106.0] * 20 + [94.0] * 10)
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["confidence"] == 67


def test_agreed_bearish_forecast_is_a_confident_sell(stub):
    stub([96.0, 95.0, 94.0] * 10)
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["action"] == "SELL"
    assert decision["confidence"] == 100
    assert decision["mu_pct"] < 0


def test_small_forecast_move_is_hold_below_the_trade_gate(stub):
    import config
    stub([100.2] * 30)  # +0.2%, under the 2.5% signal threshold
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["action"] == "HOLD"
    assert decision["confidence"] < config.MIN_TRADE_CONFIDENCE


def test_reason_reports_percentile_range_across_paths(stub):
    stub([92.0, 103.0, 114.0] * 10)
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert "80% of paths" in decision["reason"]


def test_single_path_is_holds_because_dispersion_is_undefined(stub):
    stub([104.0])
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["action"] == "HOLD"
    assert "path" in decision["reason"].lower()


def test_inference_failure_holds_and_reports(stub, monkeypatch):
    class Boom:
        def predict_batch(self, *a, **k):
            raise RuntimeError("cuda exploded")

    monkeypatch.setattr(kronos_decision, "_get_predictor", lambda: Boom())
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    assert decision["action"] == "HOLD"
    assert decision["confidence"] == 0
    assert "cuda exploded" in decision["reason"]


def test_too_few_bars_holds(stub):
    stub([104.0] * 30)
    decision = kronos_decision.get_kronos_decision("TEST", _bars(n=30))
    assert decision["action"] == "HOLD"
    assert "60 required" in decision["reason"]


def test_stats_fields_are_flat_scalars_for_csv_logging(stub):
    stub([104.0] * 30)
    decision = kronos_decision.get_kronos_decision("TEST", _bars())
    for key in ("mu_pct", "sigma_pct", "conviction", "p_up"):
        assert isinstance(decision[key], float), f"{key} must be a flat float"
