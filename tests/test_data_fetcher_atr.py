import pandas as pd

from data_fetcher import compute_indicators


def _flat_range_bars(n=30, close=100.0, half_range=1.0):
    """Bars with a constant high-low range of 2*half_range and no gaps."""
    return pd.DataFrame({
        "open": [close] * n,
        "high": [close + half_range] * n,
        "low": [close - half_range] * n,
        "close": [close] * n,
        "volume": [1000] * n,
    })


def test_atr_14_column_is_added():
    out = compute_indicators(_flat_range_bars())
    assert "atr_14" in out.columns


def test_atr_equals_true_range_when_range_is_constant():
    out = compute_indicators(_flat_range_bars(half_range=1.0))
    assert out["atr_14"].iloc[-1] == 2.0


def test_atr_accounts_for_gaps_between_bars():
    """A gap up makes |high - prev_close| the true range, exceeding high-low."""
    df = _flat_range_bars(n=30)
    df.loc[29, ["open", "high", "low", "close"]] = [120.0, 121.0, 119.0, 120.0]
    out = compute_indicators(df)
    # Final bar's true range is |121 - 100| = 21, well above its 2-point high-low.
    assert out["atr_14"].iloc[-1] > 2.0


def test_atr_is_nan_before_enough_bars():
    out = compute_indicators(_flat_range_bars(n=10))
    assert pd.isna(out["atr_14"].iloc[-1])


def test_existing_indicators_still_present():
    out = compute_indicators(_flat_range_bars())
    for col in ("sma_10", "sma_30", "rsi_14"):
        assert col in out.columns
