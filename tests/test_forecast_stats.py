import math

from forecast_stats import summarize


def test_returns_none_for_single_path():
    assert summarize([0.03]) is None


def test_returns_none_for_empty_input():
    assert summarize([]) is None


def test_non_finite_values_are_dropped():
    stats = summarize([0.01, float("nan"), 0.03, float("inf")])
    assert stats["n_paths"] == 2


def test_mu_is_the_mean_of_terminal_returns():
    stats = summarize([0.02, 0.04])
    assert stats["mu"] == 0.03


def test_p_up_is_the_fraction_of_positive_paths():
    stats = summarize([0.01, -0.01, 0.02, 0.03])
    assert stats["p_up"] == 0.75


def test_tight_agreement_beats_scattered_for_the_same_mean():
    """The whole point of the change: same forecast, different conviction."""
    tight = summarize([0.02, 0.03, 0.04])
    scattered = summarize([-0.08, 0.03, 0.14])
    assert math.isclose(tight["mu"], scattered["mu"], abs_tol=1e-9)
    assert tight["conviction"] > scattered["conviction"]


def test_conviction_is_absolute_so_shorts_are_symmetric():
    bullish = summarize([0.02, 0.03, 0.04])
    bearish = summarize([-0.02, -0.03, -0.04])
    assert bearish["ir"] < 0
    assert math.isclose(bullish["conviction"], bearish["conviction"], rel_tol=1e-9)


def test_sigma_is_floored_so_conviction_stays_finite():
    stats = summarize([0.03, 0.03, 0.03])
    assert stats["sigma"] > 0
    assert math.isfinite(stats["conviction"])


def test_percentiles_bracket_the_mean():
    stats = summarize([-0.05, 0.0, 0.03, 0.06, 0.10])
    assert stats["p10"] < stats["mu"] < stats["p90"]
