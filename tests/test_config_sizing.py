import config

REQUIRED_KEYS = {
    "target_exposure_pct",
    "min_position_pct",
    "max_position_pct",
    "risk_per_trade_pct",
    "atr_stop_multiple",
    "atr_target_multiple",
    "ir_saturation",
    "min_information_ratio",
}


def test_sizing_block_has_all_required_keys():
    assert set(config.SIZING) == REQUIRED_KEYS


def test_sizing_values_are_floats():
    for key, value in config.SIZING.items():
        assert isinstance(value, float), f"{key} is {type(value)}, expected float"


def test_sizing_defaults_match_spec():
    assert config.SIZING["target_exposure_pct"] == 65.0
    assert config.SIZING["min_position_pct"] == 2.0
    assert config.SIZING["max_position_pct"] == 12.0
    assert config.SIZING["risk_per_trade_pct"] == 0.5
    assert config.SIZING["atr_stop_multiple"] == 2.0
    assert config.SIZING["atr_target_multiple"] == 4.0
    assert config.SIZING["ir_saturation"] == 1.0
    assert config.SIZING["min_information_ratio"] == 0.2


def test_position_band_is_coherent():
    assert config.SIZING["min_position_pct"] < config.SIZING["max_position_pct"]


def test_max_open_positions_can_fill_the_exposure_target():
    """At the 2% floor, 12 slots must be able to reach the 65% target."""
    max_reachable = config.MAX_OPEN_POSITIONS * config.SIZING["max_position_pct"]
    assert max_reachable >= config.SIZING["target_exposure_pct"]


def test_kronos_sample_paths_is_enough_for_dispersion():
    assert config.KRONOS_SAMPLE_PATHS >= 10
