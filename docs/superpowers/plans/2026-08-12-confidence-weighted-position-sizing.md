# Confidence-Weighted Position Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Size every position from genuine Kronos forecast conviction bounded by an ATR risk budget, and fund candidates by rank against a portfolio-wide exposure budget instead of first-come-first-served.

**Architecture:** Three new pure-function modules (`forecast_stats`, `position_sizer`, `portfolio_allocator`) hold all the math and are unit-tested with plain dictionaries — no network, no model. `kronos_decision` recovers the real Monte-Carlo path distribution via `predict_batch`, and `orchestrator` splits into an evaluate phase (score every symbol, submit nothing) followed by an allocate-and-execute phase.

**Tech Stack:** Python 3.12, pandas, numpy, alpaca-py, pytest 9.1.1, Kronos (vendored at `../Kronos`, not modified).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-confidence-weighted-position-sizing-design.md`. Read it before starting.
- Working directory for every command is `c:\Users\user\Desktop\Projects\Stockbot\Stock-Bot`. Shell is PowerShell — `&&` is not a valid separator, use `;`, and heredocs do not work, so use multiple `-m` flags for commit messages.
- Python interpreter is always `..\.venv\Scripts\python.exe`. There is no activated venv.
- Never modify anything under `..\Kronos\` — it is a vendored upstream clone.
- Target total exposure 65% of equity; per-position floor 2%, cap 12%; **never use margin** — total exposure is additionally capped by available cash.
- Sizing is computed from **equity**, never from available cash (cash only acts as a ceiling).
- Direction comes from the sign of `mu`; all magnitude comparisons, ranking and sizing use `conviction = abs(ir)` so longs and shorts are symmetric.
- Candidates whose notional falls below the 2% floor are **rejected, never shrunk**. This is what structurally prevents 1-share stub positions.
- Every failure degrades toward not trading, never toward trading blind.
- Do not implement: correlation/sector caps, a backtest harness, fractional crypto sizing, or cleanup of the nine existing −1 share shorts. All explicitly out of scope.
- Crypto is knowingly left untradeable: BTC at ~$64k floors to 0 units under any 2–12% budget. Do not "fix" this.

---

### Task 1: Test infrastructure and sizing config

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_config_sizing.py`
- Modify: `requirements.txt` (append pytest)
- Modify: `config.py:36-40` (trade gates), `config.py:62-68` (risk block)

**Interfaces:**
- Consumes: nothing.
- Produces: `config.SIZING` — a `dict[str, float]` with exactly these keys: `target_exposure_pct`, `min_position_pct`, `max_position_pct`, `risk_per_trade_pct`, `atr_stop_multiple`, `atr_target_multiple`, `ir_saturation`, `min_information_ratio`. Also `config.KRONOS_SAMPLE_PATHS: int` and `config.MAX_OPEN_POSITIONS: int` (default now 12). Every later task reads these.

- [ ] **Step 1: Create `pytest.ini`**

`pythonpath = .` is what lets tests import the top-level modules (`config`, `position_sizer`) without a src layout.

```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 2: Create empty `tests/__init__.py`**

```bash
New-Item -ItemType File tests\__init__.py -Force
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_config_sizing.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_config_sizing.py -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'SIZING'`

- [ ] **Step 5: Add the config block**

In `config.py`, change `MAX_OPEN_POSITIONS` (currently line 38) from `"5"` to `"12"`:

```python
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "12"))
```

Then insert this immediately after the `DEFAULT_RISK` dict (currently ends line 68):

```python
def _env_float(key: str, default: float) -> float:
    """Read a float from the environment, falling back on missing or malformed values."""
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)


# --- Position sizing (all % of account EQUITY, never of cash) ---
# Sizing scales with forecast conviction (mu/sigma from Kronos sample paths),
# bounded independently by an ATR risk budget. See
# docs/superpowers/specs/2026-08-12-confidence-weighted-position-sizing-design.md
SIZING = {
    # Total deployed capital target across the whole book.
    "target_exposure_pct": _env_float("TARGET_EXPOSURE_PCT", 65.0),
    # Candidates sizing below this are rejected outright, never shrunk.
    "min_position_pct": _env_float("MIN_POSITION_PCT", 2.0),
    "max_position_pct": _env_float("MAX_POSITION_PCT", 12.0),
    # Equity risked per trade at the ATR stop.
    "risk_per_trade_pct": _env_float("RISK_PER_TRADE_PCT", 0.5),
    "atr_stop_multiple": _env_float("ATR_STOP_MULTIPLE", 2.0),
    "atr_target_multiple": _env_float("ATR_TARGET_MULTIPLE", 4.0),
    # conviction value at which max_position_pct is reached.
    "ir_saturation": _env_float("IR_SATURATION", 1.0),
    "min_information_ratio": _env_float("MIN_INFORMATION_RATIO", 0.2),
}

# Independent Kronos sample paths drawn per symbol to measure forecast dispersion.
KRONOS_SAMPLE_PATHS = int(os.getenv("KRONOS_SAMPLE_PATHS", "30"))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_config_sizing.py -v`
Expected: PASS, 6 passed

- [ ] **Step 7: Add pytest to requirements**

Append to `requirements.txt`:

```
# Testing
pytest>=8.0.0
```

- [ ] **Step 8: Commit**

```bash
git add pytest.ini tests/__init__.py tests/test_config_sizing.py requirements.txt config.py
git commit -m "test: add pytest harness and equity-based SIZING config" -m "Raises MAX_OPEN_POSITIONS 5 to 12; at a 2% floor, 5 slots could not reach the 65% exposure target."
```

---

### Task 2: ATR-14 indicator

**Files:**
- Modify: `data_fetcher.py:79-89` (`compute_indicators`), `data_fetcher.py:103-108` (snapshot dict)
- Test: `tests/test_data_fetcher_atr.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `compute_indicators(df)` gains an `atr_14` column. `get_market_snapshot()` entries gain `"atr_14": float | None`. Tasks 7, 8 and 10 read `market_data["atr_14"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_fetcher_atr.py`. Constant $2 ranges with no gaps make the expected ATR exactly 2.0, so this asserts a real number rather than "not None".

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_data_fetcher_atr.py -v`
Expected: FAIL — `assert 'atr_14' in out.columns` is False

- [ ] **Step 3: Implement ATR**

In `data_fetcher.py`, add above `compute_indicators`:

```python
def _true_range(df):
    """True range: the largest of the intrabar range and the two gap distances."""
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - prev_close).abs()
    low_prev_close = (df["low"] - prev_close).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
```

Then in `compute_indicators`, change the docstring and append the ATR line before `return df`:

```python
def compute_indicators(df):
    """Adds SMA-10, SMA-30, RSI-14 and ATR-14 to a single-symbol OHLCV dataframe."""
    df = df.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["atr_14"] = _true_range(df).rolling(14).mean()
    return df
```

- [ ] **Step 4: Expose ATR in the snapshot**

In `get_market_snapshot`, add one entry to the `snapshot[sym]` dict after `rsi_14`:

```python
            "atr_14": _smart_round(float(last["atr_14"])) if pd.notna(last["atr_14"]) else None,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_data_fetcher_atr.py -v`
Expected: PASS, 5 passed

- [ ] **Step 6: Commit**

```bash
git add data_fetcher.py tests/test_data_fetcher_atr.py
git commit -m "feat: add ATR-14 to indicators and market snapshot" -m "Volatility-aware stops and the per-trade risk budget both need ATR; flat 3%/6% stops are too tight for crypto and too loose for low-volatility names."
```

---

### Task 3: `forecast_stats` — distribution statistics

**Files:**
- Create: `forecast_stats.py`
- Test: `tests/test_forecast_stats.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `summarize(returns: Iterable[float]) -> dict | None`. Returns `None` when fewer than 2 finite values are supplied. On success the dict has exactly: `mu`, `sigma`, `p_up`, `ir`, `conviction`, `p10`, `p90`, `n_paths`. All are decimals (0.03 means +3%), except `n_paths` which is an `int`. Tasks 4, 5, 6 and 10 consume `conviction` and `mu`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forecast_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_forecast_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecast_stats'`

- [ ] **Step 3: Write the implementation**

Create `forecast_stats.py`:

```python
"""
Turns a set of Monte-Carlo forecast paths into distribution statistics.

Kronos is a stochastic sampler: drawing it K times gives K plausible futures.
The spread across those futures is the model's own uncertainty, and it is what
position sizing needs. A +3% forecast that every path agrees on is a very
different bet from a +3% forecast averaged out of paths ranging -8% to +14%.

Pure functions only — no model, no network, no config.
"""
import numpy as np

# Keeps conviction finite when every path lands on an identical value.
SIGMA_FLOOR = 1e-6


def summarize(returns) -> dict | None:
    """Summarize terminal returns (decimals, 0.03 == +3%) across sample paths.

    Returns None when fewer than two finite values are available, because
    dispersion — and therefore conviction — is undefined for a single path.
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    mu = float(arr.mean())
    sigma = max(float(arr.std(ddof=1)), SIGMA_FLOOR)
    ir = mu / sigma

    return {
        "mu": mu,
        "sigma": sigma,
        "p_up": float((arr > 0).mean()),
        "ir": ir,
        "conviction": abs(ir),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_paths": int(arr.size),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_forecast_stats.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add forecast_stats.py tests/test_forecast_stats.py
git commit -m "feat: add forecast_stats for Monte-Carlo path dispersion" -m "Produces mu, sigma, p_up and conviction=abs(mu/sigma) so sizing can distinguish an agreed forecast from a scattered one."
```

---

### Task 4: `position_sizer` — conviction and risk to shares

**Files:**
- Create: `position_sizer.py`
- Test: `tests/test_position_sizer.py`

**Interfaces:**
- Consumes: `config.SIZING` shape from Task 1 (passed in as a dict, never imported).
- Produces:
  - `conviction_weight_pct(conviction: float, sizing: dict) -> float`
  - `atr_budget_dollars(equity: float, price: float, atr: float | None, sizing: dict) -> float | None`
  - `quantity_for(dollars: float, price: float) -> float`
  - `size_position(*, price, equity, cash, atr, conviction, remaining_budget, sizing) -> dict` with keys `qty`, `dollars`, `weight_pct`, `binding`, `rejected_reason`. `qty == 0` and a non-empty `rejected_reason` means not fundable. `binding` is one of `"conviction"`, `"atr_risk"`, `"exposure_budget"`, `"cash"`. Tasks 5, 8 and 10 call `size_position`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_position_sizer.py`. Numbers are chosen so each constraint binds in exactly one test.

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_position_sizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'position_sizer'`

- [ ] **Step 3: Write the implementation**

Create `position_sizer.py`:

```python
"""
Converts one forecast's conviction into a dollar amount and an order quantity.

Two independent constraints are computed and the smallest wins: a conviction
weight (how strongly the model believes) and an ATR risk budget (how much a
stop-out would cost). Portfolio budget and cash act as further ceilings.

Literal Kelly is deliberately not used. With mu=3% and sigma=5% over a 10-hour
horizon, f* = mu/sigma^2 = 12, i.e. 1200% of equity; even quarter-Kelly
saturates the cap on every signal, making it indistinguishable from flat
sizing. The dispersion-normalized form below comes from the same family but
scales smoothly across the configured band.

Pure functions only — sizing config is passed in, never imported.
"""
import math


def conviction_weight_pct(conviction: float, sizing: dict) -> float:
    """Position weight as a % of equity, clamped to the configured band."""
    raw = sizing["max_position_pct"] * conviction / sizing["ir_saturation"]
    return min(max(raw, sizing["min_position_pct"]), sizing["max_position_pct"])


def atr_budget_dollars(equity: float, price: float, atr, sizing: dict):
    """Largest notional whose ATR stop-out costs at most risk_per_trade_pct.

    Returns None when ATR is unavailable, leaving the other constraints to bind.
    """
    if not atr or atr <= 0 or price <= 0:
        return None
    stop_fraction = (sizing["atr_stop_multiple"] * atr) / price
    if stop_fraction <= 0:
        return None
    return equity * (sizing["risk_per_trade_pct"] / 100.0) / stop_fraction


def quantity_for(dollars: float, price: float) -> float:
    """Whole shares at or above $1; fractional units below (Alpaca crypto)."""
    if price <= 0 or dollars <= 0:
        return 0.0
    raw = dollars / price
    if price >= 1.0:
        return float(math.floor(raw))
    return round(raw, 2)


def size_position(
    *,
    price: float,
    equity: float,
    cash: float,
    atr,
    conviction: float,
    remaining_budget: float,
    sizing: dict,
) -> dict:
    """Size one candidate. qty == 0 with a rejected_reason means not fundable."""
    result = {
        "qty": 0.0,
        "dollars": 0.0,
        "weight_pct": 0.0,
        "binding": None,
        "rejected_reason": "",
    }

    if price <= 0 or equity <= 0:
        result["rejected_reason"] = f"Invalid price (${price}) or equity (${equity})"
        return result

    if conviction < sizing["min_information_ratio"]:
        result["rejected_reason"] = (
            f"Conviction {conviction:.2f} below minimum "
            f"{sizing['min_information_ratio']:.2f}"
        )
        return result

    weight_pct = conviction_weight_pct(conviction, sizing)
    limits = {
        "conviction": equity * weight_pct / 100.0,
        "exposure_budget": max(remaining_budget, 0.0),
        "cash": max(cash, 0.0),
    }
    atr_dollars = atr_budget_dollars(equity, price, atr, sizing)
    if atr_dollars is not None:
        limits["atr_risk"] = atr_dollars

    binding = min(limits, key=limits.get)
    qty = quantity_for(limits[binding], price)
    notional = qty * price
    floor_dollars = equity * sizing["min_position_pct"] / 100.0

    if qty <= 0 or notional < floor_dollars:
        result["rejected_reason"] = (
            f"Notional ${notional:,.2f} below floor ${floor_dollars:,.2f} "
            f"({binding} binding at ${limits[binding]:,.2f})"
        )
        return result

    result.update(
        qty=qty,
        dollars=notional,
        weight_pct=weight_pct,
        binding=binding,
    )
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_position_sizer.py -v`
Expected: PASS, 19 passed

- [ ] **Step 5: Commit**

```bash
git add position_sizer.py tests/test_position_sizer.py
git commit -m "feat: add position_sizer with conviction and ATR risk constraints" -m "Takes the tighter of a dispersion-normalized conviction weight and an ATR risk budget, clamped by portfolio budget and cash. Below-floor candidates are rejected rather than shrunk, which is what prevents 1-share stub positions."
```

---

### Task 5: `portfolio_allocator` — rank and fund

**Files:**
- Create: `portfolio_allocator.py`
- Test: `tests/test_portfolio_allocator.py`

**Interfaces:**
- Consumes: `position_sizer.size_position` (Task 4), `config.SIZING` shape (Task 1).
- Produces:
  - `exposure_budget(*, equity, long_market_value, short_market_value, cash, sizing) -> float`
  - `allocate(candidates, *, equity, cash, budget, open_positions, max_open_positions, sizing) -> list[dict]`. Each candidate dict must carry `symbol`, `price`, `atr`, `conviction`. Each returned plan is the candidate merged with the sizing result plus `funded: bool`. Order is funding order (highest conviction first). Task 8 and Task 10 call both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_portfolio_allocator.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_portfolio_allocator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'portfolio_allocator'`

- [ ] **Step 3: Write the implementation**

Create `portfolio_allocator.py`:

```python
"""
Decides who gets funded when several symbols compete for the same capital.

Previously the bot sized each symbol against whatever cash remained at the
moment it happened to be evaluated, so iteration order decided allocation and
the first symbol of the cycle took the largest slice. Here every candidate is
scored first, then funded in conviction order against a portfolio-wide budget.

Pure functions only — the broker is never touched.
"""
from position_sizer import size_position


def exposure_budget(
    *,
    equity: float,
    long_market_value: float,
    short_market_value: float,
    cash: float,
    sizing: dict,
) -> float:
    """Dollars still available to deploy, never exceeding cash (no margin)."""
    current_exposure = abs(long_market_value) + abs(short_market_value)
    target = equity * sizing["target_exposure_pct"] / 100.0
    return max(min(target - current_exposure, cash), 0.0)


def allocate(
    candidates,
    *,
    equity: float,
    cash: float,
    budget: float,
    open_positions: int,
    max_open_positions: int,
    sizing: dict,
) -> list[dict]:
    """Fund candidates by descending conviction until budget or slots run out.

    Returns every candidate — funded or not — so callers can log why a signal
    did not fill.
    """
    ranked = sorted(candidates, key=lambda c: c["conviction"], reverse=True)
    remaining_budget = max(budget, 0.0)
    remaining_cash = max(cash, 0.0)
    slots = max(max_open_positions - open_positions, 0)

    plans = []
    for candidate in ranked:
        plan = dict(candidate)

        if slots <= 0:
            plan.update(
                funded=False,
                qty=0.0,
                dollars=0.0,
                weight_pct=0.0,
                binding=None,
                rejected_reason=(
                    f"Max open positions ({max_open_positions}) reached"
                ),
            )
            plans.append(plan)
            continue

        if remaining_budget <= 0:
            plan.update(
                funded=False,
                qty=0.0,
                dollars=0.0,
                weight_pct=0.0,
                binding=None,
                rejected_reason="Exposure budget full",
            )
            plans.append(plan)
            continue

        result = size_position(
            price=candidate["price"],
            equity=equity,
            cash=remaining_cash,
            atr=candidate.get("atr"),
            conviction=candidate["conviction"],
            remaining_budget=remaining_budget,
            sizing=sizing,
        )
        funded = result["qty"] > 0
        plan.update(result, funded=funded)

        if funded:
            remaining_budget -= result["dollars"]
            remaining_cash -= result["dollars"]
            slots -= 1

        plans.append(plan)

    return plans
```

- [ ] **Step 4: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_portfolio_allocator.py -v`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio_allocator.py tests/test_portfolio_allocator.py
git commit -m "feat: add portfolio_allocator to rank and fund candidates" -m "Replaces first-come-first-served allocation, where iteration order decided who got capital, with conviction ranking against a portfolio exposure budget capped by cash."
```

---

### Task 6: Kronos draws real sample paths

**Files:**
- Modify: `kronos_decision.py:123-172` (the whole `try` block of `get_kronos_decision`)
- Test: `tests/test_kronos_decision.py`

**Interfaces:**
- Consumes: `forecast_stats.summarize` (Task 3), `config.KRONOS_SAMPLE_PATHS` (Task 1).
- Produces: `get_kronos_decision(symbol, bars_df)` returns its existing keys (`symbol`, `action`, `confidence`, `reason`, `provider`) plus flat numeric fields `mu_pct`, `sigma_pct`, `conviction`, `p_up`. `confidence` is now `round(100 * p_up)` for BUY and `round(100 * (1 - p_up))` for SELL. Task 8 reads `conviction` and `mu_pct`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kronos_decision.py`. A stub predictor replaces the real model so this runs in milliseconds with no torch, no weights and no network.

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_kronos_decision.py -v`
Expected: FAIL — `test_sample_count_must_be_one_to_preserve_dispersion` errors because the current code calls `predict`, not `predict_batch`, so `StubPredictor` has no matching attribute.

- [ ] **Step 3: Add the import**

At the top of `kronos_decision.py`, below `import config`:

```python
from forecast_stats import summarize
```

- [ ] **Step 4: Replace the inference block**

Replace everything from `try:` (line 123) through the end of the `try` body's `return` (line 165) — keep the trailing `except Exception as e:` block exactly as it is — with:

```python
    try:
        n_paths = getattr(config, "KRONOS_SAMPLE_PATHS", 30)

        # One series per path with sample_count=1 gives K independent futures in
        # a single batched pass. sample_count > 1 would make Kronos average the
        # paths internally (model/kronos.py:465-467), which is exactly the
        # dispersion we need to measure.
        path_dfs = predictor.predict_batch(
            df_list=[input_df] * n_paths,
            x_timestamp_list=[x_timestamp] * n_paths,
            y_timestamp_list=[y_timestamp] * n_paths,
            pred_len=10,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        last_close = float(input_df["close"].iloc[-1])
        returns = [
            (float(path["close"].iloc[-1]) - last_close) / last_close
            for path in path_dfs
        ]
        stats = summarize(returns)

        if stats is None:
            return {
                "symbol": symbol, "action": "HOLD", "confidence": 0,
                "reason": (
                    f"Too few usable forecast paths ({len(returns)}); "
                    "dispersion is undefined."
                ),
                "provider": f"Kronos (Local / {KRONOS_MODEL_SIZE})",
                "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
            }

        mu_pct = stats["mu"] * 100
        sigma_pct = stats["sigma"] * 100
        signal_threshold = getattr(config, "KRONOS_SIGNAL_THRESHOLD_PCT", 2.5)
        min_confidence = getattr(config, "MIN_TRADE_CONFIDENCE", 70)

        if mu_pct > signal_threshold:
            action = "BUY"
            confidence = round(100 * stats["p_up"])
        elif mu_pct < -signal_threshold:
            action = "SELL"
            confidence = round(100 * (1 - stats["p_up"]))
        else:
            action = "HOLD"
            agreement = round(100 * max(stats["p_up"], 1 - stats["p_up"]))
            confidence = min(agreement, min_confidence - 1)

        expected_close = last_close * (1 + stats["mu"])
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": (
                f"Kronos ({KRONOS_MODEL_SIZE}) expects ${expected_close:.2f} "
                f"(now: ${last_close:.2f}, {mu_pct:+.2f}%) across "
                f"{stats['n_paths']} paths. "
                f"Dispersion {sigma_pct:.2f}%, conviction {stats['conviction']:.2f}, "
                f"{round(100 * stats['p_up'])}% of paths up. "
                f"80% of paths land between {stats['p10'] * 100:+.2f}% and "
                f"{stats['p90'] * 100:+.2f}%. "
                f"Signal threshold: {signal_threshold:.2f}%"
            ),
            "provider": f"Kronos (Local / {KRONOS_MODEL_SIZE})",
            "mu_pct": round(mu_pct, 4),
            "sigma_pct": round(sigma_pct, 4),
            "conviction": round(stats["conviction"], 4),
            "p_up": stats["p_up"],
        }
```

- [ ] **Step 5: Add the stats fields to the early-return guards**

The four early returns above the `try` (missing repo, missing timestamps, too few bars) must carry the same keys so CSV logging never sees a ragged row. Add to each of those `return` dicts:

```python
            "mu_pct": 0.0, "sigma_pct": 0.0, "conviction": 0.0, "p_up": 0.0,
```

Do the same for the final `except Exception as e:` return at the bottom of the function.

- [ ] **Step 6: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_kronos_decision.py -v`
Expected: PASS, 12 passed

- [ ] **Step 7: Commit**

```bash
git add kronos_decision.py tests/test_kronos_decision.py
git commit -m "feat: measure real forecast dispersion via predict_batch" -m "predict() averages its sample paths internally, so the old p10/p90 described drift of a mean trajectory rather than model disagreement, and confidence was just 60 + pct_change*8. Drawing K paths with sample_count=1 recovers the true distribution; confidence is now path agreement."
```

---

### Task 7: Equity-based portfolio state and ATR stops

**Files:**
- Modify: `risk_manager.py:18-24` (`get_account_summary`), `risk_manager.py:34-48` (delete `calc_position_size`), `risk_manager.py:124-132` (stop/target)
- Test: `tests/test_risk_manager_stops.py`

**Interfaces:**
- Consumes: `config.SIZING` (Task 1).
- Produces:
  - `get_portfolio_state(trading_client) -> dict` with `equity`, `cash`, `long_market_value`, `short_market_value`.
  - `atr_stop_take_profit_prices(entry_price, atr, sizing, side="BUY") -> tuple[float, float] | None`
  - `stop_target_for(entry_price, atr, sizing, risk_cfg, side="BUY") -> tuple[float, float]` — ATR-based when possible, percentage-based fallback otherwise.
  - `calc_position_size` is **removed**; Task 8 removes its import.

- [ ] **Step 1: Write the failing test**

Create `tests/test_risk_manager_stops.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_risk_manager_stops.py -v`
Expected: FAIL with `ImportError: cannot import name 'atr_stop_take_profit_prices'`

- [ ] **Step 3: Add portfolio state**

In `risk_manager.py`, replace `get_account_summary` (lines 18-24) with it plus a new function:

```python
def get_account_summary(trading_client) -> dict:
    account = trading_client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "last_equity": float(getattr(account, "last_equity", account.equity)),
    }


def get_portfolio_state(trading_client) -> dict:
    """Equity, cash and current market exposure on both sides of the book."""
    account = trading_client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "long_market_value": float(getattr(account, "long_market_value", 0.0) or 0.0),
        "short_market_value": float(getattr(account, "short_market_value", 0.0) or 0.0),
    }
```

- [ ] **Step 4: Delete `calc_position_size`**

Remove lines 34-48 entirely — the cash-percentage sizing it performed is superseded by `position_sizer.size_position`, and leaving it invites accidental reuse.

- [ ] **Step 5: Add the ATR stop functions**

Append below the existing `stop_loss_take_profit_prices`:

```python
def atr_stop_take_profit_prices(entry_price, atr, sizing: dict, side="BUY"):
    """Volatility-scaled stop and target. Returns None when ATR is unusable.

    A flat percentage stop is simultaneously too tight for crypto and too loose
    for low-volatility names; ATR adapts the distance to how much the symbol
    actually moves.
    """
    if not atr or atr <= 0 or entry_price <= 0:
        return None

    stop_distance = sizing["atr_stop_multiple"] * atr
    target_distance = sizing["atr_target_multiple"] * atr

    if stop_distance >= entry_price:
        return None

    side = side.upper()
    if side == "SELL":
        stop_price = round(entry_price + stop_distance, 8)
        target_price = round(entry_price - target_distance, 8)
        if target_price <= 0:
            return None
    else:
        stop_price = round(entry_price - stop_distance, 8)
        target_price = round(entry_price + target_distance, 8)

    return stop_price, target_price


def stop_target_for(entry_price, atr, sizing: dict, risk_cfg: dict, side="BUY"):
    """ATR-based stop and target, falling back to fixed percentages."""
    prices = atr_stop_take_profit_prices(entry_price, atr, sizing, side)
    if prices is not None:
        return prices
    return stop_loss_take_profit_prices(
        entry_price,
        risk_cfg["stop_loss_pct"],
        risk_cfg["take_profit_pct"],
        side,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_risk_manager_stops.py -v`
Expected: PASS, 9 passed

- [ ] **Step 7: Commit**

```bash
git add risk_manager.py tests/test_risk_manager_stops.py
git commit -m "feat: add portfolio state and ATR-based stops" -m "Removes calc_position_size, which sized off shrinking cash so evaluation order decided allocation. Adds get_portfolio_state for equity-based budgeting and ATR stops with a percentage fallback."
```

---

### Task 8: Two-phase orchestrator

**Files:**
- Modify: `orchestrator.py` (replace `process_symbol` and `run_once`; add decision logging with a fixed schema)
- Test: `tests/test_orchestrator_flow.py`

**Interfaces:**
- Consumes: `forecast_stats` output fields on the decision dict (Task 6), `portfolio_allocator.allocate` and `exposure_budget` (Task 5), `risk_manager.get_portfolio_state` and `stop_target_for` (Task 7), `config.SIZING` (Task 1).
- Produces:
  - `DECISION_FIELDS: list[str]` — the fixed CSV schema.
  - `log_decision(row: dict) -> None`
  - `evaluate_symbol(trading_client, symbol, market_data, provider_name, use_news, *, trading_halted=False, halt_reason="") -> tuple[dict, dict | None]` returning `(decision, candidate_or_None)`. A candidate carries `symbol`, `action`, `price`, `atr`, `conviction`, `decision`.
  - `execute_plan(trading_client, plan, risk_cfg, sizing) -> dict` returning the updated decision.
  - `run_once(symbols, provider_name, use_news, risk_cfg) -> tuple[list[dict], float, float]` (signature unchanged).
  - Task 9 imports `evaluate_symbol`, `execute_plan`, `log_decision`; Task 10 imports `evaluate_symbol`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestrator_flow.py`. Fakes stand in for Alpaca so nothing touches the network.

```python
import csv

import pytest

import orchestrator


class FakeClient:
    def __init__(self, positions=(), pending=()):
        self._positions = dict(positions)
        self._pending = set(pending)
        self.submitted = []

    # risk_manager / executor surface used by the orchestrator
    def get_all_positions(self):
        return [object()] * len(self._positions)


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.config, "DECISIONS_LOG", str(tmp_path / "decisions.csv"))
    monkeypatch.setattr(orchestrator.config, "TRADES_LOG", str(tmp_path / "trades.csv"))
    return tmp_path


def _market_data(price=100.0, atr=2.0, sma_30=90.0, rsi=55.0):
    return {"close": price, "sma_10": price, "sma_30": sma_30, "rsi_14": rsi, "atr_14": atr}


def _decision(action="BUY", confidence=90, conviction=0.6, mu_pct=3.0):
    return {
        "symbol": "TEST", "action": action, "confidence": confidence,
        "reason": "stub", "provider": "stub",
        "mu_pct": mu_pct, "sigma_pct": 5.0, "conviction": conviction, "p_up": 0.9,
    }


def _patch_decision(monkeypatch, decision):
    monkeypatch.setattr(orchestrator, "get_decision", lambda *a, **k: dict(decision))


def _patch_position_checks(monkeypatch, side=None, pending=False):
    monkeypatch.setattr(orchestrator, "get_position_side", lambda c, s: side)
    monkeypatch.setattr(orchestrator, "has_pending_order", lambda c, s: pending)


# --- evaluate_symbol ---

def test_strong_signal_becomes_a_candidate(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is not None
    assert candidate["conviction"] == 0.6
    assert candidate["price"] == 100.0
    assert candidate["atr"] == 2.0


def test_evaluate_never_submits_an_order(monkeypatch):
    """Phase separation: evaluation must be side-effect free at the broker."""
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    called = []
    monkeypatch.setattr(orchestrator, "submit_bracket_order",
                        lambda *a, **k: called.append(a))
    orchestrator.evaluate_symbol(FakeClient(), "TEST", _market_data(), "stub", False)
    assert called == []


def test_low_confidence_is_vetoed(monkeypatch):
    _patch_decision(monkeypatch, _decision(confidence=50))
    _patch_position_checks(monkeypatch)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is None


def test_trend_filter_veto_is_recorded(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(sma_30=200.0), "stub", False,
    )
    assert candidate is None
    assert "Trend filter" in decision["error"]


def test_existing_long_blocks_another_buy(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch, side="long")
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is None


def test_halt_blocks_candidates(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
        trading_halted=True, halt_reason="daily loss",
    )
    assert candidate is None
    assert decision["error"] == "daily loss"


def test_sell_without_position_is_blocked_when_shorting_is_off(monkeypatch):
    monkeypatch.setattr(orchestrator.config, "ALLOW_SHORT_SELLING", False)
    _patch_decision(monkeypatch, _decision(action="SELL", mu_pct=-3.0))
    _patch_position_checks(monkeypatch, side=None)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(sma_30=200.0, rsi=45.0), "stub", False,
    )
    assert candidate is None


# --- decision logging ---

def test_decision_log_has_a_fixed_schema(isolate_logs):
    orchestrator.log_decision({"symbol": "A", "action": "BUY", "extra": "ignored"})
    with open(orchestrator.config.DECISIONS_LOG, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == orchestrator.DECISION_FIELDS


def test_legacy_log_with_a_different_header_is_rotated(isolate_logs):
    path = orchestrator.config.DECISIONS_LOG
    with open(path, "w", encoding="utf-8") as f:
        f.write("symbol,action,confidence\nA,BUY,80\n")
    orchestrator.log_decision({"symbol": "B", "action": "SELL"})
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == orchestrator.DECISION_FIELDS
    rotated = list(isolate_logs.glob("decisions_legacy_*.csv"))
    assert len(rotated) == 1


def test_missing_fields_are_written_as_blanks(isolate_logs):
    orchestrator.log_decision({"symbol": "A"})
    with open(orchestrator.config.DECISIONS_LOG, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["symbol"] == "A"
    assert row["qty"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_orchestrator_flow.py -v`
Expected: FAIL with `AttributeError: module 'orchestrator' has no attribute 'evaluate_symbol'`

- [ ] **Step 3: Update the imports**

Replace the `risk_manager` import block (lines 12-19) with:

```python
from risk_manager import (
    get_account_equity,
    get_portfolio_state,
    stop_target_for,
    is_trading_halted,
    passes_trend_filter,
    count_open_positions,
)
from portfolio_allocator import allocate, exposure_budget
```

`calc_position_size` and `stop_loss_take_profit_prices` are no longer imported here.

- [ ] **Step 4: Add the fixed-schema decision log**

Below the existing `log_row` function, add:

```python
DECISION_FIELDS = [
    "timestamp", "symbol", "action", "confidence", "reason", "provider",
    "mu_pct", "sigma_pct", "conviction", "p_up",
    "qty", "notional", "weight_pct", "trade_submitted", "error",
]


def _rotate_legacy_log(path: str):
    """Move aside a decisions log whose header predates DECISION_FIELDS.

    Appending new columns to an old file would misalign every historical row,
    and the dashboard reads it with on_bad_lines="skip".
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    if header == ",".join(DECISION_FIELDS):
        return
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base, ext = os.path.splitext(path)
    os.replace(path, f"{base}_legacy_{stamp}{ext}")


def log_decision(row: dict):
    """Append a decision using a stable column set, ignoring extra keys."""
    _rotate_legacy_log(config.DECISIONS_LOG)
    file_exists = os.path.isfile(config.DECISIONS_LOG)
    with open(config.DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=DECISION_FIELDS, extrasaction="ignore", restval="",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
```

- [ ] **Step 5: Replace `process_symbol` with `evaluate_symbol`**

Delete `process_symbol` (lines 51-168) and put in its place:

```python
def evaluate_symbol(
    trading_client,
    symbol,
    market_data,
    provider_name,
    use_news,
    *,
    trading_halted=False,
    halt_reason="",
):
    """Score one symbol and apply every veto. Submits nothing.

    Returns (decision, candidate). candidate is None when the symbol is not
    tradable this cycle; the decision has already been logged in that case.
    """
    decision = get_decision(symbol, market_data, provider_name, use_news)
    decision["timestamp"] = datetime.utcnow().isoformat()
    decision["trade_submitted"] = False
    decision["error"] = ""

    min_confidence = getattr(config, "MIN_TRADE_CONFIDENCE", 70)
    action = decision.get("action", "HOLD").upper()
    confidence = decision.get("confidence", 0)

    def veto(reason):
        decision["error"] = reason
        log_decision(decision)
        return decision, None

    if action not in ("BUY", "SELL") or confidence < min_confidence:
        log_decision(decision)
        return decision, None

    if trading_halted:
        return veto(halt_reason)

    trend_ok, trend_reason = passes_trend_filter(action, market_data)
    if not trend_ok:
        return veto(trend_reason)

    current_side = get_position_side(trading_client, symbol)
    if (action == "BUY" and current_side == "long") or (
        action == "SELL" and current_side == "short"
    ):
        return veto(f"Skipped: already holding a {current_side} position in {symbol}.")

    if has_pending_order(trading_client, symbol):
        return veto(f"Skipped: an order for {symbol} is already pending.")

    if action == "SELL" and current_side is None and not config.ALLOW_SHORT_SELLING:
        return veto(
            "Skipped: SELL signal but no position to close (short selling disabled)."
        )

    # Closing an existing position is a full liquidation, so it bypasses sizing
    # and the exposure budget entirely.
    closing = action == "SELL" and current_side == "long"

    candidate = {
        "symbol": symbol,
        "action": action,
        "price": market_data["close"],
        "atr": market_data.get("atr_14"),
        "conviction": float(decision.get("conviction", 0.0) or 0.0),
        "closing": closing,
        "decision": decision,
    }
    return decision, candidate
```

- [ ] **Step 6: Add `execute_plan`**

```python
def execute_plan(trading_client, plan, risk_cfg, sizing):
    """Submit one funded plan (or one full liquidation) and log the outcome."""
    decision = plan["decision"]
    symbol = plan["symbol"]
    action = plan["action"]
    price = plan["price"]
    qty = plan.get("qty", 0.0)

    stop_price, target_price = stop_target_for(
        price, plan.get("atr"), sizing, risk_cfg, action,
    )

    try:
        order = submit_bracket_order(
            trading_client, symbol, qty, action, stop_price, target_price
        )
        decision["trade_submitted"] = order is not None
        if order is not None:
            decision["qty"] = qty
            decision["notional"] = round(qty * price, 2)
            decision["weight_pct"] = round(plan.get("weight_pct", 0.0), 3)
            log_row(config.TRADES_LOG, {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": action,
                "qty": qty,
                "entry_price": price,
                "stop_price": stop_price,
                "target_price": target_price,
                "order_id": str(order.id),
            })
    except Exception as e:
        decision["trade_submitted"] = False
        decision["error"] = str(e)

    log_decision(decision)
    return decision
```

- [ ] **Step 7: Rewrite `run_once` as two phases**

Replace `run_once` (lines 171-228) with:

```python
def run_once(symbols, provider_name, use_news, risk_cfg):
    trading_client = get_trading_client()
    sizing = config.SIZING
    market_open = is_stock_market_open(trading_client)

    halted, halt_reason = is_trading_halted(
        trading_client,
        risk_cfg.get("max_daily_loss_pct", config.DEFAULT_RISK["max_daily_loss_pct"]),
    )

    tradable_symbols = [s for s in symbols if _is_crypto(s) or market_open]
    results = []

    for symbol in symbols:
        if symbol in tradable_symbols:
            continue
        results.append({
            "symbol": symbol, "action": "SKIPPED", "confidence": 0,
            "reason": "Stock market closed", "provider": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "trade_submitted": False, "error": "",
        })

    snapshot = get_market_snapshot(tradable_symbols)

    # Phase 1: score everything, submit nothing.
    candidates = []
    for symbol, market_data in snapshot.items():
        decision, candidate = evaluate_symbol(
            trading_client, symbol, market_data, provider_name, use_news,
            trading_halted=halted, halt_reason=halt_reason,
        )
        results.append(decision)
        if candidate is not None:
            candidates.append(candidate)

    # Phase 2: liquidations first (they free capital), then ranked entries.
    closing = [c for c in candidates if c["closing"]]
    entries = [c for c in candidates if not c["closing"]]

    for plan in closing:
        execute_plan(trading_client, plan, risk_cfg, sizing)

    if entries:
        state = get_portfolio_state(trading_client)
        budget = exposure_budget(
            equity=state["equity"],
            long_market_value=state["long_market_value"],
            short_market_value=state["short_market_value"],
            cash=state["cash"],
            sizing=sizing,
        )
        plans = allocate(
            entries,
            equity=state["equity"],
            cash=state["cash"],
            budget=budget,
            open_positions=count_open_positions(trading_client),
            max_open_positions=config.MAX_OPEN_POSITIONS,
            sizing=sizing,
        )
        for plan in plans:
            if plan["funded"]:
                execute_plan(trading_client, plan, risk_cfg, sizing)
            else:
                decision = plan["decision"]
                decision["error"] = plan["rejected_reason"]
                log_decision(decision)

    equity, cash = get_account_equity(trading_client)
    return results, equity, cash
```

- [ ] **Step 8: Run the whole suite**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: PASS, all tests from Tasks 1-8 green (11 new here)

- [ ] **Step 9: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_flow.py
git commit -m "feat: split orchestrator into evaluate and allocate phases" -m "Scoring every symbol before any order is placed lets capital go to the highest-conviction candidates instead of whichever symbol was iterated first. Adds a fixed decision-log schema with legacy rotation so new stats columns cannot misalign historical rows."
```

---

### Task 9: Background runner drives both phases

**Files:**
- Modify: `background_runner.py:16-17` (imports), `background_runner.py:61-211` (`job`)
- Test: `tests/test_background_runner_status.py`

**Interfaces:**
- Consumes: `orchestrator.evaluate_symbol`, `execute_plan`, `log_decision` (Task 8); `portfolio_allocator` (Task 5); `risk_manager.get_portfolio_state` (Task 7).
- Produces: `runner_status.json` keeps every existing key so `dashboard.py` needs no change, and gains `"phase"` with value `"evaluating"` or `"allocating"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_background_runner_status.py`:

```python
import json

import pytest

import background_runner


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    path = tmp_path / "runner_status.json"
    monkeypatch.setattr(background_runner, "STATUS_PATH", str(path))
    monkeypatch.setattr(background_runner.config, "LOG_DIR", str(tmp_path))
    return path


def test_status_keeps_the_keys_the_dashboard_reads(status_path):
    background_runner.write_status(
        background_runner._base_running_status(5, "2026-08-12T00:00:00")
    )
    status = json.loads(status_path.read_text())
    for key in (
        "state", "cycle_started_at", "current_symbol", "current_index",
        "total_symbols", "last_completed_symbol", "last_result",
        "next_run_at", "interval_minutes", "progress_pct",
    ):
        assert key in status


def test_base_status_reports_the_evaluating_phase(status_path):
    status = background_runner._base_running_status(5, "2026-08-12T00:00:00")
    assert status["phase"] == "evaluating"


def test_allocating_phase_status_is_written(status_path):
    background_runner.write_status({
        **background_runner._base_running_status(5, "2026-08-12T00:00:00"),
        "phase": "allocating",
        "current_symbol": "placing orders…",
    })
    status = json.loads(status_path.read_text())
    assert status["phase"] == "allocating"
    assert status["current_symbol"] == "placing orders…"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_background_runner_status.py -v`
Expected: FAIL — `KeyError: 'phase'` in `test_base_status_reports_the_evaluating_phase`

- [ ] **Step 3: Update imports**

Replace lines 16-17:

```python
from orchestrator import (
    execute_plan,
    evaluate_symbol,
    is_stock_market_open,
    log_decision,
)
from portfolio_allocator import allocate, exposure_budget
from risk_manager import get_account_equity, get_portfolio_state, is_trading_halted
```

- [ ] **Step 4: Add `phase` to the base status**

In `_base_running_status`, add one key to the returned dict:

```python
        "phase": "evaluating",
```

- [ ] **Step 5: Replace the per-symbol loop with two phases**

Inside `job()`, in the `else:` branch that currently loops `process_symbol`, replace the loop body (lines 123-171) with:

```python
            write_status({
                **_base_running_status(total, started_at),
                "current_symbol": "fetching market data…",
                "current_index": 0,
                "progress_pct": 0,
            })
            snapshot = get_market_snapshot(tradable)
            snapshot_items = list(snapshot.items())
            work_total = max(len(snapshot_items), 1)

            candidates = []
            for i, (symbol, market_data) in enumerate(snapshot_items, start=1):
                write_status({
                    **_base_running_status(total, started_at),
                    "current_symbol": symbol,
                    "current_index": i,
                    "total_symbols": len(snapshot_items),
                    "progress_pct": int((i - 1) / work_total * 100),
                    "last_completed_symbol": results[-1]["symbol"] if results else None,
                    "last_result": results[-1] if results else None,
                })
                logger.info(f"Analyzing {symbol} ({i}/{len(snapshot_items)})...")

                decision, candidate = evaluate_symbol(
                    trading_client,
                    symbol,
                    market_data,
                    PROVIDER_NAME,
                    USE_NEWS,
                    trading_halted=halted,
                    halt_reason=halt_reason,
                )
                results.append(decision)
                if candidate is not None:
                    candidates.append(candidate)
                logger.info(
                    f"  {symbol}: {decision.get('action')} "
                    f"(confidence={decision.get('confidence')}, "
                    f"conviction={decision.get('conviction')})"
                )

                write_status({
                    **_base_running_status(total, started_at),
                    "current_symbol": symbol,
                    "current_index": i,
                    "total_symbols": len(snapshot_items),
                    "progress_pct": int(i / work_total * 100),
                    "last_completed_symbol": symbol,
                    "last_result": decision,
                })

            _allocate_and_execute(trading_client, candidates, total, started_at)
```

- [ ] **Step 6: Add the allocation helper**

Above `job()`, add:

```python
def _allocate_and_execute(trading_client, candidates, total, started_at):
    """Close exiting positions, then fund entries in conviction order."""
    if not candidates:
        return

    write_status({
        **_base_running_status(total, started_at),
        "phase": "allocating",
        "current_symbol": "placing orders…",
        "progress_pct": 100,
    })

    sizing = config.SIZING
    closing = [c for c in candidates if c["closing"]]
    entries = [c for c in candidates if not c["closing"]]

    for plan in closing:
        decision = execute_plan(trading_client, plan, RISK_CFG, sizing)
        logger.info(f"  {plan['symbol']}: CLOSE submitted={decision['trade_submitted']}")

    if not entries:
        return

    state = get_portfolio_state(trading_client)
    budget = exposure_budget(
        equity=state["equity"],
        long_market_value=state["long_market_value"],
        short_market_value=state["short_market_value"],
        cash=state["cash"],
        sizing=sizing,
    )
    logger.info(
        f"Allocating ${budget:,.2f} across {len(entries)} candidates "
        f"(equity ${state['equity']:,.2f}, cash ${state['cash']:,.2f})"
    )

    plans = allocate(
        entries,
        equity=state["equity"],
        cash=state["cash"],
        budget=budget,
        open_positions=count_open_positions(trading_client),
        max_open_positions=config.MAX_OPEN_POSITIONS,
        sizing=sizing,
    )

    for plan in plans:
        if plan["funded"]:
            decision = execute_plan(trading_client, plan, RISK_CFG, sizing)
            logger.info(
                f"  {plan['symbol']}: {plan['action']} qty={plan['qty']} "
                f"${plan['dollars']:,.2f} ({plan['weight_pct']:.1f}% of equity, "
                f"{plan['binding']} binding) submitted={decision['trade_submitted']}"
            )
        else:
            decision = plan["decision"]
            decision["error"] = plan["rejected_reason"]
            log_decision(decision)
            logger.info(f"  {plan['symbol']}: unfunded — {plan['rejected_reason']}")
```

Add `count_open_positions` to the `risk_manager` import line from Step 3:

```python
from risk_manager import (
    count_open_positions,
    get_account_equity,
    get_portfolio_state,
    is_trading_halted,
)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_background_runner_status.py -v`
Expected: PASS, 3 passed

- [ ] **Step 8: Verify nothing else broke**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: PASS, all tests green

- [ ] **Step 9: Commit**

```bash
git add background_runner.py tests/test_background_runner_status.py
git commit -m "feat: run evaluate and allocate phases in the background runner" -m "Per-symbol progress still drives the dashboard during evaluation, then a single allocation pass funds the best candidates. runner_status.json keeps its existing keys and gains phase."
```

---

### Task 10: `plan_preview` dry-run script

**Files:**
- Create: `plan_preview.py`
- Test: manual, against the live paper account (read-only)

**Interfaces:**
- Consumes: everything from Tasks 1-8. Submits no orders.
- Produces: a CLI script printing the sizing plan.

- [ ] **Step 1: Write the script**

Create `plan_preview.py`:

```python
"""
Prints what the bot WOULD trade this cycle. Submits nothing.

Run before trusting a sizing change with real orders:
    ..\\.venv\\Scripts\\python.exe plan_preview.py
"""
import config
from data_fetcher import get_market_snapshot
from executor import get_trading_client
from orchestrator import evaluate_symbol, is_stock_market_open
from portfolio_allocator import allocate, exposure_budget
from risk_manager import count_open_positions, get_portfolio_state, stop_target_for


def main():
    trading_client = get_trading_client()
    state = get_portfolio_state(trading_client)
    sizing = config.SIZING
    market_open = is_stock_market_open(trading_client)

    print(f"Equity      ${state['equity']:>12,.2f}")
    print(f"Cash        ${state['cash']:>12,.2f}")
    print(f"Long MV     ${state['long_market_value']:>12,.2f}")
    print(f"Short MV    ${state['short_market_value']:>12,.2f}")

    budget = exposure_budget(
        equity=state["equity"],
        long_market_value=state["long_market_value"],
        short_market_value=state["short_market_value"],
        cash=state["cash"],
        sizing=sizing,
    )
    open_positions = count_open_positions(trading_client)
    print(f"Budget      ${budget:>12,.2f}  (target {sizing['target_exposure_pct']}% of equity)")
    print(f"Positions   {open_positions} open / {config.MAX_OPEN_POSITIONS} max")
    print(f"Market      {'open' if market_open else 'closed'}\n")

    symbols = [s for s in config.DEFAULT_SYMBOLS if "/" in s or market_open]
    snapshot = get_market_snapshot(symbols)

    candidates = []
    print(f"{'SYMBOL':<10} {'ACTION':<7} {'CONF':>5} {'MU%':>8} {'SIGMA%':>8} {'CONV':>6}  NOTE")
    for symbol, market_data in snapshot.items():
        decision, candidate = evaluate_symbol(
            trading_client, symbol, market_data, config.DEFAULT_AI_PROVIDER, False,
        )
        print(
            f"{symbol:<10} {decision.get('action', ''):<7} "
            f"{decision.get('confidence', 0):>5} "
            f"{decision.get('mu_pct', 0):>8.2f} "
            f"{decision.get('sigma_pct', 0):>8.2f} "
            f"{decision.get('conviction', 0):>6.2f}  "
            f"{decision.get('error', '')}"
        )
        if candidate is not None:
            candidates.append(candidate)

    entries = [c for c in candidates if not c["closing"]]
    closing = [c for c in candidates if c["closing"]]

    print(f"\n{len(closing)} position(s) would be closed: "
          f"{', '.join(c['symbol'] for c in closing) or 'none'}")

    if not entries:
        print("No entry candidates.")
        return

    plans = allocate(
        entries,
        equity=state["equity"],
        cash=state["cash"],
        budget=budget,
        open_positions=open_positions,
        max_open_positions=config.MAX_OPEN_POSITIONS,
        sizing=sizing,
    )

    print(f"\n{'SYMBOL':<10} {'CONV':>6} {'WEIGHT%':>8} {'QTY':>10} "
          f"{'NOTIONAL':>12} {'STOP':>10} {'TARGET':>10}  BINDING / REASON")
    funded_total = 0.0
    for plan in plans:
        if plan["funded"]:
            stop, target = stop_target_for(
                plan["price"], plan.get("atr"), sizing, config.DEFAULT_RISK,
                plan["action"],
            )
            funded_total += plan["dollars"]
            print(
                f"{plan['symbol']:<10} {plan['conviction']:>6.2f} "
                f"{plan['weight_pct']:>8.2f} {plan['qty']:>10.2f} "
                f"${plan['dollars']:>11,.2f} {stop:>10.2f} {target:>10.2f}  "
                f"{plan['binding']}"
            )
        else:
            print(
                f"{plan['symbol']:<10} {plan['conviction']:>6.2f} "
                f"{'—':>8} {'—':>10} {'—':>12} {'—':>10} {'—':>10}  "
                f"{plan['rejected_reason']}"
            )

    pct = funded_total / state["equity"] * 100 if state["equity"] else 0
    print(f"\nWould deploy ${funded_total:,.2f} ({pct:.1f}% of equity). "
          f"NO ORDERS SUBMITTED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `..\.venv\Scripts\python.exe plan_preview.py`

Expected: the account block, one row per symbol with `mu%`, `sigma%` and conviction, then a funding table. Verify by inspection:
- No orders appear in Alpaca (check the dashboard or `git status` on `logs/trades.csv` — trades.csv must be unchanged).
- Crypto symbols are rejected with a below-floor reason (expected and accepted).
- `sigma%` is materially larger than the old logged ±0.2% ranges — this confirms real dispersion is being measured.
- Funded notionals differ across symbols in line with conviction.

Note this script *does* append to `logs/decisions.csv`, because `evaluate_symbol` logs vetoed decisions. That is intended.

- [ ] **Step 3: Commit**

```bash
git add plan_preview.py
git commit -m "feat: add plan_preview dry-run for sizing inspection" -m "Prints conviction, weights, quantities and binding constraint per symbol without submitting orders, so sizing changes can be checked against a live account first."
```

---

## Post-implementation verification

- [ ] Full suite green: `..\.venv\Scripts\python.exe -m pytest -v`
- [ ] `plan_preview.py` shows funded notionals between 2% and 12% of equity, and a total near but not over 65%.
- [ ] Flatten the nine legacy −1 share shorts manually in Alpaca before starting the runner. Until then a BUY signal on one of those symbols would flip a short into a long in a single order.
- [ ] Start the runner (`..\.venv\Scripts\python.exe background_runner.py`), watch one cycle, and confirm the log shows an allocation line with a dollar budget and per-symbol `binding` reasons.
- [ ] Confirm the dashboard still renders decisions and the runner card. On first run the old `decisions.csv` is rotated to `decisions_legacy_<timestamp>.csv`, so the decisions table starts empty — expected.
