# Confidence-Weighted Position Sizing

**Date:** 2026-08-12
**Status:** Approved, ready for implementation planning

## Problem

The bot allocates capital badly, and the reported symptom ("the first stock goes all in,
leaving nothing for the others") turned out to be three distinct defects rather than one.

Observed live account state at time of writing:

| Metric | Value |
|---|---|
| Equity | $98,509 |
| Cash idle | $96,650 |
| Long market value | $5,660 |
| Short market value | −$3,801 |
| Capital deployed | ~9.6% of equity |
| Open positions | 11 (2 real, 9 stubs of exactly −1 share) |

### Defect 1: shorts bypass sizing entirely

`orchestrator.process_symbol` calls `calc_position_size()` only on the BUY branch. The SELL
branch hardcodes `qty = 1`, which is why every short position is exactly −1 share regardless
of price, volatility, or conviction. MU (−1 share, $912) and NFLX (−1 share, $74) received
identical "sizing".

### Defect 2: `MAX_OPEN_POSITIONS` is not enforced

The cap check sits behind `if action == "BUY"`, so short entries were never counted against
it. The configured limit is 5; 11 positions are open.

### Defect 3: sizing is off cash, not equity, and floors to whole units

`calc_position_size(cash, price, max_position_pct)` takes `max_position_pct` of *available
cash*. Cash shrinks as positions open, so each successive symbol in a cycle is sized against
a smaller base — the first symbol evaluated gets the largest dollar allocation purely because
of evaluation order. Combined with `max_position_pct = 3.0` and `MAX_OPEN_POSITIONS = 5`, the
bot can never deploy more than roughly 15% of the account.

Additionally, `int(raw_qty)` floors quantities for anything priced at or above $1, so BTC/USD
at $64k with a 3%-of-cash budget rounds to **0 units**. Only sub-$1 assets receive fractional
sizing.

### Defect 4 (root cause): "confidence" contains no information about uncertainty

`kronos_decision.py` manufactures confidence from the forecast magnitude:

```python
confidence = min(int(60 + pct_change * 8), 95)
```

Confidence is therefore a monotone relabeling of `pct_change`. Sizing by confidence would be
identical to sizing by predicted move, taking the largest positions in the noisiest names.

Worse, the uncertainty estimate that *is* computed is not measuring uncertainty. Kronos
averages its sample paths internally before `predict()` returns
(`Kronos/model/kronos.py:465-469`):

```python
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)
return preds
```

The `p10`/`p90` values labelled "80% range" in the decision log are quantiles across the
**10 time steps of a single averaged path**, describing how the mean forecast drifts over the
horizon — not model disagreement. This is why logged ranges are implausibly tight, e.g.
"BTC/USD 80% range: $64,266–$64,519", a ±0.2% band on a 10-hour Bitcoin forecast.

## Goals

1. Position size scales with genuine forecast conviction, not forecast magnitude alone.
2. Capital spreads across multiple opportunities instead of being consumed by evaluation order.
3. Both sides of the book are sized by the same logic; stub positions become impossible.
4. Exposure and position-count caps are enforced portfolio-wide.

## Non-goals

- Correlation / sector cluster caps. Deferred. SPY, QQQ, AAPL, MSFT, NVDA and GOOGL are
  substantially one bet, but limiting that is a separate change.
- Backtest / simulation harness. Deferred.
- Fractional crypto sizing. Deferred by explicit decision — see Accepted Consequences.
- Cleaning up the nine existing −1 share shorts. The user will flatten these manually in
  Alpaca before the new code runs.

## Constraints (user-specified)

| Parameter | Value |
|---|---|
| Target total exposure when fully invested | 60–70% of equity (65% default) |
| Per-position floor | 2% of equity |
| Per-position cap | 12% of equity |
| Margin | Never. Total exposure capped at cash on hand. |
| Account | Alpaca paper, ~$98.5k equity |

## Architecture

Three new pure-function modules with no I/O, plus targeted edits to four existing files.
Pure modules are unit-testable without a network call or a loaded model.

| Module | Responsibility | Depends on |
|---|---|---|
| `forecast_stats.py` (new) | Sample paths → distribution statistics | numpy |
| `position_sizer.py` (new) | Stats + equity + ATR → dollars and share quantity | config |
| `portfolio_allocator.py` (new) | Rank candidates, spend the exposure budget | config |
| `kronos_decision.py` (edit) | Draw K paths, attach stats, emit calibrated confidence | forecast_stats |
| `data_fetcher.py` (edit) | Add ATR-14 to `compute_indicators` | — |
| `risk_manager.py` (edit) | Equity-based budget, caps on both sides, ATR stops | — |
| `orchestrator.py` (edit) | Split evaluate phase from execute phase | allocator, sizer |

`position_sizer` answers "how large should this bet be in isolation". `portfolio_allocator`
answers "given everything competing for the same capital, who gets funded". Keeping them
separate allows the conviction math to change without touching budget logic.

### Data flow

```
bars → compute_indicators (+ATR-14)
     → kronos_decision: predict_batch K paths
     → forecast_stats: mu, sigma, p_up, conviction
     → gates: confidence, signal threshold, min_information_ratio, trend, duplicate, halt
     → candidate list
     → portfolio_allocator: rank by conviction, spend budget
     → position_sizer: dollars → qty
     → executor: bracket order with ATR stops
```

## Component 1: Real uncertainty from Kronos

`predict_batch` accepts a list of series and batches them along the batch dimension. Passing
the same dataframe K times with `sample_count=1` yields K independent single-sample paths in
one batched forward pass, at roughly the compute cost of the current 50-sample averaged call.

```python
paths = predictor.predict_batch(
    df_list=[input_df] * KRONOS_SAMPLE_PATHS,
    x_timestamp_list=[x_timestamp] * KRONOS_SAMPLE_PATHS,
    y_timestamp_list=[y_timestamp] * KRONOS_SAMPLE_PATHS,
    pred_len=10, T=1.0, top_p=0.9, sample_count=1, verbose=False,
)
returns = [(float(p["close"].iloc[-1]) - last_close) / last_close for p in paths]
```

`sample_count=1` is required: any value above 1 re-triggers the internal averaging that
destroys dispersion. The vendored `Kronos/` directory is **not** modified.

### `forecast_stats.summarize(returns) -> dict`

| Field | Definition |
|---|---|
| `mu` | mean of terminal returns (decimal) |
| `sigma` | standard deviation of terminal returns (decimal) |
| `p_up` | fraction of paths with a positive terminal return |
| `ir` | `mu / sigma`, signed — negative for a bearish forecast |
| `conviction` | `abs(ir)`, the direction-agnostic conviction score |
| `p10`, `p90` | 10th and 90th percentile terminal return across paths |

Direction comes from the sign of `mu`; magnitude comes from `conviction`. All sizing,
ranking, and threshold comparisons use `conviction` so that long and short candidates are
treated symmetrically. `ir` is retained in the decision log for readability.

`sigma` is clamped to a small positive floor (1e-6) to keep `ir` finite when all paths
collapse to an identical value.

### Calibrated confidence

`confidence = round(100 * p_up)` for BUY, `round(100 * (1 - p_up))` for SELL.

The existing `MIN_TRADE_CONFIDENCE = 70` gate is unchanged numerically but now means "at
least 70% of sampled futures agree on direction". The `p10`/`p90` in the reason string become
genuine cross-path percentiles.

## Component 2: Sizing math

### Why not literal Kelly

With `mu = 3%` and `sigma = 5%` over a 10-hour horizon, `f* = mu / sigma² = 12`, i.e. 1200%
of equity. Quarter-Kelly still gives 300%. Every signal would saturate the 12% cap, making
Kelly indistinguishable from flat sizing. Kelly also assumes the edge estimate is correct,
and the Kronos edge is unvalidated. We use the dispersion-normalized form from the same
family, which does not saturate.

### Conviction weight

```
weight = clamp(MIN_POSITION_PCT, MAX_POSITION_PCT,
               MAX_POSITION_PCT * conviction / IR_SATURATION)
```

Behaviour at `IR_SATURATION = 1.0` with a 2–12% band:

| Case | mu | sigma | conviction | Weight |
|---|---|---|---|---|
| Strong, paths agree | +3% | 3% | 1.00 | 12% (cap) |
| Typical signal | +3% | 5% | 0.60 | 7.2% |
| Same forecast, scattered paths | +3% | 15% | 0.20 | 2% (floor) |
| Weak and scattered | +1% | 15% | 0.07 | rejected |

Candidates with `conviction < min_information_ratio` are rejected before reaching the clamp,
so the floor is never applied to a signal too weak to qualify.

An identical predicted move produces a materially different bet depending on model agreement.
This is the central requirement.

### Final dollar amount

The conviction weight and an independent ATR risk budget are both computed; the smaller wins.

```
stop_distance = ATR_STOP_MULTIPLE * atr
risk_dollars  = equity * RISK_PER_TRADE_PCT
dollars = min(
    equity * weight,                        # conviction
    risk_dollars / (stop_distance / price), # equal risk per trade
    remaining_exposure_budget,              # portfolio budget
    available_cash,                         # no margin
)
```

Quantity is `floor(dollars / price)` for stocks, and for sub-$1 assets a fractional quantity
rounded to 2 decimals (existing behaviour preserved).

**Rejection rather than shrinkage:** if the resulting notional falls below
`MIN_POSITION_PCT` of equity, the candidate is skipped and logged. This structurally prevents
stub positions and is the fix for Defect 1.

### ATR-based stops

`compute_indicators` gains ATR-14 using true range
(`max(high−low, |high−prev_close|, |low−prev_close|)`), exposed in the market snapshot as
`atr_14`. Stops and targets become `entry ∓ ATR_STOP_MULTIPLE * atr` and
`entry ± ATR_TARGET_MULTIPLE * atr`, replacing flat 3%/6% which is simultaneously too tight
for crypto and too loose for low-volatility names. `stop_loss_take_profit_prices()` keeps its
percentage path as the fallback when ATR is unavailable.

## Component 3: Rank-then-allocate

`process_symbol` splits into two functions:

- `evaluate_symbol(...) -> candidate | None` — runs the decision and every veto (halt,
  confidence, signal threshold,   `min_information_ratio`, trend filter, duplicate position,
  pending order). Submits nothing. Returns a candidate carrying the decision plus `conviction`,
  `mu`, `sigma`, `price`, and `atr`.
- `execute_plan(...)` — sizes and submits a single funded candidate.

The cycle becomes:

1. Evaluate every symbol, collecting candidates and logging vetoed decisions.
2. `budget = equity * TARGET_EXPOSURE_PCT − current_exposure`, where `current_exposure` is
   `long_market_value + abs(short_market_value)`, additionally capped by available cash.
3. Sort candidates by `conviction` descending.
4. Fund down the list, decrementing the budget, until budget is exhausted or
   `MAX_OPEN_POSITIONS` is reached.
5. Submit funded candidates; log unfunded ones with reason `"exposure budget full"` or
   `"max open positions reached"`.

`MAX_OPEN_POSITIONS` rises from 5 to 12 — at a 2% floor and 65% target exposure, 5 was
arithmetically incompatible with diversification. The cap counts positions on both sides,
fixing Defect 2.

Shorts are sized by the identical path using `abs(mu)`, mirrored. This only activates when
`ALLOW_SHORT_SELLING=true`, which is currently `false`; SELL continues to mean "close an
existing long" until that is turned on.

### Impact on `background_runner`

`background_runner.job()` currently calls `process_symbol` per symbol to drive the dashboard
progress bar. It is restructured to report per-symbol progress during the evaluate phase, then
write a `"placing orders"` status during the execute phase. `runner_status.json` keeps its
existing shape so `dashboard.py` needs no changes.

## Configuration

New `SIZING` block in `config.py`, each key overridable by environment variable:

| Key | Default | Meaning |
|---|---|---|
| `target_exposure_pct` | 65.0 | Total deployed capital target, % of equity |
| `min_position_pct` | 2.0 | Reject candidates below this notional |
| `max_position_pct` | 12.0 | Hard per-position cap |
| `risk_per_trade_pct` | 0.5 | Equity risked per trade at the ATR stop |
| `atr_stop_multiple` | 2.0 | Stop distance in ATR units |
| `atr_target_multiple` | 4.0 | Target distance in ATR units |
| `ir_saturation` | 1.0 | `conviction` value at which the position cap is reached |
| `min_information_ratio` | 0.2 | Reject candidates with `conviction` below this |
| `kronos_sample_paths` | 30 | K independent paths drawn per symbol |

`MAX_OPEN_POSITIONS` default changes from 5 to 12. `DEFAULT_RISK` is retained for
`max_daily_loss_pct` and as the ATR fallback percentages.

## Error handling

Every failure degrades toward not trading, never toward trading blind.

| Failure | Behaviour |
|---|---|
| ATR unavailable or NaN | Fall back to flat `stop_loss_pct` / `take_profit_pct` |
| `predict_batch` raises | Return HOLD, log the error, no order |
| Fewer than 2 usable paths | Return HOLD (`sigma` undefined) |
| `sigma` collapses to ~0 | Clamp to 1e-6; cap keeps `ir` from exploding |
| Exposure budget exhausted | Log candidate as unfunded with explicit reason |
| Equity or cash query fails | Abort the cycle, log, retry next interval |

## Testing

`Stock-Bot` currently has no tests. This change introduces pytest with unit tests over the
three pure modules — no network, no model.

- `forecast_stats`: tight vs scattered path sets; `p_up` at boundaries; degenerate all-equal
  paths; single-path input.
- `position_sizer`: conviction cap and floor clamping; ATR budget binding instead of
  conviction; below-floor rejection; no-margin cash clamp; whole-share flooring vs sub-$1
  fractional.
- `portfolio_allocator`: ranking order by `conviction`; budget exhaustion mid-list; position cap;
  correct exposure arithmetic with existing shorts.

A `plan_preview.py` script prints the full sizing plan for the live universe — candidates,
`conviction`, weights, dollar amounts, quantities, and who gets funded — **without submitting
any order**, so the numbers can be inspected before the bot trades on them.

## Accepted consequences

**8 of the 22 configured symbols are crypto and will all be rejected.** The 2% floor is
~$1,970 while BTC trades near $64k, and quantities floor to whole units above $1, so
BTC/ETH/SOL/LINK/XRP/PAXG/BNB/MKR size to zero. Fractional crypto sizing was explicitly
deferred; crypto remains untradeable until it is implemented.

**The nine existing −1 share shorts are left in place.** They will be flattened manually. Note
that until they are, a BUY signal on any of those symbols is not blocked by the duplicate
guard (which only catches long-on-long) and would submit a full-size buy that flips the short
into a long in a single order.
