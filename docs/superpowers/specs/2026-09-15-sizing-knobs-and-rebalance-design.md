# Sizing knobs + conservative rebalance — Design

**Date:** 2026-09-15  
**Status:** Approved (user: "go")

## Config
- `target_exposure_pct`: 65 → **72**
- `max_position_pct`: 12 → **10**
- `min_position_pct`: 2 → **1.5**
- New: `rebalance_conviction_gap`: **0.3**

## Rebalance
When a new entry cannot be funded under the current budget/slots, close at most one weaker open position per such entry if `new_conviction >= held_conviction + gap`. Recompute portfolio state, then allocate as today. Never close a symbol that is itself an entry this cycle or already closing.
