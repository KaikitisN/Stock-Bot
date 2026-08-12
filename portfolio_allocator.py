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
