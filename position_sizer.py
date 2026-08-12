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
    # Rounding first absorbs binary representation error: a 12% weight on
    # $100k lands on $7,199.999... which would otherwise floor a share away.
    raw = round(dollars / price, 9)
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
