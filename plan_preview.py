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
