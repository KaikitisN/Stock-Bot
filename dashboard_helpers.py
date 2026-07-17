"""Data helpers and chart builders for the AIBots-style dashboard."""

import pandas as pd
import plotly.graph_objects as go

from dashboard_theme import CHART_LAYOUT, symbol_color


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def _chart_layout(**overrides) -> dict:
    layout = dict(CHART_LAYOUT)
    layout.update(overrides)
    return layout


def portfolio_allocation(positions, equity: float, cash: float) -> tuple[list[dict], float]:
    """Build donut-chart segments from open positions + cash.

    Uses account equity as the authoritative total. Pie slices use absolute
    values (shorts are valid) while the legend shows signed qty and value.
    """
    segments = []
    for p in positions:
        market_value = float(p.market_value) if hasattr(p, "market_value") else float(p.qty) * float(p.current_price)
        segments.append({
            "label": p.symbol,
            "value": abs(market_value),
            "display_value": market_value,
            "qty": float(p.qty),
            "color": symbol_color(p.symbol),
        })

    if cash != 0:
        segments.append({
            "label": "Cash",
            "value": abs(cash),
            "display_value": cash,
            "qty": cash,
            "color": symbol_color("CASH"),
        })

    gross = sum(s["value"] for s in segments) or abs(equity) or 1.0
    for s in segments:
        s["pct"] = s["value"] / gross * 100

    return segments, equity


def build_donut_chart(segments: list[dict], total: float) -> go.Figure:
    labels = [s["label"] for s in segments] or ["No holdings"]
    values = [s["value"] for s in segments] or [1]
    colors = [s["color"] for s in segments] or ["#64748b"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.72,
        marker=dict(colors=colors, line=dict(color="#0a0b14", width=2)),
        textinfo="none",
        hovertemplate="%{label}<br>$%{value:,.2f}<extra></extra>",
    )])
    fig.update_layout(
        **_chart_layout(
            showlegend=False,
            height=220,
            annotations=[dict(
                text=(
                    f"<b>$ {total:,.2f}</b><br>"
                    f"<span style='font-size:11px;color:#64748b'>Total Balance</span>"
                ),
                x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#f1f5f9",
            )],
        ),
    )
    return fig


def _equity_at_offset(equity_series: list[float], days_ago: int) -> float | None:
    if not equity_series:
        return None
    idx = max(0, len(equity_series) - 1 - days_ago)
    return float(equity_series[idx])


def compute_pl_periods(
    positions,
    equity: float,
    last_equity: float,
    portfolio_history=None,
) -> dict:
    """Compute P/L for 24H, 7D, 30D, and ALL from Alpaca equity history."""
    result = {"24H": equity - last_equity}

    equity_series = []
    if portfolio_history is not None:
        equity_series = [float(v) for v in (portfolio_history.equity or []) if v is not None]

    if equity_series:
        eq_7d = _equity_at_offset(equity_series, 7)
        eq_30d = _equity_at_offset(equity_series, 30)
        eq_first = float(equity_series[0])
        result["7D"] = equity - eq_7d if eq_7d is not None else result["24H"]
        result["30D"] = equity - eq_30d if eq_30d is not None else result["7D"]
        result["ALL"] = equity - eq_first
    else:
        total_unrealized = sum(float(p.unrealized_pl) for p in positions) if positions else 0.0
        result["7D"] = result["24H"]
        result["30D"] = result["24H"]
        result["ALL"] = total_unrealized

    return result


def build_pl_bar_chart(pl_data: dict) -> go.Figure:
    labels = list(pl_data.keys())
    values = [pl_data[k] for k in labels]
    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in values]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=[abs(v) for v in values],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"+$ {abs(v):,.0f}" if v >= 0 else f"-$ {abs(v):,.0f}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="%{x}: %{text}<extra></extra>",
    )])
    fig.update_layout(
        **_chart_layout(
            height=220,
            yaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.08)", zeroline=False, showticklabels=False),
            xaxis=dict(showgrid=False),
            bargap=0.35,
        ),
    )
    return fig


def latest_decisions_by_symbol(decisions_df: pd.DataFrame) -> dict:
    if decisions_df.empty or "symbol" not in decisions_df.columns:
        return {}
    df = decisions_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    latest = df.groupby("symbol").last().reset_index()
    return {row["symbol"]: row.to_dict() for _, row in latest.iterrows()}


def group_bots_by_provider(decisions: dict, symbols: list[str], provider_name: str) -> list[dict]:
    """Build active-bot groups matching the reference UI pattern."""
    bots = []
    for sym in symbols:
        dec = decisions.get(sym, {})
        action = dec.get("action", "HOLD")
        confidence = dec.get("confidence", 0)
        reason = dec.get("reason", "Awaiting first run")
        pct_str = f"+{confidence:.0f}%" if action == "BUY" else f"{confidence:.0f}%"
        bots.append({
            "symbol": sym,
            "action": action,
            "confidence": confidence,
            "status_label": f"{pct_str} LIVE" if action in ("BUY", "SELL") else "STANDBY",
            "reason": reason[:60],
            "provider": dec.get("provider", provider_name),
        })
    groups = {}
    for b in bots:
        prov = b["provider"] or provider_name
        groups.setdefault(prov, []).append(b)
    return [{"provider": k, "bots": v, "count": len(v)} for k, v in groups.items()]


def build_sparkline(prices: list[float], color: str = "#3b82f6") -> go.Figure:
    if not prices:
        prices = [0.0, 0.0]
    fig = go.Figure(data=[go.Scatter(
        y=prices,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
    )])
    fig.update_layout(
        **_chart_layout(
            height=40,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
        ),
    )
    return fig


def positions_table_data(positions, equity: float) -> list[dict]:
    rows = []
    for p in positions:
        value = float(p.market_value) if hasattr(p, "market_value") else float(p.qty) * float(p.current_price)
        pl = float(p.unrealized_pl)
        pl_pct = float(p.unrealized_plpc) * 100 if hasattr(p, "unrealized_plpc") else 0
        rows.append({
            "symbol": p.symbol,
            "exchange": "Alpaca",
            "qty": float(p.qty),
            "value": value,
            "pl": pl,
            "pl_pct": pl_pct,
            "price": float(p.current_price),
        })
    denom = abs(equity) or sum(abs(r["value"]) for r in rows) or 1
    for r in rows:
        r["allocation"] = abs(r["value"]) / denom * 100
    return rows


def build_allocation_ring(segments: list[dict]) -> go.Figure:
    """Circular allocation graphic for the positions widget."""
    labels = [s["label"] for s in segments] or ["—"]
    values = [s["value"] for s in segments] or [1]
    colors = [s["color"] for s in segments] or ["#64748b"]
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.78,
        marker=dict(colors=colors, line=dict(color="#0a0b14", width=3)),
        textinfo="none",
    )])
    n_coins = len([s for s in segments if s["label"] != "Cash"])
    fig.update_layout(
        **_chart_layout(
            showlegend=False,
            height=280,
            annotations=[dict(
                text=(
                    f"<b>{n_coins}</b><br>"
                    f"<span style='font-size:10px;color:#64748b'>STOCKS<br>IN 1 EXCHANGE</span>"
                ),
                x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#f1f5f9",
            )],
        ),
    )
    return fig
