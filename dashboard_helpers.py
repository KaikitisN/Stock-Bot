"""Data helpers and chart builders for the AIBots-style dashboard."""

import math
from html import escape

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


def _pos_field(p, key: str, default=None):
    """Read a field from an Alpaca position object or a serialized dict."""
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def portfolio_allocation(positions, equity: float, cash: float) -> tuple[list[dict], float]:
    """Build donut-chart segments from open positions + cash.

    Uses account equity as the authoritative total. Pie slices use absolute
    values (shorts are valid) while the legend shows signed qty and value.
    """
    segments = []
    for p in positions:
        qty = float(_pos_field(p, "qty") or 0)
        price = float(_pos_field(p, "current_price") or 0)
        raw_mv = _pos_field(p, "market_value")
        market_value = float(raw_mv) if raw_mv is not None else qty * price
        symbol = str(_pos_field(p, "symbol") or "")
        segments.append({
            "label": symbol,
            "value": abs(market_value),
            "display_value": market_value,
            "qty": qty,
            "color": symbol_color(symbol),
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


# Geometry for the donut, in viewBox units. r=42 with a 16-wide stroke puts the
# inner edge at 34, reproducing the 0.68 hole ratio of the chart this replaced.
_DONUT_RADIUS = 42.0
_DONUT_STROKE = 16.0
_DONUT_CIRCUMFERENCE = 2 * math.pi * _DONUT_RADIUS
# Arc length erased between slices to read as a separator rather than a seam.
_DONUT_GAP = 1.6


def build_donut_svg(segments: list[dict], total: float) -> str:
    """Render the holdings donut as inline SVG.

    Deliberately not a Plotly figure: a single st.plotly_chart call makes the
    browser download a 4.5 MB JS chunk, which dominates first-page load on a
    small VM. This costs nothing beyond the markup itself.
    """
    drawn = [s for s in segments if s["value"] > 0]
    gross = sum(s["value"] for s in drawn)

    if not drawn or gross <= 0:
        arcs = (
            f'<circle cx="50" cy="50" r="{_DONUT_RADIUS}" fill="none" '
            f'stroke="#27272a" stroke-width="{_DONUT_STROKE}" />'
        )
    else:
        # A lone holding gets a continuous ring; a gap there is just a nick.
        gap = _DONUT_GAP if len(drawn) > 1 else 0.0
        arcs = ""
        offset = 0.0
        for s in drawn:
            arc = s["value"] / gross * _DONUT_CIRCUMFERENCE
            visible = max(arc - gap, 0.4)
            pct = s["value"] / gross * 100
            arcs += (
                f'<circle cx="50" cy="50" r="{_DONUT_RADIUS}" fill="none" '
                f'stroke="{s["color"]}" stroke-width="{_DONUT_STROKE}" '
                f'stroke-dasharray="{visible:.3f} {_DONUT_CIRCUMFERENCE - visible:.3f}" '
                f'stroke-dashoffset="{-offset:.3f}">'
                f'<title>{escape(str(s["label"]))} — '
                f'$ {s["value"]:,.2f} ({pct:.1f}%)</title>'
                f"</circle>"
            )
            offset += arc

    return (
        '<div class="cc-donut">'
        '<svg viewBox="0 0 100 100" role="img" '
        f'aria-label="Holdings allocation totalling ${total:,.2f}">'
        # Rotate so the first slice starts at 12 o'clock instead of 3.
        f'<g transform="rotate(-90 50 50)">{arcs}</g>'
        '<text class="cc-donut-total" x="50" y="49.5" text-anchor="middle">'
        f"$ {total:,.2f}</text>"
        '<text class="cc-donut-caption" x="50" y="56" text-anchor="middle">'
        "Holdings</text>"
        "</svg></div>"
    )


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
        total_unrealized = sum(
            float(_pos_field(p, "unrealized_pl") or 0) for p in positions
        ) if positions else 0.0
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
        qty = float(_pos_field(p, "qty") or 0)
        price = float(_pos_field(p, "current_price") or 0)
        raw_mv = _pos_field(p, "market_value")
        value = float(raw_mv) if raw_mv is not None else qty * price
        pl = float(_pos_field(p, "unrealized_pl") or 0)
        raw_plpc = _pos_field(p, "unrealized_plpc")
        pl_pct = float(raw_plpc or 0) * 100
        rows.append({
            "symbol": str(_pos_field(p, "symbol") or ""),
            "exchange": "Alpaca",
            "qty": qty,
            "value": value,
            "pl": pl,
            "pl_pct": pl_pct,
            "price": price,
        })
    denom = abs(equity) or sum(abs(r["value"]) for r in rows) or 1
    for r in rows:
        r["allocation"] = abs(r["value"]) / denom * 100
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return rows


def range_change(points, current_equity: float) -> tuple[float, float]:
    """Dollar and percent change from the first chart point to current equity."""
    if not points:
        return 0.0, 0.0
    start = float(points[0][1])
    if start == 0:
        return 0.0, 0.0
    delta = float(current_equity) - start
    return delta, delta / start * 100.0


def _axis_money(value: float) -> str:
    if value == 0 or abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def build_equity_chart_svg(points, x_labels) -> str:
    """Gold area chart. points are (timestamp, equity). x_labels are start, mid, end."""
    if len(points) < 2:
        return '<div class="cc-empty">Not enough history for this range yet.</div>'

    values = [float(v) for _, v in points]
    data_lo, data_hi = min(values), max(values)
    span = data_hi - data_lo or max(abs(data_hi) * 0.02, 1.0)
    lo = data_lo - span * 0.12
    hi = data_hi + span * 0.12

    width, height = 640, 176
    left, right, top, bottom = 8, 78, 10, 28
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(index, value):
        x = left + plot_w * (index / (len(points) - 1))
        y = top + plot_h * (1 - (value - lo) / (hi - lo))
        return x, y

    coords = [xy(i, value) for i, value in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    base = top + plot_h
    area = f"{coords[0][0]:.1f},{base:.1f} {line} {coords[-1][0]:.1f},{base:.1f}"

    grid = ""
    for fraction in (0.25, 0.5, 0.75):
        y = top + plot_h * (1 - fraction)
        grid += (
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="rgba(255,255,255,0.06)" stroke-width="1"/>'
        )

    y_labels = (
        (data_hi, xy(0, data_hi)[1]),
        ((data_lo + data_hi) / 2, xy(0, (data_lo + data_hi) / 2)[1]),
        (data_lo, xy(0, data_lo)[1]),
    )
    y_text = ""
    for value, y in y_labels:
        y_text += (
            f'<text x="{width - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="cc-chart-label">{escape(_axis_money(value))}</text>'
        )

    x_text = ""
    anchors = (
        (left, "start", x_labels[0] if len(x_labels) > 0 else ""),
        (left + plot_w / 2, "middle", x_labels[1] if len(x_labels) > 1 else ""),
        (left + plot_w, "end", x_labels[2] if len(x_labels) > 2 else ""),
    )
    for x, anchor, label in anchors:
        x_text += (
            f'<text x="{x:.1f}" y="{height - 6}" text-anchor="{anchor}" '
            f'class="cc-chart-label">{escape(str(label))}</text>'
        )

    return (
        '<div class="cc-equity-chart">'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Portfolio value over the selected range">'
        "<defs>"
        '<linearGradient id="ccEquityFill" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#f59e0b" stop-opacity="0.38"/>'
        '<stop offset="100%" stop-color="#f59e0b" stop-opacity="0"/>'
        "</linearGradient></defs>"
        f"{grid}"
        f'<polygon points="{area}" fill="url(#ccEquityFill)"/>'
        f'<polyline class="cc-equity-line" points="{line}" fill="none" '
        'stroke="#f59e0b" stroke-width="2.25" stroke-linejoin="round" '
        'stroke-linecap="round"/>'
        f"{y_text}{x_text}"
        "</svg></div>"
    )


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
