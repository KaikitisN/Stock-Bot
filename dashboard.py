"""
Trading Command Center — single-page Streamlit dashboard.
Cash / holdings pie / background runner / decisions / orders.
Run with: streamlit run dashboard.py
"""
import json
from datetime import datetime, timezone
from html import escape, unescape

import pandas as pd
import streamlit as st
from alpaca.trading.requests import GetPortfolioHistoryRequest

import config
from dashboard_helpers import (
    build_donut_svg,
    build_equity_chart_svg,
    portfolio_allocation,
    positions_table_data,
    range_change,
)
from dashboard_theme import inject_theme
from executor import get_open_positions, get_trading_client
from news_fetcher import news_scan
from risk_manager import get_account_summary

PORTFOLIO_RANGES = {
    "1D": ("1D", "5Min", True),
    "1M": ("1M", "1D", False),
    "1Y": ("1A", "1D", False),
    "All": ("all", "1D", False),
}

REFRESH_SECONDS = 45

st.set_page_config(
    page_title="Trading Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(
                path, encoding=encoding, on_bad_lines="skip", engine="python"
            )
    return pd.read_csv(path, encoding="latin1", on_bad_lines="skip", engine="python")


def fmt_ts(ts_value) -> str:
    """Format timestamps as readable local date/time."""
    if ts_value is None or ts_value == "":
        return ""
    try:
        if pd.isna(ts_value):
            return ""
    except Exception:
        pass

    tz_name = getattr(config, "DISPLAY_TIMEZONE", "Europe/Athens")
    try:
        from zoneinfo import ZoneInfo
        display_tz = ZoneInfo(tz_name)
    except Exception:
        display_tz = timezone.utc
        tz_name = "UTC"

    try:
        ts = pd.to_datetime(ts_value, utc=True)
        local_ts = ts.tz_convert(display_tz)
        return local_ts.strftime(f"%d %b %Y, %H:%M:%S ({tz_name})")
    except Exception:
        try:
            raw = str(ts_value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(display_tz).strftime(f"%d %b %Y, %H:%M:%S ({tz_name})")
        except Exception:
            return str(ts_value)


def _serialize_position(p) -> dict:
    return {
        "symbol": p.symbol,
        "qty": float(p.qty),
        "market_value": float(p.market_value),
        "current_price": float(p.current_price),
        "unrealized_pl": float(p.unrealized_pl),
        "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0),
    }


# TTL sits just under the refresh interval so each refresh gets fresh figures
# instead of alternating between a live call and a still-valid cache entry.
@st.cache_data(ttl=REFRESH_SECONDS - 5, show_spinner=False)
def fetch_account_snapshot() -> dict:
    """Cached Alpaca account + positions (no unused portfolio history)."""
    try:
        client = get_trading_client()
        summary = get_account_summary(client)
        positions = [_serialize_position(p) for p in get_open_positions(client)]
        market_open = False
        try:
            market_open = bool(client.get_clock().is_open)
        except Exception:
            market_open = False
        return {
            "equity": summary["equity"],
            "cash": summary["cash"],
            "last_equity": summary.get("last_equity", summary["equity"]),
            "buying_power": summary.get("buying_power", summary["cash"]),
            "market_open": market_open,
            "positions": positions,
            "error": None,
        }
    except Exception as e:
        return {
            "equity": 0.0,
            "cash": 0.0,
            "last_equity": 0.0,
            "buying_power": 0.0,
            "market_open": False,
            "positions": [],
            "error": str(e),
        }


@st.cache_data(ttl=REFRESH_SECONDS - 5, show_spinner=False)
def fetch_portfolio_series(period: str, timeframe: str, extended: bool) -> dict:
    try:
        client = get_trading_client()
        history = client.get_portfolio_history(GetPortfolioHistoryRequest(
            period=period,
            timeframe=timeframe,
            extended_hours=extended,
        ))
        points = []
        for ts, value in zip(history.timestamp or [], history.equity or []):
            if ts is None or value is None:
                continue
            equity_value = float(value)
            # Alpaca writes 0 on days before the account existed.
            if equity_value <= 0:
                continue
            points.append((int(ts), equity_value))
        return {"points": points, "error": None}
    except Exception as e:
        return {"points": [], "error": str(e)}


@st.cache_data(ttl=REFRESH_SECONDS - 5, show_spinner=False)
def fetch_news_scan(market_open: bool) -> dict:
    try:
        return news_scan(
            tuple(config.DEFAULT_SYMBOLS), market_open, include_closed=True,
        )
    except Exception as e:
        return {"error": str(e), "items": []}


def load_runner_status() -> dict | None:
    status_path = f"{config.LOG_DIR}/runner_status.json"
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        return {"state": "error", "error": str(e)}


def render_header(mode_label: str):
    now_label = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    st.markdown(
        f"""
        <div class="cc-header">
          <div>
            <div class="cc-eyebrow">
              <span class="cc-live-dot"></span>
              Live
            </div>
            <h1 class="cc-title">Trading Command Center</h1>
            <div class="cc-subtitle">Updated {now_label}</div>
          </div>
          <div>
            <span class="cc-badge {mode_label.lower()}">{mode_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_tz():
    tz_name = getattr(config, "DISPLAY_TIMEZONE", "Europe/Athens")
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _as_of_label() -> str:
    now = datetime.now(_display_tz())
    clock = now.strftime("%B %d, %I:%M %p")
    abbrev = now.tzname() or ""
    return f"{clock} {abbrev}".strip()


def _chart_labels(points) -> list[str]:
    if not points:
        return ["", "", ""]
    stamps = [p[0] for p in points]
    span = stamps[-1] - stamps[0]

    def fmt(ts):
        moment = datetime.fromtimestamp(ts, _display_tz())
        if span <= 36 * 3600:
            return moment.strftime("%I %p").lstrip("0")
        return moment.strftime("%d %b")

    mid = stamps[len(stamps) // 2]
    return [fmt(stamps[0]), fmt(mid), fmt(stamps[-1])]


def _news_text(value) -> str:
    text = unescape(str(value or ""))
    if len(text) > 420:
        return text[:417].rstrip() + "…"
    return text


def _signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def render_portfolio(equity, cash, buying_power, last_equity):
    title_col, range_col = st.columns([2.4, 1])
    with title_col:
        st.markdown(
            '<div class="cc-section" style="margin-top:0">'
            '<div class="cc-section-title">Your portfolio</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with range_col:
        if "portfolio_span" not in st.session_state:
            st.session_state.portfolio_span = "1D"
        selected = st.segmented_control(
            "Range",
            list(PORTFOLIO_RANGES),
            key="portfolio_span",
            label_visibility="collapsed",
        )
    selected = selected or "1D"
    period, timeframe, extended = PORTFOLIO_RANGES[selected]
    series = fetch_portfolio_series(period, timeframe, extended)
    points = series["points"]
    if selected == "1D" and last_equity:
        pct = (equity - last_equity) / last_equity * 100.0
    else:
        _, pct = range_change(points, equity)

    pct_class = "up" if pct >= 0 else "down"
    daily = equity - (last_equity or equity)
    daily_class = "up" if daily >= 0 else "down"
    chart = build_equity_chart_svg(points, _chart_labels(points))
    if series["error"]:
        chart = (
            '<div class="cc-empty">Chart unavailable: '
            f"{escape(series['error'])}</div>"
        )

    st.markdown(
        '<div class="cc-card cc-portfolio">'
        '<div class="cc-equity-row">'
        f'<span class="cc-equity-value">${equity:,.2f}</span>'
        f'<span class="cc-equity-pct {pct_class}">{pct:+.2f}%</span>'
        "</div>"
        f'<div class="cc-asof">{escape(_as_of_label())}</div>'
        f"{chart}"
        '<div class="cc-balances">'
        "<div>"
        '<div class="cc-metric-label">Buying Power</div>'
        f'<div class="cc-balance-value">${buying_power:,.2f}</div>'
        "</div><div>"
        '<div class="cc-metric-label">Cash</div>'
        f'<div class="cc-balance-value">${cash:,.2f}</div>'
        "</div><div>"
        '<div class="cc-metric-label">Daily Change</div>'
        f'<div class="cc-balance-value {daily_class}">{_signed_money(daily)}</div>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )


def render_news(market_open: bool):
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">News scan</div>'
        '<div class="cc-section-sub">Headlines that added a symbol, and the text sent with its forecast</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    scan = fetch_news_scan(market_open)
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)
    if scan.get("error"):
        st.markdown(
            f'<div class="cc-empty">News scan failed: {escape(scan["error"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not market_open:
        st.caption(
            "The stock market is closed. Stock names were gathered and are sent when it opens. Crypto is sent now."
        )

    items = scan.get("items") or []
    if not items:
        st.markdown(
            '<div class="cc-empty">No new names in the latest headlines. '
            "The watchlist is still forecast on its own.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    body = ""
    for item in items:
        scale = float(item.get("position_scale") or 1)
        body += (
            '<div class="cc-news-item">'
            f'<div class="cc-news-symbol">{escape(str(item.get("symbol", "")))}</div>'
            '<div class="cc-news-meta">'
            f'Confidence floor {int(item.get("confidence_floor") or 0)}'
            f" · size cap ×{scale:g}"
            "</div>"
            '<div class="cc-news-kicker">'
            + (
                "Gathered. Not sent yet, because the stock market is closed."
                if item.get("status") == "waiting"
                else "Sent to the model"
            )
            + "</div>"
            f'<p class="cc-news-sent">{escape(_news_text(item.get("sent", "")))}</p>'
            "</div>"
        )
    st.markdown(body, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_strip(positions, runner_status, mode_label):
    state = (runner_status or {}).get("state", "unknown")
    state_label = {
        "running": "Running",
        "idle": "Idle",
        "error": "Error",
    }.get(state, "Unknown")
    state_class = state if state in ("running", "idle", "error") else "unknown"

    cols = st.columns(3)
    cells = [
        ("Open Positions", str(len(positions)), ""),
        ("Runner", state_label, state_class),
        ("Mode", mode_label, ""),
    ]
    for col, (label, value, extra) in zip(cols, cells):
        value_class = f"cc-metric-value {extra}".strip()
        with col:
            st.markdown(
                f'<div class="cc-metric">'
                f'<div class="cc-metric-label">{label}</div>'
                f'<div class="{value_class}">{value}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_holdings(positions, equity, cash):
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">Holdings</div>'
        '<div class="cc-section-sub">Allocation of what you currently hold</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    segments, _ = portfolio_allocation(positions, equity, cash)
    holdings = [s for s in segments if s["label"] != "Cash"]
    holdings_total = sum(s["value"] for s in holdings) or 1.0
    for s in holdings:
        s["pct"] = s["value"] / holdings_total * 100

    st.markdown('<div class="cc-card">', unsafe_allow_html=True)
    if holdings:
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(
                build_donut_svg(holdings, sum(s["value"] for s in holdings)),
                unsafe_allow_html=True,
            )
        with right:
            legend_html = ""
            for s in holdings:
                display_val = s.get("display_value", s["value"])
                sign = "+" if display_val >= 0 else "-"
                legend_html += (
                    f'<div class="cc-legend-item">'
                    f'<span class="cc-dot" style="background:{s["color"]}"></span>'
                    f'<span><b>{s["label"]}</b> &nbsp; {s["qty"]:,.4f} &nbsp; '
                    f'<span style="color:#71717a">({sign}$ {abs(display_val):,.2f}'
                    f' · {s["pct"]:.1f}%)</span></span></div>'
                )
            st.markdown(legend_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="cc-empty">No open positions — holdings pie will appear here once you hold stock.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _fmt_money(value: float, signed: bool = False) -> str:
    if signed:
        sign = "+" if value >= 0 else "-"
        return f"{sign}$ {abs(value):,.2f}"
    if value < 0:
        return f"-$ {abs(value):,.2f}"
    return f"$ {value:,.2f}"


def _fmt_qty(qty: float) -> str:
    if abs(qty - round(qty)) < 1e-9:
        return f"{int(round(qty))}"
    return f"{qty:,.4f}".rstrip("0").rstrip(".")


def render_top_positions(positions, equity: float):
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">Top Positions</div>'
        '<div class="cc-section-sub">Open holdings by market value</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)

    rows = positions_table_data(positions, equity)
    if not rows:
        st.markdown(
            '<div class="cc-empty">No open positions.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    body = ""
    for r in rows:
        pl_pill = "pos" if r["pl"] >= 0 else "neg"
        body += (
            "<tr>"
            f'<td class="asset">{r["symbol"]}</td>'
            f'<td class="num">{_fmt_money(r["price"])}</td>'
            f'<td class="num">{_fmt_qty(r["qty"])}</td>'
            f'<td class="num">{_fmt_money(r["value"])}</td>'
            f'<td class="num">'
            f'<span class="cc-pos-pill {pl_pill}">{_fmt_money(r["pl"], signed=True)}</span>'
            "</td>"
            "</tr>"
        )

    st.markdown(
        '<table class="cc-pos-table">'
        "<thead><tr>"
        "<th>Asset</th>"
        '<th class="num">Price</th>'
        '<th class="num">Qty</th>'
        '<th class="num">Market Value</th>'
        '<th class="num">Total P/L ($)</th>'
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_runner(runner_status):
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">Background Runner</div>'
        '<div class="cc-section-sub">Live progress of the automated trading cycle</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)

    if runner_status is None:
        st.warning("Background runner hasn't started yet, or status file not found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    state = runner_status.get("state", "unknown")
    current_symbol = runner_status.get("current_symbol")
    current_index = runner_status.get("current_index", 0)
    total_symbols = runner_status.get("total_symbols", 0)
    next_run_at = runner_status.get("next_run_at")
    next_run_label = fmt_ts(next_run_at) if next_run_at else "—"

    if state == "running":
        label = current_symbol or "starting…"
        pct = runner_status.get("progress_pct")
        if pct is None and total_symbols:
            pct = int((current_index / max(total_symbols, 1)) * 100)
        st.info(
            f"Processing **{label}** "
            f"({current_index}/{total_symbols}"
            f"{f', {pct}%' if pct is not None else ''})"
        )
        if total_symbols > 0:
            st.progress(min((pct or 0) / 100.0, 1.0))
        last_completed = runner_status.get("last_completed_symbol")
        if last_completed:
            st.caption(f"Last completed: **{last_completed}**")
    elif state == "idle":
        finished_at = fmt_ts(runner_status.get("cycle_finished_at", ""))
        interval = runner_status.get("interval_minutes")
        interval_note = f" (every {interval} min)" if interval else ""
        st.success(
            f"Idle — last cycle finished at **{finished_at or '—'}**. "
            f"Next cycle at **{next_run_label}**{interval_note}."
        )
    elif state == "error":
        st.error(
            f"Runner error: {runner_status.get('error')} — "
            f"next retry at **{next_run_label}**"
        )
    else:
        st.warning("Background runner status unknown.")

    if next_run_at and state != "running":
        try:
            next_dt = pd.to_datetime(next_run_at, utc=True)
            remaining = next_dt - pd.Timestamp.now(tz="UTC")
            secs = max(int(remaining.total_seconds()), 0)
            hours, rem = divmod(secs, 3600)
            mins, sec_rem = divmod(rem, 60)
            if hours:
                countdown = f"{hours:d}h {mins:02d}m {sec_rem:02d}s"
            else:
                countdown = f"{mins:02d}:{sec_rem:02d}"
            st.caption(f"Countdown to next background cycle: **{countdown}**")
        except Exception:
            pass

    last_result = runner_status.get("last_result")
    if last_result:
        st.caption(
            f"Last decision: {last_result.get('symbol')} → "
            f"{last_result.get('action')} "
            f"(confidence={last_result.get('confidence')}, "
            f"submitted={last_result.get('trade_submitted')})"
        )

    st.markdown("</div>", unsafe_allow_html=True)


def prepare_log_table(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Sort by timestamp newest-first, format for display, keep top N rows."""
    if df.empty:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        sort_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.assign(_sort_ts=sort_ts).sort_values(
            "_sort_ts", ascending=False, na_position="last"
        )
        out = out.drop(columns=["_sort_ts"]).head(limit)
        out["timestamp"] = out["timestamp"].apply(fmt_ts)
    else:
        out = out.tail(limit).iloc[::-1]
    return out.reset_index(drop=True)


def render_decisions():
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">Trade Decisions</div>'
        '<div class="cc-section-sub">Recent AI decisions from the decision log</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)
    try:
        hist = prepare_log_table(read_csv_with_fallback(config.DECISIONS_LOG))
        if hist.empty:
            st.markdown(
                '<div class="cc-empty">No decisions logged yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(hist, width="stretch")
    except FileNotFoundError:
        st.markdown(
            '<div class="cc-empty">No decisions logged yet.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_orders():
    st.markdown(
        '<div class="cc-section">'
        '<div class="cc-section-title">Orders</div>'
        '<div class="cc-section-sub">Submitted trades from the trade log</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cc-card">', unsafe_allow_html=True)
    try:
        trades = prepare_log_table(read_csv_with_fallback(config.TRADES_LOG))
        if trades.empty:
            st.markdown(
                '<div class="cc-empty">No trades executed yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(trades, width="stretch")
    except FileNotFoundError:
        st.markdown(
            '<div class="cc-empty">No trades executed yet.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# Everything live sits in one fragment so a refresh reruns only this function.
# The previous st_autorefresh reran the whole script, re-executing module setup
# and rebuilding the entire page for the browser every 45 seconds.
@st.fragment(run_every=REFRESH_SECONDS)
def render_live():
    mode_label = "PAPER" if config.ALPACA_PAPER else "LIVE"
    snapshot = fetch_account_snapshot()
    equity = snapshot["equity"]
    cash = snapshot["cash"]
    positions = snapshot["positions"]
    error = snapshot["error"]
    runner_status = load_runner_status()

    render_header(mode_label)

    if error:
        st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({error})")

    render_portfolio(
        equity,
        cash,
        snapshot.get("buying_power", cash),
        snapshot.get("last_equity", equity),
    )
    render_news(bool(snapshot.get("market_open")))
    render_metric_strip(positions, runner_status, mode_label)
    render_holdings(positions, equity, cash)
    render_top_positions(positions, equity)
    render_runner(runner_status)
    render_decisions()
    render_orders()


def main():
    render_live()


if __name__ == "__main__":
    main()
