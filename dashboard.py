"""
Trading Command Center — single-page Streamlit dashboard.
Cash / holdings pie / background runner / decisions / orders.
Run with: streamlit run dashboard.py
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import config
from dashboard_helpers import build_donut_chart, portfolio_allocation
from dashboard_theme import inject_theme
from executor import get_open_positions, get_portfolio_history, get_trading_client
from risk_manager import get_account_summary

st.set_page_config(
    page_title="Trading Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()
st_autorefresh(interval=10000, key="dashboard_autorefresh")


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


def fetch_account_data():
    try:
        client = get_trading_client()
        summary = get_account_summary(client)
        equity = summary["equity"]
        cash = summary["cash"]
        last_equity = summary.get("last_equity", equity)
        positions = get_open_positions(client)
        history = None
        try:
            history = get_portfolio_history(client)
        except Exception:
            pass
        return client, equity, cash, last_equity, positions, history, None
    except Exception as e:
        return None, 0.0, 0.0, 0.0, [], None, str(e)


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


def render_metric_strip(cash, equity, positions, runner_status, mode_label):
    state = (runner_status or {}).get("state", "unknown")
    state_label = {
        "running": "Running",
        "idle": "Idle",
        "error": "Error",
    }.get(state, "Unknown")
    state_class = state if state in ("running", "idle", "error") else "unknown"

    cols = st.columns(5)
    cells = [
        ("Cash", f"$ {cash:,.2f}", "cash"),
        ("Equity", f"$ {equity:,.2f}", ""),
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
            st.plotly_chart(
                build_donut_chart(holdings, sum(s["value"] for s in holdings)),
                width="stretch",
                config={"displayModeBar": False},
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
        hist = read_csv_with_fallback(config.DECISIONS_LOG)
        if "timestamp" in hist.columns:
            hist = hist.copy()
            hist["timestamp"] = hist["timestamp"].apply(fmt_ts)
        if hist.empty:
            st.markdown(
                '<div class="cc-empty">No decisions logged yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(hist.tail(50), width="stretch")
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
        trades = read_csv_with_fallback(config.TRADES_LOG)
        if "timestamp" in trades.columns:
            trades = trades.copy()
            trades["timestamp"] = trades["timestamp"].apply(fmt_ts)
        if trades.empty:
            st.markdown(
                '<div class="cc-empty">No trades executed yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(trades.tail(50), width="stretch")
    except FileNotFoundError:
        st.markdown(
            '<div class="cc-empty">No trades executed yet.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    mode_label = "PAPER" if config.ALPACA_PAPER else "LIVE"
    _, equity, cash, _, positions, _, error = fetch_account_data()
    runner_status = load_runner_status()

    render_header(mode_label)

    if error:
        st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({error})")

    render_metric_strip(cash, equity, positions, runner_status, mode_label)
    render_holdings(positions, equity, cash)
    render_runner(runner_status)
    render_decisions()
    render_orders()


if __name__ == "__main__":
    main()
