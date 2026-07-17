"""
Streamlit control panel for the trading bot.
Run with: streamlit run dashboard.py
"""
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from orchestrator import log_row, run_once, process_symbol, is_stock_market_open
from executor import get_trading_client, get_open_positions, liquidate_position
from risk_manager import get_account_equity, is_trading_halted
from data_fetcher import get_market_snapshot
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("AI Trading Bot Dashboard")
st_autorefresh(interval=60000, key="dashboard_autorefresh")

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
    try:
        if pd.isna(ts_value):
            return ""
    except Exception:
        pass
    try:
        dt = pd.to_datetime(ts_value, utc=True).to_pydatetime()
        local_dt = dt.astimezone()
        return local_dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return str(ts_value)


def ensure_scheduler_state():
    if "selected_symbols" not in st.session_state:
        st.session_state["selected_symbols"] = list(config.DEFAULT_SYMBOLS)
    if "auto_running" not in st.session_state:
        st.session_state["auto_running"] = False
    if "next_run_time" not in st.session_state:
        st.session_state["next_run_time"] = None
    if "cycle_running" not in st.session_state:
        st.session_state["cycle_running"] = False
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_run_time" not in st.session_state:
        st.session_state["last_run_time"] = None
    if "cycle_just_finished" not in st.session_state:
        st.session_state["cycle_just_finished"] = False


def schedule_next_run(run_interval_minutes: int):
    st.session_state["next_run_time"] = (
        pd.Timestamp.utcnow() + pd.Timedelta(minutes=run_interval_minutes)
    )


ensure_scheduler_state()

# ---- Sidebar controls ----
st.sidebar.header("Bot Configuration")

provider_name = st.sidebar.selectbox(
    "AI model / API key",
    list(config.AI_PROVIDERS.keys()),
    index=list(config.AI_PROVIDERS.keys()).index(config.DEFAULT_AI_PROVIDER),
)

use_news = st.sidebar.toggle(
    "Factor in news (Perplexity Sonar)",
    value=config.USE_NEWS_DEFAULT,
)

frequency_label = st.sidebar.selectbox(
    "Run frequency",
    list(config.FREQUENCY_OPTIONS.keys()),
    index=list(config.FREQUENCY_OPTIONS.keys()).index(config.DEFAULT_FREQUENCY),
)
run_interval_minutes = config.FREQUENCY_OPTIONS[frequency_label]

# ---- Symbol search & selection ----
st.sidebar.subheader("Symbols")

search_query = st.sidebar.text_input(
    "Search symbol (stocks & crypto)",
    placeholder="e.g. AAPL, BTC/USD, ETH..."
).strip().upper()

KNOWN_SYMBOLS = list(dict.fromkeys(config.DEFAULT_SYMBOLS + [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
    "SPY", "QQQ", "DIA", "IWM", "ARKK",
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
    "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "SHIB/USD",
]))

if search_query:
    filtered = [s for s in KNOWN_SYMBOLS if search_query in s]
else:
    filtered = KNOWN_SYMBOLS

selected = st.sidebar.multiselect(
    "Select symbols to trade",
    options=filtered if filtered else KNOWN_SYMBOLS,
    default=[
        s for s in st.session_state["selected_symbols"]
        if s in (filtered if filtered else KNOWN_SYMBOLS)
    ],
    key="symbol_multiselect",
)

custom_symbol = st.sidebar.text_input(
    "Add custom symbol (not in list above)",
    placeholder="e.g. PLTR, MSTR, APE/USD"
).strip().upper()

if custom_symbol and custom_symbol not in selected:
    selected = selected + [custom_symbol]

st.session_state["selected_symbols"] = selected
symbols = selected if selected else list(config.DEFAULT_SYMBOLS)

st.sidebar.subheader("Risk limits (% of account equity)")
max_position_pct = st.sidebar.slider(
    "Max position size", 1.0, 20.0, config.DEFAULT_RISK["max_position_pct"]
)
stop_loss_pct = st.sidebar.slider(
    "Stop-loss", 0.5, 10.0, config.DEFAULT_RISK["stop_loss_pct"]
)
take_profit_pct = st.sidebar.slider(
    "Take-profit", 0.5, 20.0, config.DEFAULT_RISK["take_profit_pct"]
)
max_daily_loss_pct = st.sidebar.slider(
    "Max daily loss (halts bot)", 0.5, 10.0, config.DEFAULT_RISK["max_daily_loss_pct"]
)

risk_cfg = {
    "max_position_pct": max_position_pct,
    "stop_loss_pct": stop_loss_pct,
    "take_profit_pct": take_profit_pct,
    "max_daily_loss_pct": max_daily_loss_pct,
}

st.sidebar.caption(
    f"Min confidence: **{config.MIN_TRADE_CONFIDENCE}** | "
    f"Max positions: **{config.MAX_OPEN_POSITIONS}** | "
    f"Trend filter: **{'on' if config.USE_TREND_FILTER else 'off'}**"
)

mode_label = "PAPER TRADING" if config.ALPACA_PAPER else "LIVE TRADING"
st.sidebar.markdown(f"**Mode:** `{mode_label}`")

run_now = st.sidebar.button("Run one cycle now", type="primary")
auto_run = st.sidebar.toggle("Auto-run continuously")

# ---- Account overview ----
col1, col2, col3 = st.columns(3)
try:
    trading_client = get_trading_client()
    equity, cash = get_account_equity(trading_client)
    positions = get_open_positions(trading_client)
    col1.metric("Account Equity", f"${equity:,.2f}")
    col2.metric("Cash", f"${cash:,.2f}")
    col3.metric("Open Positions", len(positions))

    halted, halt_reason = is_trading_halted(trading_client, max_daily_loss_pct)
    if halted:
        st.error(f"Trading halted: {halt_reason}")
except Exception as e:
    st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({e})")
    positions = []
# ---- Background Runner Live Status ----
import json as _json

st.subheader("Background Runner Status")
status_path = f"{config.LOG_DIR}/runner_status.json"
try:
    with open(status_path, "r", encoding="utf-8") as f:
        runner_status = _json.load(f)

    state = runner_status.get("state", "unknown")
    current_symbol = runner_status.get("current_symbol")
    current_index = runner_status.get("current_index", 0)
    total_symbols = runner_status.get("total_symbols", 0)

    if state == "running" and current_symbol:
        st.info(f"🔄 Processing **{current_symbol}** ({current_index}/{total_symbols})")
        if total_symbols > 0:
            st.progress(current_index / total_symbols)
    elif state == "idle":
        finished_at = runner_status.get("cycle_finished_at", "")
        st.success(f"✅ Idle — last cycle finished at {finished_at}")
    elif state == "error":
        st.error(f"❌ Runner error: {runner_status.get('error')}")
    else:
        st.warning("Background runner status unknown.")

    last_result = runner_status.get("last_result")
    if last_result:
        st.caption(
            f"Last decision: {last_result.get('symbol')} → "
            f"{last_result.get('action')} "
            f"(confidence={last_result.get('confidence')}, "
            f"submitted={last_result.get('trade_submitted')})"
        )
except FileNotFoundError:
    st.warning("Background runner hasn't started yet, or status file not found.")
except Exception as e:
    st.error(f"Could not read runner status: {e}")
# ---- Open Positions with Liquidate buttons ----
if positions:
    st.subheader("Open Positions")
    pos_df = pd.DataFrame([
        {
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Avg Entry": p.avg_entry_price,
            "Current Price": p.current_price,
            "Market Value": p.market_value,
            "Unrealized P/L": p.unrealized_pl,
        }
        for p in positions
    ])
    st.dataframe(pos_df, width="stretch")

    st.markdown("**Manually liquidate a position (sells full holding in one order):**")
    liq_cols = st.columns(min(len(positions), 6))
    for idx, pos in enumerate(positions):
        col = liq_cols[idx % 6]
        if col.button(f"Sell all {pos.symbol}", key=f"liq_{pos.symbol}"):
            try:
                order = liquidate_position(trading_client, pos.symbol)
                if order:
                    st.success(f"✅ Liquidation order submitted for {pos.symbol} (order id: {order.id})")
                else:
                    st.warning(f"No open position found for {pos.symbol}.")
            except Exception as e:
                st.error(f"❌ Failed to liquidate {pos.symbol}: {e}")


# ---- Run cycle with per-symbol Kronos progress ----
def execute_cycle():
    is_kronos = "kronos" in provider_name.lower()
    total = len(symbols)

    if is_kronos and total > 0:
        trading_client_local = get_trading_client()
        _, cash = get_account_equity(trading_client_local)
        halted, halt_reason = is_trading_halted(
            trading_client_local, risk_cfg["max_daily_loss_pct"]
        )
        market_open = is_stock_market_open(trading_client_local)
        tradable = [s for s in symbols if "/" in s or market_open]
        skipped = [s for s in symbols if s not in tradable]

        results = [
            {
                "symbol": symbol,
                "action": "SKIPPED",
                "confidence": 0,
                "reason": "Stock market closed",
                "provider": provider_name,
                "timestamp": datetime.utcnow().isoformat(),
                "trade_submitted": False,
                "error": "",
            }
            for symbol in skipped
        ]

        if halted:
            st.warning(halt_reason)
            for symbol in tradable:
                results.append({
                    "symbol": symbol,
                    "action": "HALTED",
                    "confidence": 0,
                    "reason": halt_reason,
                    "provider": provider_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "trade_submitted": False,
                    "error": halt_reason,
                })
        else:
            st.info(f"Running Kronos inference on {len(tradable)} symbol(s)...")
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()
            snapshot = get_market_snapshot(tradable)

            if not snapshot and not results:
                st.warning("No market data returned for the selected symbols.")
                st.session_state["last_results"] = []
                st.session_state["last_run_time"] = datetime.utcnow().isoformat()
                return

            tradable_count = max(len(snapshot), 1)
            for i, (symbol, market_data) in enumerate(snapshot.items()):
                pct = int(((i + 1) / tradable_count) * 100)
                progress_bar.progress(pct, text=f"Analyzing {symbol} ({i + 1}/{len(snapshot)})...")
                status_text.markdown(f"**Kronos is working on:** `{symbol}`")

                decision, cash = process_symbol(
                    trading_client_local,
                    symbol,
                    market_data,
                    provider_name,
                    use_news,
                    risk_cfg,
                    cash,
                    trading_halted=halted,
                    halt_reason=halt_reason,
                )
                results.append(decision)

            progress_bar.progress(100, text="Done!")
            status_text.markdown("**Kronos finished all symbols ✅**")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

    else:
        with st.spinner(f"Running decision cycle with {provider_name}..."):
            results, _, _ = run_once(symbols, provider_name, use_news, risk_cfg)

    st.session_state["last_results"] = results
    st.session_state["last_run_time"] = datetime.utcnow().isoformat()
    st.session_state["cycle_just_finished"] = True


# ---- Manual run ----
if run_now and not st.session_state["cycle_running"]:
    st.session_state["cycle_running"] = True
    try:
        execute_cycle()
        if st.session_state["auto_running"]:
            schedule_next_run(run_interval_minutes)
    finally:
        st.session_state["cycle_running"] = False

# ---- Auto-run toggle handling ----
if auto_run and not st.session_state["auto_running"]:
    st.session_state["auto_running"] = True
    if st.session_state["next_run_time"] is None:
        st.session_state["next_run_time"] = pd.Timestamp.utcnow()
elif not auto_run and st.session_state["auto_running"]:
    st.session_state["auto_running"] = False
    st.session_state["next_run_time"] = None

status_placeholder = st.empty()

# ---- Non-blocking auto scheduler ----
if st.session_state["auto_running"]:
    now = pd.Timestamp.utcnow()
    next_run = st.session_state["next_run_time"]

    if next_run is None:
        next_run = now
        st.session_state["next_run_time"] = next_run

    if st.session_state["cycle_running"]:
        last = fmt_ts(st.session_state.get("last_run_time", ""))
        status_placeholder.info(
            f"Auto-run is enabled. A cycle is currently running. Last run: {last if last else 'Not yet'}."
        )
    else:
        remaining = next_run - now
        secs = max(int(remaining.total_seconds()), 0)
        mins = secs // 60
        sec_rem = secs % 60

        last = fmt_ts(st.session_state.get("last_run_time", ""))
        status_placeholder.info(
            f"Last run: {last if last else 'Not yet'}. Next auto-run in {mins:02d}:{sec_rem:02d}."
        )

        if now >= next_run:
            st.session_state["cycle_running"] = True
            try:
                execute_cycle()
                schedule_next_run(run_interval_minutes)
            finally:
                st.session_state["cycle_running"] = False

    if st.session_state.get("last_results"):
        st.subheader("Latest AI Decisions")
        df_results = pd.DataFrame(st.session_state["last_results"])
        if "timestamp" in df_results.columns:
            df_results["timestamp"] = df_results["timestamp"].apply(fmt_ts)
        st.dataframe(df_results, width="stretch")

    if not st.session_state["cycle_just_finished"]:
        time.sleep(1)
        st.rerun()
    else:
        st.session_state["cycle_just_finished"] = False
        time.sleep(2)
        st.rerun()

else:
    last = fmt_ts(st.session_state.get("last_run_time", ""))
    if last:
        status_placeholder.info(f"Last run: {last}.")

    if st.session_state.get("last_results"):
        st.subheader("Latest AI Decisions")
        df_results = pd.DataFrame(st.session_state["last_results"])
        if "timestamp" in df_results.columns:
            df_results["timestamp"] = df_results["timestamp"].apply(fmt_ts)
        st.dataframe(df_results, width="stretch")

# ---- History from logs ----
st.subheader("Decision History")
try:
    hist = read_csv_with_fallback(config.DECISIONS_LOG)
    if "timestamp" in hist.columns:
        hist["timestamp"] = hist["timestamp"].apply(fmt_ts)
    st.dataframe(hist.tail(50), width="stretch")
except FileNotFoundError:
    st.info("No decisions logged yet - run a cycle to generate history.")

st.subheader("Trade History")
try:
    trades = read_csv_with_fallback(config.TRADES_LOG)
    if "timestamp" in trades.columns:
        trades["timestamp"] = trades["timestamp"].apply(fmt_ts)
    st.dataframe(trades, width="stretch")

    if not trades.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades["timestamp"],
            y=trades["entry_price"],
            mode="markers+lines",
            name="Entry Price",
        ))
        fig.update_layout(
            title="Trade Entries Over Time",
            xaxis_title="Time",
            yaxis_title="Price",
        )
        st.plotly_chart(fig, width="stretch")
except FileNotFoundError:
    st.info("No trades executed yet.")
