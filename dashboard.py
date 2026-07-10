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
from orchestrator import run_once
from executor import get_trading_client, get_open_positions
from risk_manager import get_account_equity


st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("AI Trading Bot Dashboard")


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


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

KNOWN_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
    "SPY", "QQQ", "DIA", "IWM", "ARKK",
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
    "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "SHIB/USD",
]

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
except Exception as e:
    st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({e})")
    positions = []

if positions:
    pos_df = pd.DataFrame([
        {
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Avg Entry": p.avg_entry_price,
            "Current Price": p.current_price,
            "Unrealized P/L": p.unrealized_pl,
        }
        for p in positions
    ])
    st.subheader("Open Positions")
    st.dataframe(pos_df, width="stretch")

# ---- Run cycle with per-symbol Kronos progress ----
def execute_cycle():
    is_kronos = "kronos" in provider_name.lower()
    total = len(symbols)

    if is_kronos and total > 0:
        st.info(f"Running Kronos inference on {total} symbol(s)...")
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()

        results = []
        from orchestrator import log_row
        from data_fetcher import get_market_snapshot
        from ai_decision import get_decision
        from risk_manager import calc_position_size, stop_loss_take_profit_prices
        from executor import submit_bracket_order
        import config as _cfg

        trading_client_local = get_trading_client()
        from risk_manager import get_account_equity as _gae
        equity_local, _ = _gae(trading_client_local)
        snapshot = get_market_snapshot(symbols)

        if not snapshot:
            st.warning("No market data returned for the selected symbols.")
            st.session_state["last_results"] = []
            st.session_state["last_run_time"] = datetime.utcnow().isoformat()
            return

        for i, (symbol, market_data) in enumerate(snapshot.items()):
            pct = int(((i + 1) / total) * 100)
            progress_bar.progress(
                pct,
                text=f"Analyzing {symbol} ({i + 1}/{total})..."
            )
            status_text.markdown(f"**Kronos is working on:** `{symbol}`")

            decision = get_decision(symbol, market_data, provider_name, use_news)
            decision["timestamp"] = datetime.utcnow().isoformat()
            log_row(_cfg.DECISIONS_LOG, decision)

            if decision["action"] in ("BUY", "SELL") and decision.get("confidence", 0) >= 60:
                price = market_data["close"]
                qty = calc_position_size(equity_local, price, risk_cfg["max_position_pct"])
                if qty > 0:
                    stop_price, target_price = stop_loss_take_profit_prices(
                        price,
                        risk_cfg["stop_loss_pct"],
                        risk_cfg["take_profit_pct"],
                        decision["action"],
                    )
                    try:
                        order = submit_bracket_order(
                            trading_client_local,
                            symbol,
                            qty,
                            decision["action"],
                            stop_price,
                            target_price,
                        )
                        from orchestrator import log_row as _lr
                        _lr(_cfg.TRADES_LOG, {
                            "timestamp": datetime.utcnow().isoformat(),
                            "symbol": symbol,
                            "side": decision["action"],
                            "qty": qty,
                            "entry_price": price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "order_id": str(order.id),
                        })
                        decision["trade_submitted"] = True
                    except Exception as e:
                        decision["trade_submitted"] = False
                        decision["error"] = str(e)

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
            # Do NOT st.rerun() here — let the page render the results first.
            # The rerun below (after the 1s sleep) will handle the timer refresh.

    # ---- Latest AI Decisions (shown inline during auto-run so results are visible) ----
    if st.session_state.get("last_results"):
        st.subheader("Latest AI Decisions")
        df_results = pd.DataFrame(st.session_state["last_results"])
        if "timestamp" in df_results.columns:
            df_results["timestamp"] = df_results["timestamp"].apply(fmt_ts)
        st.dataframe(df_results, width="stretch")

    # Only rerun to tick the countdown — results are already rendered above.
    if not st.session_state["cycle_just_finished"]:
        time.sleep(1)
        st.rerun()
    else:
        # First render after a cycle: show results, reset flag, then start ticking.
        st.session_state["cycle_just_finished"] = False
        time.sleep(2)
        st.rerun()

else:
    last = fmt_ts(st.session_state.get("last_run_time", ""))
    if last:
        status_placeholder.info(f"Last run: {last}.")

    # ---- Latest AI Decisions (non-auto-run path) ----
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
