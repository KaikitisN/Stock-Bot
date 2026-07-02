"""
Streamlit control panel for the trading bot.
Run with: streamlit run dashboard.py
"""
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
from orchestrator import run_once
from executor import get_trading_client, get_open_positions
from risk_manager import get_account_equity

st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("AI Trading Bot Dashboard")

# ---- Sidebar controls ----
st.sidebar.header("Bot Configuration")

provider_name = st.sidebar.selectbox(
    "AI model / API key",
    list(config.AI_PROVIDERS.keys()),
    index=list(config.AI_PROVIDERS.keys()).index(config.DEFAULT_AI_PROVIDER)
)

use_news = st.sidebar.toggle("Factor in news (Perplexity Sonar)", value=config.USE_NEWS_DEFAULT)

frequency_label = st.sidebar.selectbox(
    "Run frequency",
    list(config.FREQUENCY_OPTIONS.keys()),
    index=list(config.FREQUENCY_OPTIONS.keys()).index(config.DEFAULT_FREQUENCY)
)
run_interval_minutes = config.FREQUENCY_OPTIONS[frequency_label]

symbols_input = st.sidebar.text_input("Symbols (comma-separated)", ", ".join(config.DEFAULT_SYMBOLS))
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

st.sidebar.subheader("Risk limits (% of account equity)")
max_position_pct = st.sidebar.slider("Max position size", 1.0, 20.0, config.DEFAULT_RISK["max_position_pct"])
stop_loss_pct = st.sidebar.slider("Stop-loss", 0.5, 10.0, config.DEFAULT_RISK["stop_loss_pct"])
take_profit_pct = st.sidebar.slider("Take-profit", 0.5, 20.0, config.DEFAULT_RISK["take_profit_pct"])
max_daily_loss_pct = st.sidebar.slider("Max daily loss (halts bot)", 0.5, 10.0, config.DEFAULT_RISK["max_daily_loss_pct"])

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
    pos_df = pd.DataFrame([{
        "Symbol": p.symbol, "Qty": p.qty, "Avg Entry": p.avg_entry_price,
        "Current Price": p.current_price, "Unrealized P/L": p.unrealized_pl,
    } for p in positions])
    st.subheader("Open Positions")
    st.dataframe(pos_df, use_container_width=True)

# ---- Run cycle ----
def execute_cycle():
    with st.spinner(f"Running decision cycle with {provider_name}..."):
        results, eq, cash_ = run_once(symbols, provider_name, use_news, risk_cfg)
    st.session_state["last_results"] = results
    st.session_state["last_run_time"] = pd.Timestamp.utcnow()

if run_now:
    execute_cycle()

if auto_run:
    placeholder = st.empty()
    while True:
        execute_cycle()
        placeholder.info(f"Last run: {st.session_state.get('last_run_time')}. Next in {run_interval_minutes} min.")
        time.sleep(run_interval_minutes * 60)

# ---- Latest decisions ----
if "last_results" in st.session_state:
    st.subheader("Latest AI Decisions")
    st.dataframe(pd.DataFrame(st.session_state["last_results"]), use_container_width=True)

# ---- History from logs ----
st.subheader("Decision History")
try:
    hist = pd.read_csv(config.DECISIONS_LOG)
    st.dataframe(hist.tail(50), use_container_width=True)
except FileNotFoundError:
    st.info("No decisions logged yet - run a cycle to generate history.")

st.subheader("Trade History")
try:
    trades = pd.read_csv(config.TRADES_LOG)
    st.dataframe(trades, use_container_width=True)
    if not trades.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trades["timestamp"], y=trades["entry_price"],
                                 mode="markers+lines", name="Entry Price"))
        fig.update_layout(title="Trade Entries Over Time", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.info("No trades executed yet.")
