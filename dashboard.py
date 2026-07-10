"""
Streamlit control panel for the trading bot.
Run with: streamlit run dashboard.py
"""
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    """Convert any ISO timestamp or Timestamp to a readable local string."""
    try:
        if pd.isna(ts_value):
            return ""
    except Exception:
        pass
    try:
        dt = pd.to_datetime(ts_value, utc=True).to_pydatetime()
        # Convert UTC -> local by offsetting with the system timezone
        local_dt = dt.astimezone()
        return local_dt.strftime("%d/%m/%Y  %H:%M:%S")
    except Exception:
        return str(ts_value)


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

# ---- Symbol search & selection ----
st.sidebar.subheader("Symbols")

# Search box — type ticker or name to filter
search_query = st.sidebar.text_input(
    "Search symbol (stocks & crypto)",
    placeholder="e.g. AAPL, BTC/USD, ETH..."
).strip().upper()

# Start from the config defaults as the current selection
if "selected_symbols" not in st.session_state:
    st.session_state["selected_symbols"] = list(config.DEFAULT_SYMBOLS)

# Well-known list that users can search / filter inside the sidebar
KNOWN_SYMBOLS = [
    # Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
    "SPY", "QQQ", "DIA", "IWM", "ARKK",
    # Crypto (Alpaca format)
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
    "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "SHIB/USD",
]

if search_query:
    filtered = [s for s in KNOWN_SYMBOLS if search_query in s]
else:
    filtered = KNOWN_SYMBOLS

# Multiselect pre-seeded with session state
selected = st.sidebar.multiselect(
    "Select symbols to trade",
    options=filtered if filtered else KNOWN_SYMBOLS,
    default=[s for s in st.session_state["selected_symbols"] if s in (filtered if filtered else KNOWN_SYMBOLS)],
    key="symbol_multiselect"
)

# Allow typing a custom symbol not in the list
custom_symbol = st.sidebar.text_input(
    "Add custom symbol (not in list above)",
    placeholder="e.g. PLTR, MSTR, APE/USD"
).strip().upper()
if custom_symbol and custom_symbol not in selected:
    selected = selected + [custom_symbol]

st.session_state["selected_symbols"] = selected
symbols = selected if selected else list(config.DEFAULT_SYMBOLS)

st.sidebar.subheader("Risk limits (% of account equity)")
max_position_pct = st.sidebar.slider("Max position size", 1.0, 20.0, config.DEFAULT_RISK["max_position_pct"])
stop_loss_pct    = st.sidebar.slider("Stop-loss", 0.5, 10.0, config.DEFAULT_RISK["stop_loss_pct"])
take_profit_pct  = st.sidebar.slider("Take-profit", 0.5, 20.0, config.DEFAULT_RISK["take_profit_pct"])
max_daily_loss_pct = st.sidebar.slider("Max daily loss (halts bot)", 0.5, 10.0, config.DEFAULT_RISK["max_daily_loss_pct"])

risk_cfg = {
    "max_position_pct": max_position_pct,
    "stop_loss_pct": stop_loss_pct,
    "take_profit_pct": take_profit_pct,
    "max_daily_loss_pct": max_daily_loss_pct,
}

mode_label = "PAPER TRADING" if config.ALPACA_PAPER else "LIVE TRADING"
st.sidebar.markdown(f"**Mode:** `{mode_label}`")

run_now  = st.sidebar.button("Run one cycle now", type="primary")
auto_run = st.sidebar.toggle("Auto-run continuously")

# ---- Account overview ----
col1, col2, col3 = st.columns(3)
try:
    trading_client = get_trading_client()
    equity, cash    = get_account_equity(trading_client)
    positions       = get_open_positions(trading_client)
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
    st.dataframe(pos_df, width="stretch")


# ---- Run cycle with per-symbol Kronos progress ----
def execute_cycle():
    is_kronos = "kronos" in provider_name.lower()
    total = len(symbols)

    if is_kronos and total > 0:
        st.info(f"Running Kronos inference on {total} symbol(s)...")
        progress_bar = st.progress(0, text="Starting...")
        status_text  = st.empty()

        results = []
        from orchestrator import log_row
        from data_fetcher import get_market_snapshot
        from ai_decision import get_decision
        from risk_manager import calc_position_size, stop_loss_take_profit_prices
        from executor import submit_bracket_order
        from datetime import datetime
        import config as _cfg

        trading_client = get_trading_client()
        from risk_manager import get_account_equity as _gae
        equity, cash_ = _gae(trading_client)
        snapshot = get_market_snapshot(symbols)

        for i, (symbol, market_data) in enumerate(snapshot.items()):
            pct = int((i / total) * 100)
            progress_bar.progress(pct, text=f"Analyzing {symbol} ({i+1}/{total})...")
            status_text.markdown(f"**Kronos is working on:** `{symbol}`")

            decision = get_decision(symbol, market_data, provider_name, use_news)
            decision["timestamp"] = datetime.utcnow().isoformat()
            log_row(_cfg.DECISIONS_LOG, decision)

            if decision["action"] in ("BUY", "SELL") and decision.get("confidence", 0) >= 60:
                price = market_data["close"]
                qty   = calc_position_size(equity, price, risk_cfg["max_position_pct"])
                if qty > 0:
                    stop_price, target_price = stop_loss_take_profit_prices(
                        price, risk_cfg["stop_loss_pct"], risk_cfg["take_profit_pct"], decision["action"]
                    )
                    try:
                        order = submit_bracket_order(trading_client, symbol, qty, decision["action"], stop_price, target_price)
                        from orchestrator import log_row as _lr
                        _lr(_cfg.TRADES_LOG, {
                            "timestamp": datetime.utcnow().isoformat(),
                            "symbol": symbol, "side": decision["action"],
                            "qty": qty, "entry_price": price,
                            "stop_price": stop_price, "target_price": target_price,
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
            results, equity, cash_ = run_once(symbols, provider_name, use_news, risk_cfg)

    st.session_state["last_results"]  = results
    st.session_state["last_run_time"] = datetime.utcnow().isoformat()


if run_now:
    execute_cycle()

if auto_run:
    placeholder = st.empty()
    while True:
        execute_cycle()
        last = fmt_ts(st.session_state.get("last_run_time", ""))
        placeholder.info(f"Last run: {last}. Next in {run_interval_minutes} min.")
        time.sleep(run_interval_minutes * 60)


# ---- Latest decisions ----
if "last_results" in st.session_state:
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
            x=trades["timestamp"], y=trades["entry_price"],
            mode="markers+lines", name="Entry Price"
        ))
        fig.update_layout(title="Trade Entries Over Time", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, width="stretch")
except FileNotFoundError:
    st.info("No trades executed yet.")
