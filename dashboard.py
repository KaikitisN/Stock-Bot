"""
Streamlit control panel for the trading bot.
Run with: streamlit run dashboard.py

UI: Dark trading dashboard inspired by professional crypto/stock bot designs.
Color palette: #040405 bg, #1a1b2e cards, #1E40AD accent blue, #10b981 green, #ef4444 red
"""
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
from orchestrator import run_once
from executor import get_trading_client, get_open_positions
from risk_manager import get_account_equity

st.set_page_config(
    page_title="AI Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI Trading Bot — Alpaca + Kronos / GPT / Claude"},
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Base ---- */
[data-testid="stAppViewContainer"] {
    background: #040405;
    color: #b8bcd1;
}
[data-testid="stSidebar"] {
    background: #0d0e1a;
    border-right: 1px solid #1e2235;
}
[data-testid="stSidebar"] * { color: #b8bcd1 !important; }

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: #12132a;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 20px 24px;
}
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .06em; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #1E40AD 0%, #3b5fd6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    width: 100%;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ---- Tables / dataframes ---- */
[data-testid="stDataFrame"] {
    background: #12132a !important;
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.6rem 0 .6rem 0;
    padding-bottom: .4rem;
    border-bottom: 1px solid #1e2235;
}

/* ---- Status badge ---- */
.badge-paper  { background:#1e3a5f; color:#60a5fa; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }
.badge-live   { background:#3b1f1f; color:#f87171; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }

/* ---- Signal chips ---- */
.chip-buy  { background:#064e3b; color:#34d399; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.82rem;}
.chip-sell { background:#450a0a; color:#f87171; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.82rem;}
.chip-hold { background:#1c1f2e; color:#94a3b8; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.82rem;}

/* ---- Progress bar ---- */
[data-testid="stProgress"] > div > div { background: #1E40AD !important; }

/* ---- Sidebar inputs ---- */
.stTextInput input, .stSelectbox select {
    background: #1a1b2e !important;
    border: 1px solid #1e2235 !important;
    color: #fff !important;
    border-radius: 8px !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0e1a; }
::-webkit-scrollbar-thumb { background: #1e2235; border-radius: 3px; }

/* ---- Divider ---- */
hr { border-color: #1e2235 !important; }

/* ---- Page title ---- */
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0;
}
.title-accent { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
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
        dt = pd.to_datetime(ts_value, utc=True).to_pydatetime().astimezone()
        return dt.strftime("%d/%m/%Y  %H:%M:%S")
    except Exception:
        return str(ts_value)


def action_chip(action: str) -> str:
    a = str(action).upper()
    css = {"BUY": "chip-buy", "SELL": "chip-sell"}.get(a, "chip-hold")
    return f'<span class="{css}">{a}</span>'


def section(title: str):
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:16px 0 8px'>
        <span style='font-size:1.4rem;font-weight:800;color:#fff;'>⚡ AI <span style='color:#3b82f6'>Trade</span>Bot</span>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("**🤖 AI Engine**")
    provider_name = st.selectbox(
        "Model",
        list(config.AI_PROVIDERS.keys()),
        index=list(config.AI_PROVIDERS.keys()).index(config.DEFAULT_AI_PROVIDER),
        label_visibility="collapsed",
    )

    use_news = st.toggle("📰 Factor in news (Perplexity Sonar)", value=config.USE_NEWS_DEFAULT)
    st.divider()

    st.markdown("**⏱ Run Frequency**")
    frequency_label = st.selectbox(
        "Frequency",
        list(config.FREQUENCY_OPTIONS.keys()),
        index=list(config.FREQUENCY_OPTIONS.keys()).index(config.DEFAULT_FREQUENCY),
        label_visibility="collapsed",
    )
    run_interval_minutes = config.FREQUENCY_OPTIONS[frequency_label]
    st.divider()

    st.markdown("**🔍 Symbol Selection**")
    search_query = st.text_input(
        "Search", placeholder="BTC/USD, AAPL, ETH..."
    ).strip().upper()

    if "selected_symbols" not in st.session_state:
        st.session_state["selected_symbols"] = list(config.DEFAULT_SYMBOLS)

    KNOWN_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
        "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
        "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
        "SPY", "QQQ", "DIA", "IWM", "ARKK",
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
        "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
        "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "SHIB/USD",
    ]
    filtered = [s for s in KNOWN_SYMBOLS if search_query in s] if search_query else KNOWN_SYMBOLS

    selected = st.multiselect(
        "Symbols",
        options=filtered if filtered else KNOWN_SYMBOLS,
        default=[s for s in st.session_state["selected_symbols"] if s in (filtered if filtered else KNOWN_SYMBOLS)],
        key="symbol_multiselect",
        label_visibility="collapsed",
    )
    custom_symbol = st.text_input(
        "Custom symbol", placeholder="e.g. PLTR, MSTR, APE/USD"
    ).strip().upper()
    if custom_symbol and custom_symbol not in selected:
        selected = selected + [custom_symbol]

    st.session_state["selected_symbols"] = selected
    symbols = selected if selected else list(config.DEFAULT_SYMBOLS)
    st.divider()

    st.markdown("**⚖️ Risk Limits**")
    max_position_pct   = st.slider("Max position size %",  1.0, 20.0, config.DEFAULT_RISK["max_position_pct"])
    stop_loss_pct      = st.slider("Stop-loss %",          0.5, 10.0, config.DEFAULT_RISK["stop_loss_pct"])
    take_profit_pct    = st.slider("Take-profit %",        0.5, 20.0, config.DEFAULT_RISK["take_profit_pct"])
    max_daily_loss_pct = st.slider("Max daily loss %",     0.5, 10.0, config.DEFAULT_RISK["max_daily_loss_pct"])
    st.divider()

    mode_label = "PAPER TRADING" if config.ALPACA_PAPER else "LIVE TRADING"
    badge_cls  = "badge-paper" if config.ALPACA_PAPER else "badge-live"
    st.markdown(f'<span class="{badge_cls}">🔵 {mode_label}</span>', unsafe_allow_html=True)
    st.markdown("")

    run_now  = st.button("▶ Run One Cycle Now", type="primary")
    auto_run = st.toggle("🔁 Auto-run continuously")

risk_cfg = {
    "max_position_pct": max_position_pct,
    "stop_loss_pct": stop_loss_pct,
    "take_profit_pct": take_profit_pct,
    "max_daily_loss_pct": max_daily_loss_pct,
}


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>
    ⚡ AI <span class='title-accent'>&nbsp;TradeBot&nbsp;</span> Dashboard
</div>
""", unsafe_allow_html=True)
st.markdown("<p style='color:#6b7280;font-size:.85rem;margin-top:2px;margin-bottom:1rem;'>Automated stock & crypto trading powered by Kronos / GPT / Claude</p>", unsafe_allow_html=True)


# ── Account overview ─────────────────────────────────────────────────────────
try:
    trading_client = get_trading_client()
    equity, cash   = get_account_equity(trading_client)
    positions      = get_open_positions(trading_client)
except Exception as e:
    st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({e})")
    positions = []
    equity, cash = 0.0, 0.0

pnl_color = "#10b981" if equity >= 100_000 else "#ef4444"

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Account Equity",  f"${equity:,.2f}")
col2.metric("💵 Available Cash",  f"${cash:,.2f}")
col3.metric("📂 Open Positions",  len(positions))
col4.metric("🤖 AI Provider",     provider_name)


# ── Open positions ───────────────────────────────────────────────────────────
if positions:
    section("📊 Open Positions")
    pos_rows = []
    for p in positions:
        pl = float(p.unrealized_pl)
        pl_str = f"+${pl:,.2f}" if pl >= 0 else f"-${abs(pl):,.2f}"
        pl_color = "#10b981" if pl >= 0 else "#ef4444"
        pos_rows.append({
            "Symbol":        p.symbol,
            "Qty":           p.qty,
            "Avg Entry":     f"${float(p.avg_entry_price):,.2f}",
            "Current Price": f"${float(p.current_price):,.2f}",
            "Unrealized P/L": pl_str,
        })
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)


# ── Execute cycle ─────────────────────────────────────────────────────────────
def execute_cycle():
    is_kronos = "kronos" in provider_name.lower()
    total = len(symbols)

    if is_kronos and total > 0:
        st.info(f"⚡ Running Kronos inference on {total} symbol(s)...")
        progress_bar = st.progress(0, text="Starting Kronos...")
        status_text  = st.empty()

        results = []
        from orchestrator import log_row
        from data_fetcher import get_market_snapshot
        from ai_decision import get_decision
        from risk_manager import calc_position_size, stop_loss_take_profit_prices
        from executor import submit_bracket_order
        import config as _cfg

        tc = get_trading_client()
        from risk_manager import get_account_equity as _gae
        eq, _ = _gae(tc)
        snapshot = get_market_snapshot(symbols)

        for i, (symbol, market_data) in enumerate(snapshot.items()):
            pct = int((i / total) * 100)
            progress_bar.progress(pct, text=f"🔍 Analyzing {symbol}  ({i+1} / {total})")
            status_text.markdown(
                f"<p style='color:#60a5fa;font-size:.9rem;'>⚡ Kronos processing: <b>{symbol}</b></p>",
                unsafe_allow_html=True,
            )

            decision = get_decision(symbol, market_data, provider_name, use_news)
            decision["timestamp"] = datetime.utcnow().isoformat()
            log_row(_cfg.DECISIONS_LOG, decision)

            if decision["action"] in ("BUY", "SELL") and decision.get("confidence", 0) >= 60:
                price = market_data["close"]
                qty   = calc_position_size(eq, price, risk_cfg["max_position_pct"])
                if qty > 0:
                    stop_price, target_price = stop_loss_take_profit_prices(
                        price, risk_cfg["stop_loss_pct"], risk_cfg["take_profit_pct"], decision["action"]
                    )
                    try:
                        order = submit_bracket_order(tc, symbol, qty, decision["action"], stop_price, target_price)
                        log_row(_cfg.TRADES_LOG, {
                            "timestamp": datetime.utcnow().isoformat(),
                            "symbol": symbol, "side": decision["action"],
                            "qty": qty, "entry_price": price,
                            "stop_price": stop_price, "target_price": target_price,
                            "order_id": str(order.id),
                        })
                        decision["trade_submitted"] = True
                    except Exception as exc:
                        decision["trade_submitted"] = False
                        decision["error"] = str(exc)
            results.append(decision)

        progress_bar.progress(100, text="✅ Kronos complete!")
        status_text.markdown(
            "<p style='color:#34d399;font-size:.9rem;font-weight:600;'>✅ Kronos finished all symbols</p>",
            unsafe_allow_html=True,
        )
        time.sleep(1.2)
        progress_bar.empty()
        status_text.empty()

    else:
        with st.spinner(f"Running cycle with {provider_name}..."):
            results, _, __ = run_once(symbols, provider_name, use_news, risk_cfg)

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


# ── Latest AI decisions ───────────────────────────────────────────────────────
if "last_results" in st.session_state:
    section("🧠 Latest AI Decisions")
    df_res = pd.DataFrame(st.session_state["last_results"])
    if "timestamp" in df_res.columns:
        df_res["timestamp"] = df_res["timestamp"].apply(fmt_ts)
    st.dataframe(df_res, use_container_width=True, hide_index=True)


# ── Decision history ──────────────────────────────────────────────────────────
section("📋 Decision History")
try:
    hist = read_csv_with_fallback(config.DECISIONS_LOG)
    if "timestamp" in hist.columns:
        hist["timestamp"] = hist["timestamp"].apply(fmt_ts)
    st.dataframe(hist.tail(50), use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.info("No decisions logged yet — run a cycle to generate history.")


# ── Trade history + chart ─────────────────────────────────────────────────────
section("💹 Trade History")
try:
    trades = read_csv_with_fallback(config.TRADES_LOG)
    if "timestamp" in trades.columns:
        trades["timestamp"] = trades["timestamp"].apply(fmt_ts)
    st.dataframe(trades, use_container_width=True, hide_index=True)

    if not trades.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades["timestamp"],
            y=trades["entry_price"],
            mode="markers+lines",
            name="Entry Price",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color="#3b82f6", size=7),
        ))
        fig.update_layout(
            title=dict(text="Trade Entries Over Time", font=dict(color="#ffffff", size=14)),
            paper_bgcolor="#12132a",
            plot_bgcolor="#12132a",
            font=dict(color="#b8bcd1"),
            xaxis=dict(gridcolor="#1e2235", title="Time"),
            yaxis=dict(gridcolor="#1e2235", title="Price"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.info("No trades executed yet.")
