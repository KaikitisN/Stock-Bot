"""
AIBots-inspired Streamlit control panel for the AI trading bot.
Preserves all trading features (Kronos progress, liquidate, auto-run, crypto).
Run with: streamlit run dashboard.py
"""
import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import config
from dashboard_helpers import (
    build_allocation_ring,
    build_donut_chart,
    build_pl_bar_chart,
    build_sparkline,
    compute_pl_periods,
    group_bots_by_provider,
    latest_decisions_by_symbol,
    portfolio_allocation,
    positions_table_data,
)
from dashboard_theme import inject_theme, symbol_color
from data_fetcher import get_market_snapshot, get_mover_stats
from executor import (
    get_open_positions,
    get_portfolio_history,
    get_trading_client,
    liquidate_position,
)
from orchestrator import is_stock_market_open, process_symbol, run_once
from risk_manager import get_account_equity, get_account_summary, is_trading_halted

st.set_page_config(
    page_title="AIBots — AI Trading",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()
st_autorefresh(interval=60000, key="dashboard_autorefresh")

PAGES = ["Dashboard", "AI Trading Bots", "My Exchange", "Settings"]

KNOWN_SYMBOLS = list(dict.fromkeys(config.DEFAULT_SYMBOLS + [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
    "SPY", "QQQ", "DIA", "IWM", "ARKK",
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
    "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "SHIB/USD",
]))


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


def init_session_state():
    defaults = {
        "page": "Dashboard",
        "provider_name": config.DEFAULT_AI_PROVIDER,
        "use_news": config.USE_NEWS_DEFAULT,
        "frequency_label": config.DEFAULT_FREQUENCY,
        "selected_symbols": list(config.DEFAULT_SYMBOLS),
        "max_position_pct": config.DEFAULT_RISK["max_position_pct"],
        "stop_loss_pct": config.DEFAULT_RISK["stop_loss_pct"],
        "take_profit_pct": config.DEFAULT_RISK["take_profit_pct"],
        "max_daily_loss_pct": config.DEFAULT_RISK["max_daily_loss_pct"],
        "show_all_bots": False,
        "auto_running": False,
        "next_run_time": None,
        "cycle_running": False,
        "last_results": [],
        "last_run_time": None,
        "cycle_just_finished": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_risk_cfg() -> dict:
    return {
        "max_position_pct": st.session_state.max_position_pct,
        "stop_loss_pct": st.session_state.stop_loss_pct,
        "take_profit_pct": st.session_state.take_profit_pct,
        "max_daily_loss_pct": st.session_state.max_daily_loss_pct,
    }


def schedule_next_run(run_interval_minutes: int):
    st.session_state["next_run_time"] = (
        pd.Timestamp.utcnow() + pd.Timedelta(minutes=run_interval_minutes)
    )


def fetch_account_data():
    try:
        client = get_trading_client()
        summary = get_account_summary(client)
        equity = summary["equity"]
        cash = summary["cash"]
        last_equity = summary["last_equity"]
        positions = get_open_positions(client)
        try:
            history = get_portfolio_history(client, period="all")
        except Exception:
            try:
                history = get_portfolio_history(client, period="1A")
            except Exception:
                history = None
        return client, equity, cash, last_equity, positions, history, None
    except Exception as e:
        return None, 0.0, 0.0, 0.0, [], None, str(e)


def render_nav():
    mode = "paper" if config.ALPACA_PAPER else "live"
    mode_label = "PAPER" if config.ALPACA_PAPER else "LIVE"
    cols = st.columns([2, 4, 2])
    with cols[0]:
        st.markdown(
            '<div class="aibots-logo">'
            '<div class="aibots-logo-icon">&#129302;</div>AIBots</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        nav_cols = st.columns(len(PAGES))
        for i, page in enumerate(PAGES):
            with nav_cols[i]:
                is_active = st.session_state.page == page
                if st.button(
                    page,
                    key=f"nav_{page}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.page = page
                    st.rerun()
    with cols[2]:
        st.markdown(
            f'<div class="aibots-nav-right">'
            f'<span class="aibots-badge {mode}">{mode_label}</span>'
            f'<span>Support</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def execute_cycle():
    provider_name = st.session_state.provider_name
    use_news = st.session_state.use_news
    symbols = st.session_state.selected_symbols or list(config.DEFAULT_SYMBOLS)
    risk_cfg = get_risk_cfg()
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
            status_text.markdown("**Kronos finished all symbols**")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    else:
        with st.spinner(f"Running decision cycle with {provider_name}..."):
            results, _, _ = run_once(symbols, provider_name, use_news, risk_cfg)

    st.session_state["last_results"] = results
    st.session_state["last_run_time"] = datetime.utcnow().isoformat()
    st.session_state["cycle_just_finished"] = True


def render_dashboard_page(equity, cash, last_equity, positions, history, error):
    st.markdown('<div class="aibots-page-title">My Dashboard</div>', unsafe_allow_html=True)
    if error:
        st.error(f"Could not connect to Alpaca. Check your API keys in .env. ({error})")

    segments, total = portfolio_allocation(positions, equity, cash)
    pl_data = compute_pl_periods(positions, equity, last_equity, history)
    try:
        decisions_df = read_csv_with_fallback(config.DECISIONS_LOG)
    except FileNotFoundError:
        decisions_df = pd.DataFrame()
    latest = latest_decisions_by_symbol(decisions_df)
    mover_stats = {}
    try:
        mover_stats = get_mover_stats(st.session_state.selected_symbols or config.DEFAULT_SYMBOLS)
    except Exception:
        pass

    row1 = st.columns([1.1, 1.2, 1])
    with row1[0]:
        st.markdown('<div class="aibots-card"><div class="aibots-card-title">Overview</div>', unsafe_allow_html=True)
        if segments:
            st.plotly_chart(
                build_donut_chart(segments, total or equity),
                width="stretch",
                config={"displayModeBar": False},
            )
            legend_html = ""
            for s in segments:
                display_val = s.get("display_value", s["value"])
                sign = "+" if display_val >= 0 else "-"
                legend_html += (
                    f'<div class="aibots-legend-item">'
                    f'<span class="aibots-dot" style="background:{s["color"]}"></span>'
                    f'<span><b>{s["label"]}</b> &nbsp; {s["qty"]:,.4f} &nbsp; '
                    f'<span style="color:#64748b">({sign}$ {abs(display_val):,.2f})</span></span></div>'
                )
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="aibots-center-stat">'
                f'<div class="aibots-center-stat-value">$ {equity:,.2f}</div>'
                f'<div class="aibots-center-stat-label">Total Balance</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with row1[1]:
        st.markdown('<div class="aibots-card"><div class="aibots-card-title">Profit and Loss</div>', unsafe_allow_html=True)
        st.radio("P/L period", ["Days", "Weeks", "Months"], horizontal=True, label_visibility="collapsed")
        st.plotly_chart(build_pl_bar_chart(pl_data), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with row1[2]:
        st.markdown('<div class="aibots-card"><div class="aibots-card-title">Active Bots</div>', unsafe_allow_html=True)
        bot_groups = group_bots_by_provider(
            latest,
            st.session_state.selected_symbols or config.DEFAULT_SYMBOLS,
            st.session_state.provider_name,
        )
        show_limit = None if st.session_state.show_all_bots else 2
        for group in (bot_groups[:show_limit] if show_limit else bot_groups):
            st.markdown(
                f'<div class="aibots-bot-group">'
                f'<div class="aibots-bot-group-header">{group["provider"]} ({group["count"]})</div>',
                unsafe_allow_html=True,
            )
            for bot in group["bots"]:
                status_class = bot["action"].lower()
                st.markdown(
                    f'<div class="aibots-bot-row">'
                    f'<span class="aibots-bot-name">{bot["symbol"]} bot</span>'
                    f'<span class="aibots-bot-status {status_class}">{bot["status_label"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if len(bot_groups) > 2 and not st.session_state.show_all_bots:
            if st.button("SHOW MORE", key="show_more_bots"):
                st.session_state.show_all_bots = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns([1.6, 1])
    with row2[0]:
        st.markdown(
            '<div class="aibots-card"><div class="aibots-card-title">'
            'My Stocks on Exchanges</div>',
            unsafe_allow_html=True,
        )
        pos_cols = st.columns([1, 1.4])
        with pos_cols[0]:
            st.plotly_chart(
                build_allocation_ring(segments),
                width="stretch",
                config={"displayModeBar": False},
            )
        with pos_cols[1]:
            pos_rows = positions_table_data(positions, equity)
            st.markdown(
                '<div class="aibots-pos-header">'
                '<span>STOCK / EXCHANGE</span><span>AMOUNT</span>'
                '<span>VALUE</span><span>ALLOCATION</span></div>',
                unsafe_allow_html=True,
            )
            if pos_rows:
                for row in pos_rows:
                    pl_class = "aibots-pos-pl-positive" if row["pl"] >= 0 else "aibots-pos-pl-negative"
                    with st.expander(f"{row['symbol']} — ${row['value']:,.2f} ({row['allocation']:.0f}%)"):
                        st.markdown(
                            f'<div class="aibots-pos-row">'
                            f'<span><span class="aibots-pos-symbol">{row["symbol"]}</span>'
                            f'<br><span style="font-size:0.75rem;color:#64748b">{row["exchange"]}</span></span>'
                            f'<span>{row["qty"]:,.2f}</span>'
                            f'<span>${row["value"]:,.2f}<br><span class="{pl_class}">{row["pl"]:+,.2f}</span></span>'
                            f'<span>{row["allocation"]:.0f}%</span></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No open positions — run a trading cycle to get started.")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2[1]:
        st.markdown(
            '<div class="aibots-card"><div class="aibots-card-title">'
            'The Best Stocks<span class="aibots-card-subtitle">Watchlist</span></div>',
            unsafe_allow_html=True,
        )
        mover_tab = st.radio(
            "Mover filter", ["Trending", "Biggest Gainers"],
            horizontal=True, label_visibility="collapsed",
        )
        if mover_stats:
            sorted_movers = sorted(
                mover_stats.items(),
                key=lambda x: x[1]["change_24h_pct"],
                reverse=(mover_tab == "Biggest Gainers"),
            )
            for sym, stats in sorted_movers[:6]:
                chg = stats["change_24h_pct"]
                chg_class = "aibots-mover-change-pos" if chg >= 0 else "aibots-mover-change-neg"
                mcols = st.columns([1.2, 1, 0.7])
                with mcols[0]:
                    st.markdown(
                        f'<div class="aibots-mover-symbol">{sym}</div>'
                        f'<div style="font-size:0.75rem;color:#64748b">Vol {stats["volume"]:,.0f}</div>',
                        unsafe_allow_html=True,
                    )
                with mcols[1]:
                    st.plotly_chart(
                        build_sparkline(stats["sparkline"], symbol_color(sym)),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                with mcols[2]:
                    st.markdown(
                        f'<div style="text-align:right">'
                        f'<div>${stats["price"]:,.2f}</div>'
                        f'<div class="{chg_class}">{chg:+.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Market data unavailable — check Alpaca API keys.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_trading_bots_page(equity, cash, positions, error, trading_client):
    st.markdown('<div class="aibots-page-title">AI Trading Bots</div>', unsafe_allow_html=True)
    if error:
        st.error(f"Could not connect to Alpaca. ({error})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Account Equity", f"${equity:,.2f}")
    m2.metric("Cash", f"${cash:,.2f}")
    m3.metric("Open Positions", len(positions))
    m4.metric("AI Provider", st.session_state.provider_name)

    if trading_client is not None:
        halted, halt_reason = is_trading_halted(
            trading_client, st.session_state.max_daily_loss_pct
        )
        if halted:
            st.error(f"Trading halted: {halt_reason}")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        run_now = st.button("Run one cycle now", type="primary", width="stretch")
    with ctrl2:
        auto_run = st.toggle("Auto-run continuously", value=st.session_state.auto_running)
    with ctrl3:
        last = fmt_ts(st.session_state.get("last_run_time", ""))
        if last:
            st.info(f"Last run: {last}")

    run_interval_minutes = config.FREQUENCY_OPTIONS[st.session_state.frequency_label]

    if run_now and not st.session_state["cycle_running"]:
        st.session_state["cycle_running"] = True
        try:
            execute_cycle()
            if st.session_state["auto_running"]:
                schedule_next_run(run_interval_minutes)
        finally:
            st.session_state["cycle_running"] = False

    if auto_run and not st.session_state["auto_running"]:
        st.session_state["auto_running"] = True
        if st.session_state["next_run_time"] is None:
            st.session_state["next_run_time"] = pd.Timestamp.utcnow()
    elif not auto_run and st.session_state["auto_running"]:
        st.session_state["auto_running"] = False
        st.session_state["next_run_time"] = None

    status_placeholder = st.empty()
    if st.session_state["auto_running"]:
        now = pd.Timestamp.utcnow()
        next_run = st.session_state["next_run_time"]
        if next_run is None:
            next_run = now
            st.session_state["next_run_time"] = next_run

        if st.session_state["cycle_running"]:
            last = fmt_ts(st.session_state.get("last_run_time", ""))
            status_placeholder.info(
                f"Auto-run enabled. Cycle running. Last run: {last if last else 'Not yet'}."
            )
        else:
            remaining = next_run - now
            secs = max(int(remaining.total_seconds()), 0)
            mins, sec_rem = secs // 60, secs % 60
            last = fmt_ts(st.session_state.get("last_run_time", ""))
            status_placeholder.info(
                f"Last run: {last if last else 'Not yet'}. "
                f"Next auto-run in {mins:02d}:{sec_rem:02d}."
            )
            if now >= next_run:
                st.session_state["cycle_running"] = True
                try:
                    execute_cycle()
                    schedule_next_run(run_interval_minutes)
                finally:
                    st.session_state["cycle_running"] = False

        if not st.session_state["cycle_just_finished"]:
            time.sleep(1)
            st.rerun()
        else:
            st.session_state["cycle_just_finished"] = False
            time.sleep(2)
            st.rerun()

    # Background runner status
    st.subheader("Background Runner Status")
    status_path = f"{config.LOG_DIR}/runner_status.json"
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            runner_status = json.load(f)
        state = runner_status.get("state", "unknown")
        current_symbol = runner_status.get("current_symbol")
        current_index = runner_status.get("current_index", 0)
        total_symbols = runner_status.get("total_symbols", 0)
        if state == "running" and current_symbol:
            st.info(f"Processing **{current_symbol}** ({current_index}/{total_symbols})")
            if total_symbols > 0:
                st.progress(current_index / total_symbols)
        elif state == "idle":
            finished_at = runner_status.get("cycle_finished_at", "")
            st.success(f"Idle — last cycle finished at {finished_at}")
        elif state == "error":
            st.error(f"Runner error: {runner_status.get('error')}")
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

    # Open positions + liquidate
    if positions:
        st.subheader("Open Positions")
        pos_df = pd.DataFrame([{
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Avg Entry": p.avg_entry_price,
            "Current Price": p.current_price,
            "Market Value": p.market_value,
            "Unrealized P/L": p.unrealized_pl,
        } for p in positions])
        st.dataframe(pos_df, width="stretch")

        st.markdown("**Manually liquidate a position:**")
        liq_cols = st.columns(min(len(positions), 6))
        for idx, pos in enumerate(positions):
            col = liq_cols[idx % 6]
            if col.button(f"Sell all {pos.symbol}", key=f"liq_{pos.symbol}"):
                try:
                    order = liquidate_position(trading_client, pos.symbol)
                    if order:
                        st.success(f"Liquidation order submitted for {pos.symbol} (id: {order.id})")
                    else:
                        st.warning(f"No open position found for {pos.symbol}.")
                except Exception as e:
                    st.error(f"Failed to liquidate {pos.symbol}: {e}")

    if st.session_state.get("last_results"):
        st.subheader("Latest AI Decisions")
        df_results = pd.DataFrame(st.session_state["last_results"])
        if "timestamp" in df_results.columns:
            df_results["timestamp"] = df_results["timestamp"].apply(fmt_ts)
        st.dataframe(df_results, width="stretch")

    st.subheader("Decision History")
    try:
        hist = read_csv_with_fallback(config.DECISIONS_LOG)
        if "timestamp" in hist.columns:
            hist["timestamp"] = hist["timestamp"].apply(fmt_ts)
        st.dataframe(hist.tail(50), width="stretch")
    except FileNotFoundError:
        st.info("No decisions logged yet — run a cycle to generate history.")

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
                line=dict(color="#3b82f6"),
            ))
            fig.update_layout(
                title="Trade Entries Over Time",
                xaxis_title="Time",
                yaxis_title="Price",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(18,21,42,0.5)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig, width="stretch")
    except FileNotFoundError:
        st.info("No trades executed yet.")


def render_exchange_page(equity, cash, positions, error):
    st.markdown('<div class="aibots-page-title">My Exchange</div>', unsafe_allow_html=True)
    if error:
        st.error(f"Connection failed: {error}")
        st.markdown(
            '<div class="aibots-exchange-card">'
            '<div class="aibots-exchange-name">Alpaca Markets</div>'
            '<div class="aibots-exchange-status" style="color:#f87171">Disconnected</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        mode = "Paper Trading" if config.ALPACA_PAPER else "Live Trading"
        st.markdown(
            f'<div class="aibots-exchange-card">'
            f'<div class="aibots-exchange-name">Alpaca Markets</div>'
            f'<div class="aibots-exchange-status">Connected — {mode}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Equity", f"${equity:,.2f}")
        c2.metric("Available Cash", f"${cash:,.2f}")
        c3.metric("Active Positions", len(positions))
        if positions:
            st.subheader("Holdings on Alpaca")
            pos_df = pd.DataFrame([{
                "Symbol": p.symbol,
                "Quantity": p.qty,
                "Market Value": f"${float(p.market_value):,.2f}",
                "Unrealized P/L": f"${float(p.unrealized_pl):+,.2f}",
            } for p in positions])
            st.dataframe(pos_df, width="stretch")


def render_settings_page():
    st.markdown('<div class="aibots-page-title">Settings</div>', unsafe_allow_html=True)

    st.subheader("Bot Configuration")
    provider_keys = list(config.AI_PROVIDERS.keys())
    idx = provider_keys.index(st.session_state.provider_name) if st.session_state.provider_name in provider_keys else 0
    st.session_state.provider_name = st.selectbox("AI model / API key", provider_keys, index=idx)
    st.session_state.use_news = st.toggle(
        "Factor in news (Perplexity Sonar)", value=st.session_state.use_news
    )

    freq_keys = list(config.FREQUENCY_OPTIONS.keys())
    freq_idx = (
        freq_keys.index(st.session_state.frequency_label)
        if st.session_state.frequency_label in freq_keys else 0
    )
    st.session_state.frequency_label = st.selectbox("Run frequency", freq_keys, index=freq_idx)

    st.subheader("Symbols")
    search_query = st.text_input(
        "Search symbol (stocks & crypto)",
        placeholder="e.g. AAPL, BTC/USD, ETH...",
    ).strip().upper()
    filtered = [s for s in KNOWN_SYMBOLS if search_query in s] if search_query else KNOWN_SYMBOLS
    selected = st.multiselect(
        "Select symbols to trade",
        options=filtered if filtered else KNOWN_SYMBOLS,
        default=[
            s for s in st.session_state["selected_symbols"]
            if s in (filtered if filtered else KNOWN_SYMBOLS)
        ],
        key="symbol_multiselect",
    )
    custom_symbol = st.text_input(
        "Add custom symbol (not in list above)",
        placeholder="e.g. PLTR, MSTR, APE/USD",
    ).strip().upper()
    if custom_symbol and custom_symbol not in selected:
        selected = selected + [custom_symbol]
    st.session_state["selected_symbols"] = selected if selected else list(config.DEFAULT_SYMBOLS)

    st.subheader("Risk limits (% of account equity)")
    st.session_state.max_position_pct = st.slider(
        "Max position size", 1.0, 20.0, st.session_state.max_position_pct,
    )
    st.session_state.stop_loss_pct = st.slider(
        "Stop-loss", 0.5, 10.0, st.session_state.stop_loss_pct,
    )
    st.session_state.take_profit_pct = st.slider(
        "Take-profit", 0.5, 20.0, st.session_state.take_profit_pct,
    )
    st.session_state.max_daily_loss_pct = st.slider(
        "Max daily loss (halts bot)", 0.5, 10.0, st.session_state.max_daily_loss_pct,
    )
    st.caption(
        f"Min confidence: **{config.MIN_TRADE_CONFIDENCE}** | "
        f"Max positions: **{config.MAX_OPEN_POSITIONS}** | "
        f"Trend filter: **{'on' if config.USE_TREND_FILTER else 'off'}**"
    )
    mode_label = "PAPER TRADING" if config.ALPACA_PAPER else "LIVE TRADING"
    st.markdown(f"**Trading Mode:** `{mode_label}`")
    st.caption("Change ALPACA_PAPER in .env to switch between paper and live trading.")


def main():
    init_session_state()
    render_nav()

    client, equity, cash, last_equity, positions, history, error = fetch_account_data()
    page = st.session_state.page

    if page == "Dashboard":
        render_dashboard_page(equity, cash, last_equity, positions, history, error)
    elif page == "AI Trading Bots":
        render_trading_bots_page(equity, cash, positions, error, client)
    elif page == "My Exchange":
        render_exchange_page(equity, cash, positions, error)
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
