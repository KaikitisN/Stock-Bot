"""AIBots-inspired dark theme for the Streamlit dashboard."""

AIBOTS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #0a0b14;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='800' viewBox='0 0 800 800'%3E%3Cg fill='none' stroke='%231e293b' stroke-width='0.6' opacity='0.35'%3E%3Cpath d='M0 200 Q200 180 400 200 T800 200'/%3E%3Cpath d='M0 300 Q200 280 400 300 T800 300'/%3E%3Cpath d='M0 400 Q200 380 400 400 T800 400'/%3E%3Cpath d='M0 500 Q200 480 400 500 T800 500'/%3E%3Cpath d='M0 600 Q200 580 400 600 T800 600'/%3E%3Cpath d='M100 0 Q120 200 100 400 T100 800'/%3E%3Cpath d='M300 0 Q320 200 300 400 T300 800'/%3E%3Cpath d='M500 0 Q520 200 500 400 T500 800'/%3E%3Cpath d='M700 0 Q720 200 700 400 T700 800'/%3E%3C/g%3E%3C/svg%3E");
    background-size: auto, auto, 800px 800px;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Top navigation */
.aibots-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1.25rem 0;
    border-bottom: 1px solid rgba(59, 130, 246, 0.12);
    margin-bottom: 1.5rem;
}

.aibots-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}

.aibots-logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.aibots-nav-links {
    display: flex;
    gap: 0.35rem;
    align-items: center;
}

.aibots-nav-link {
    padding: 0.45rem 1rem;
    border-radius: 8px;
    color: #94a3b8;
    font-size: 0.875rem;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.15s;
}

.aibots-nav-link.active {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.aibots-nav-right {
    display: flex;
    align-items: center;
    gap: 1rem;
    color: #94a3b8;
    font-size: 0.85rem;
}

.aibots-badge {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.aibots-badge.paper {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
}

.aibots-badge.live {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
}

/* Page title */
.aibots-page-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 1.25rem;
}

/* Cards */
.aibots-card {
    background: linear-gradient(145deg, rgba(18, 21, 42, 0.95), rgba(15, 18, 35, 0.9));
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    height: 100%;
}

.aibots-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.aibots-card-subtitle {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 400;
}

/* Legend items */
.aibots-legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0;
    font-size: 0.82rem;
    color: #cbd5e1;
}

.aibots-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Bot list */
.aibots-bot-group {
    margin-bottom: 0.75rem;
}

.aibots-bot-group-header {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.aibots-bot-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.75rem;
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    margin-bottom: 0.35rem;
    font-size: 0.85rem;
}

.aibots-bot-name {
    color: #e2e8f0;
    font-weight: 500;
}

.aibots-bot-status {
    color: #4ade80;
    font-weight: 600;
    font-size: 0.8rem;
}

.aibots-bot-status.hold {
    color: #94a3b8;
}

.aibots-bot-status.sell {
    color: #f87171;
}

/* Position table */
.aibots-pos-header {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 0.8fr;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    font-size: 0.7rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid rgba(59, 130, 246, 0.1);
}

.aibots-pos-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 0.8fr;
    gap: 0.5rem;
    padding: 0.6rem 0.75rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    border-bottom: 1px solid rgba(30, 41, 59, 0.5);
    align-items: center;
}

.aibots-pos-row:hover {
    background: rgba(59, 130, 246, 0.05);
}

.aibots-pos-symbol {
    font-weight: 600;
    color: #f1f5f9;
}

.aibots-pos-pl-positive {
    color: #4ade80;
}

.aibots-pos-pl-negative {
    color: #f87171;
}

/* Mover rows */
.aibots-mover-row {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 0.8fr;
    gap: 0.5rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(30, 41, 59, 0.5);
    align-items: center;
    font-size: 0.85rem;
    color: #cbd5e1;
}

.aibots-mover-symbol {
    font-weight: 600;
    color: #f1f5f9;
}

.aibots-mover-change-pos {
    color: #4ade80;
    font-weight: 600;
}

.aibots-mover-change-neg {
    color: #f87171;
    font-weight: 600;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: rgba(18, 21, 42, 0.6);
    border: 1px solid rgba(59, 130, 246, 0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button[kind="secondary"] {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    color: #94a3b8;
    border: 1px solid transparent;
    padding: 0.4rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #60a5fa !important;
    border-color: rgba(59, 130, 246, 0.35) !important;
}

.stExpander {
    background: rgba(18, 21, 42, 0.6);
    border: 1px solid rgba(59, 130, 246, 0.1);
    border-radius: 10px;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border-color: rgba(59, 130, 246, 0.1);
}

/* Hide streamlit nav button styling for our custom nav */
div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100%;
}

.aibots-show-more {
    color: #60a5fa;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
    padding: 0.5rem;
    cursor: pointer;
    letter-spacing: 0.04em;
}

.aibots-center-stat {
    text-align: center;
    padding: 1rem 0;
}

.aibots-center-stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
}

.aibots-center-stat-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}

.aibots-exchange-card {
    background: rgba(30, 41, 59, 0.35);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.aibots-exchange-name {
    font-weight: 600;
    color: #e2e8f0;
    font-size: 1rem;
}

.aibots-exchange-status {
    color: #4ade80;
    font-size: 0.85rem;
}
</style>
"""

SYMBOL_COLORS = {
    "AAPL": "#f97316",
    "MSFT": "#06b6d4",
    "NVDA": "#22c55e",
    "TSLA": "#a855f7",
    "GOOGL": "#eab308",
    "AMZN": "#ec4899",
    "META": "#3b82f6",
    "CASH": "#8b5cf6",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=10, b=10),
)


def inject_theme():
    import streamlit as st
    st.markdown(AIBOTS_CSS, unsafe_allow_html=True)


def symbol_color(symbol: str) -> str:
    return SYMBOL_COLORS.get(symbol.upper(), "#3b82f6")
