"""Command-center dark theme (amber accents) for the Streamlit dashboard."""

COMMAND_CENTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --cc-bg: #0c0d10;
    --cc-card: #14151a;
    --cc-border: rgba(245, 158, 11, 0.14);
    --cc-border-subtle: rgba(255, 255, 255, 0.06);
    --cc-ink: #f4f4f5;
    --cc-muted: #a1a1aa;
    --cc-faint: #71717a;
    --cc-amber: #f59e0b;
    --cc-amber-soft: rgba(245, 158, 11, 0.15);
    --cc-beat: #34d399;
    --cc-miss: #f87171;
}

html, body, [class*="css"], .stApp, .stMarkdown, .stText {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: var(--cc-bg);
    background-image:
        radial-gradient(ellipse at 12% 0%, rgba(245, 158, 11, 0.07) 0%, transparent 42%),
        radial-gradient(ellipse at 88% 8%, rgba(245, 158, 11, 0.04) 0%, transparent 36%);
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1120px;
}

/* Header */
.cc-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--cc-border-subtle);
}

.cc-eyebrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cc-muted);
    margin-bottom: 0.35rem;
}

.cc-live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cc-amber);
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.7);
}

.cc-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--cc-ink);
    line-height: 1.15;
    margin: 0;
}

.cc-subtitle {
    margin-top: 0.35rem;
    font-size: 0.875rem;
    color: var(--cc-muted);
    font-variant-numeric: tabular-nums;
}

.cc-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.cc-badge.paper {
    background: var(--cc-amber-soft);
    color: var(--cc-amber);
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.cc-badge.live {
    background: rgba(248, 113, 113, 0.12);
    color: var(--cc-miss);
    border: 1px solid rgba(248, 113, 113, 0.35);
}

/* Metric strip */
.cc-metric {
    background: var(--cc-card);
    border: 1px solid var(--cc-border);
    border-radius: 0.85rem;
    padding: 0.95rem 1rem;
    height: 100%;
}

.cc-metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--cc-faint);
    margin-bottom: 0.4rem;
}

.cc-metric-value {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--cc-ink);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}

.cc-metric-value.cash {
    color: var(--cc-amber);
    font-size: 1.35rem;
}

.cc-metric-value.running { color: var(--cc-amber); }
.cc-metric-value.idle { color: var(--cc-beat); }
.cc-metric-value.error { color: var(--cc-miss); }
.cc-metric-value.unknown { color: var(--cc-muted); }

/* Sections */
.cc-section {
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

.cc-section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--cc-ink);
    margin-bottom: 0.25rem;
}

.cc-section-sub {
    font-size: 0.8rem;
    color: var(--cc-muted);
    margin-bottom: 0.9rem;
}

.cc-card {
    background: var(--cc-card);
    border: 1px solid var(--cc-border);
    border-radius: 0.9rem;
    padding: 1.1rem 1.2rem;
}

.cc-legend-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.4rem 0;
    font-size: 0.85rem;
    color: #d4d4d8;
    font-variant-numeric: tabular-nums;
}

.cc-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.cc-empty {
    color: var(--cc-faint);
    font-size: 0.875rem;
    padding: 0.75rem 0;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: var(--cc-card);
    border: 1px solid var(--cc-border);
    border-radius: 0.85rem;
    padding: 0.75rem 1rem;
}

div[data-testid="stMetric"] label {
    color: var(--cc-muted) !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--cc-ink) !important;
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #d97706, var(--cc-amber)) !important;
}

.stDataFrame {
    border-radius: 0.75rem;
    overflow: hidden;
    border: 1px solid var(--cc-border-subtle);
}

hr {
    border-color: var(--cc-border-subtle);
}

div[data-testid="stAlert"] {
    border-radius: 0.75rem;
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
    "CASH": "#f59e0b",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a1a1aa", family="DM Sans"),
    margin=dict(l=10, r=10, t=10, b=10),
)


def inject_theme():
    import streamlit as st
    st.markdown(COMMAND_CENTER_CSS, unsafe_allow_html=True)


def symbol_color(symbol: str) -> str:
    return SYMBOL_COLORS.get(symbol.upper(), "#f59e0b")
