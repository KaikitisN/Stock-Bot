"""
Central configuration. Values here are defaults — the Streamlit dashboard
lets you override AI_PROVIDER, RUN_INTERVAL_MINUTES, USE_NEWS and risk %
at runtime without editing this file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Broker ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# --- AI provider selection ---
# "kronos_local" is handled separately in kronos_decision.py (no API key needed).
# All other values are LiteLLM model strings.
AI_PROVIDERS = {
    "Kronos (Local)": "kronos_local",          # runs on your machine, free
    "Perplexity Sonar": "perplexity/sonar",
    "Perplexity Sonar Pro": "perplexity/sonar-pro",
    "OpenAI GPT-4o": "gpt-4o",
    "OpenAI GPT-4o-mini": "gpt-4o-mini",
    "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
}
DEFAULT_AI_PROVIDER = "Kronos (Local)"

# --- Kronos local model size ---
# mini (~1 GB, CPU ok) | small (~2 GB) | base (~4 GB) | large (~8 GB GPU)
KRONOS_MODEL_SIZE = os.getenv("KRONOS_MODEL_SIZE", "mini")
KRONOS_REPO_PATH = os.getenv("KRONOS_REPO_PATH", "../Kronos")
# Minimum forecast move required before Kronos turns a prediction into BUY/SELL.
KRONOS_SIGNAL_THRESHOLD_PCT = float(os.getenv("KRONOS_SIGNAL_THRESHOLD_PCT", "1.0"))

# --- News toggle (Perplexity Sonar) ---
USE_NEWS_DEFAULT = False

# --- Trading frequency options (minutes) ---
FREQUENCY_OPTIONS = {
    "Every 5 minutes": 5,
    "Every 15 minutes": 15,
    "Every 30 minutes": 30,
    "Every hour": 60,
    "Once a day": 1440,
}
DEFAULT_FREQUENCY = "Every 30 minutes"

# --- Universe ---
DEFAULT_SYMBOLS = [    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "BABA", "PYPL", "SQ", "SHOP", "COIN", "HOOD",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA",
    "SPY", "QQQ", "DIA", "IWM", "ARKK",
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "XRP/USD",
    "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD"]

# --- Risk limits (all as % of account equity) ---
DEFAULT_RISK = {
    "max_position_pct": 5.0,
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0,
    "max_daily_loss_pct": 3.0,
}

LOG_DIR = "logs"
DECISIONS_LOG = f"{LOG_DIR}/decisions.csv"
TRADES_LOG = f"{LOG_DIR}/trades.csv"
