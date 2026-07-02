"""
Central configuration. Values here are defaults - the Streamlit dashboard
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

# --- AI provider selection (swappable via LiteLLM) ---
AI_PROVIDERS = {
    "Perplexity Sonar":      "perplexity/sonar",
    "Perplexity Sonar Pro":  "perplexity/sonar-pro",
    "OpenAI GPT-4o":         "gpt-4o",
    "OpenAI GPT-4o-mini":    "gpt-4o-mini",
    "Claude 3.5 Sonnet":     "claude-3-5-sonnet-20241022",
}
DEFAULT_AI_PROVIDER = "Perplexity Sonar"

# --- News toggle ---
USE_NEWS_DEFAULT = False

# --- Trading frequency options (minutes) ---
FREQUENCY_OPTIONS = {
    "Every 5 minutes": 5,
    "Every 15 minutes": 15,
    "Every 30 minutes": 30,
    "Every hour": 60,
    "Once a day": 1440,
}
DEFAULT_FREQUENCY = "Every 15 minutes"

# --- Universe ---
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA"]

# --- Risk limits (all as % of account equity) ---
DEFAULT_RISK = {
    "max_position_pct": 5.0,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "max_daily_loss_pct": 3.0,
}

LOG_DIR = "logs"
DECISIONS_LOG = f"{LOG_DIR}/decisions.csv"
TRADES_LOG = f"{LOG_DIR}/trades.csv"
