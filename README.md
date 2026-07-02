# AI Trading Bot — Alpaca + Perplexity / OpenAI / Claude

An automated US stock trading bot powered by AI (Perplexity Sonar, GPT-4o, or Claude 3.5 Sonnet),
executing trades through Alpaca with a Streamlit dashboard for full control.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/KaikitisN/Stock-Bot.git
cd Stock-Bot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in your Alpaca paper keys + at least one AI provider key

# 5. Launch the dashboard
streamlit run dashboard.py
```

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | All defaults: risk %, AI providers, frequency options |
| `data_fetcher.py` | Fetches OHLCV bars + computes SMA-10, SMA-30, RSI-14 from Alpaca |
| `news_fetcher.py` | Optional Perplexity Sonar news summary (toggle in dashboard) |
| `ai_decision.py` | LiteLLM wrapper — same code works for Perplexity, GPT, or Claude |
| `risk_manager.py` | Position sizing, stop-loss/take-profit as % of account equity |
| `executor.py` | Places bracket orders (entry + SL + TP) via alpaca-py SDK |
| `orchestrator.py` | Runs one full decision-and-trade cycle, logs to CSV |
| `dashboard.py` | Streamlit UI: model picker, news toggle, frequency, risk sliders |

## Dashboard Features
- **AI model selector** — switch between Perplexity Sonar, Sonar Pro, GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet
- **News toggle** — factor in live Perplexity Sonar news summaries per symbol
- **Frequency selector** — 5 min / 15 min / 30 min / hourly / daily
- **Risk sliders** — max position size, stop-loss, take-profit, max daily loss (all as % of equity)
- **Account overview** — live equity, cash, and open positions
- **History tables + chart** — all AI decisions and executed trades logged to CSV

## Paper vs Live Trading

Set `ALPACA_PAPER=true` in `.env` to use Alpaca's free paper trading account ($100k simulated balance).
Change to `ALPACA_PAPER=false` only after opening and funding an individual Alpaca live account.

> **Note:** As a non-US individual (e.g. Cyprus resident), you can open a live Alpaca account
> with no minimum deposit. Funding and withdrawals go via international wire or Wise.

## Risk Logic
- The AI returns a JSON decision with `action` (BUY/SELL/HOLD) and `confidence` (0–100).
- Trades are only submitted if `confidence >= 60`.
- Position size is always capped at your configured `max_position_pct` of current equity.
- Bracket orders attach stop-loss and take-profit automatically at order submission.
- Daily loss limit halts the bot for the rest of the trading day if breached.
