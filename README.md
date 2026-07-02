# AI Trading Bot — Alpaca + Kronos / Perplexity / OpenAI / Claude

An automated US stock trading bot with a Streamlit dashboard.
Supports **local AI inference via Kronos** (free, no API key) and cloud providers
(Perplexity Sonar, GPT-4o, Claude 3.5 Sonnet) — switchable from the dashboard.

---

## Quick Start

Python compatibility: use Python 3.11 or 3.12 for best support.
Python 3.14 can fail on scientific/ML wheels (torch/numpy/matplotlib builds).

```bash
# 1. Clone this repo
git clone https://github.com/KaikitisN/Stock-Bot.git
cd Stock-Bot

# 2. Virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install bot dependencies
pip install -r requirements.txt

# 4. (Optional but recommended) Clone Kronos for local inference
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
pip install -r ../Kronos/requirements.txt

# 5. Configure secrets
cp .env.example .env
# Fill in your Alpaca paper keys + any cloud AI keys you want

# 6. Launch dashboard
streamlit run dashboard.py
```

### Windows recovery (if `pip` breaks in venv)

```bash
# From project root (parent folder containing .venv)
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

If you still get build/metadata errors, recreate venv with Python 3.12 and reinstall:

```bash
# Example path from python.org installer
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## AI Provider Options

| Provider | Requires | Cost | Best for |
|---|---|---|---|
| **Kronos (Local)** | Clone Kronos repo + torch | Free forever | Price/pattern forecasting |
| **Perplexity Sonar** | `PERPLEXITY_API_KEY` | ~$5/mo (Pro credit) | News-aware decisions |
| **Perplexity Sonar Pro** | `PERPLEXITY_API_KEY` | Pay-as-you-go | Deep research + news |
| **OpenAI GPT-4o** | `OPENAI_API_KEY` | Pay-as-you-go | General reasoning |
| **OpenAI GPT-4o-mini** | `OPENAI_API_KEY` | Pay-as-you-go | Fast + cheap |
| **Claude 3.5 Sonnet** | `ANTHROPIC_API_KEY` | Pay-as-you-go | Strong reasoning |

### Kronos model sizes
| Size | Parameters | RAM needed | Device |
|---|---|---|---|
| `mini` | 4.1M | ~1 GB | CPU ✅ |
| `small` | ~50M | ~2 GB | GPU recommended |
| `base` | ~150M | ~4 GB | GPU |
| `large` | 499M | ~8 GB | GPU required |

Set `KRONOS_MODEL_SIZE=mini` in `.env` to start (CPU-friendly).

---

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | All defaults: risk %, AI providers, frequency options |
| `data_fetcher.py` | OHLCV bars + SMA-10, SMA-30, RSI-14 from Alpaca |
| `kronos_decision.py` | **Local Kronos inference** — price forecast → BUY/SELL/HOLD |
| `news_fetcher.py` | Optional Perplexity Sonar news summary (toggle in dashboard) |
| `ai_decision.py` | Routes to Kronos or LiteLLM depending on selected provider |
| `risk_manager.py` | Position sizing, stop-loss/take-profit as % of account equity |
| `executor.py` | Bracket orders (entry + SL + TP) via alpaca-py |
| `orchestrator.py` | One full decision-and-trade cycle, logs to CSV |
| `dashboard.py` | Streamlit UI: model picker, news toggle, frequency, risk sliders |

---

## Dashboard Features
- **AI model selector** — Kronos (local), Perplexity Sonar/Pro, GPT-4o/mini, Claude 3.5
- **News toggle** — adds live Perplexity Sonar news to any provider's decision
- **Frequency selector** — 5 min / 15 min / 30 min / hourly / daily
- **Risk sliders** — max position size, stop-loss, take-profit, max daily loss (% of equity)
- **Account overview** — live equity, cash, open positions
- **History tables + chart** — all AI decisions and trades logged to CSV

---

## Paper vs Live Trading

`ALPACA_PAPER=true` → free paper account ($100k simulated).  
`ALPACA_PAPER=false` → live individual account (no minimum deposit for non-US residents).

---

## Risk Logic
- AI confidence must be **≥ 60** for a trade to execute.
- Position size is always capped at your `max_position_pct` of current equity.
- Bracket orders attach stop-loss and take-profit at submission.
- Daily loss limit halts the bot for the rest of the day if breached.
