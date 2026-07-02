"""
Optional news/sentiment layer, only called when the dashboard's
'Factor in news' toggle is ON. Uses Perplexity Sonar directly since
it is the only provider with live web search built in.
"""
import os
import requests

PPLX_URL = "https://api.perplexity.ai/chat/completions"


def get_news_summary(symbol: str) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        return "No news available (Perplexity API key not set)."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a financial news summarizer. Be concise, factual, and neutral."},
            {"role": "user", "content": f"Summarize the most important news from the last 24 hours for {symbol} stock in 2-3 sentences, noting if sentiment is bullish, bearish, or neutral."},
        ],
    }
    try:
        resp = requests.post(PPLX_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"News fetch failed: {e}"
