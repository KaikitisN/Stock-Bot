"""
Provider-agnostic decision engine. Swapping AI models is just a
matter of changing `provider_name` — everything else stays identical
thanks to LiteLLM's unified completion() call.
"""
import json
import litellm
import config
from news_fetcher import get_news_summary

PROMPT_TEMPLATE = """You are a disciplined trading assistant. Given the market data below for {symbol},
decide one action: BUY, SELL, or HOLD.

Market data:
{market_data}

{news_block}

Respond ONLY with valid JSON in this exact format:
{{"action": "BUY|SELL|HOLD", "confidence": 0-100, "reason": "short explanation"}}
"""


def get_decision(symbol: str, market_data: dict, provider_name: str, use_news: bool) -> dict:
    model_string = config.AI_PROVIDERS.get(provider_name, config.AI_PROVIDERS[config.DEFAULT_AI_PROVIDER])
    news_block = ""
    if use_news:
        news_block = f"Recent news summary:\n{get_news_summary(symbol)}"
    prompt = PROMPT_TEMPLATE.format(
        symbol=symbol,
        market_data=json.dumps(market_data),
        news_block=news_block
    )
    try:
        response = litellm.completion(
            model=model_string,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response["choices"][0]["message"]["content"]
        start, end = raw.find("{"), raw.rfind("}") + 1
        decision = json.loads(raw[start:end])
    except Exception as e:
        decision = {"action": "HOLD", "confidence": 0, "reason": f"AI call failed: {e}"}
    decision["symbol"] = symbol
    decision["provider"] = provider_name
    return decision
