"""
Macro-event and regime analysis driven by global news.

Fetches recent global news headlines, queries GPT-4o-mini for an
assessment of benefiting vs at-risk sectors based on world events,
and provides adjustments for the scoring engine.
"""

import logging
import json
import time
from typing import Dict, Any, List
from utils.sentiment import fetch_market_news, _call_openai

logger = logging.getLogger(__name__)

_MACRO_CACHE = None
_MACRO_CACHE_TS = 0
MACRO_CACHE_TTL = 12 * 3600  # 12 hours


def get_macro_bias() -> Dict[str, Any]:
    """Fetch recent market news and use GPT-4o-mini to establish sector bias.
    
    Returns a dict with 'benefiting' and 'at_risk' sector lists.
    Uses in-memory caching to limit API calls (every 12 hours).
    """
    global _MACRO_CACHE, _MACRO_CACHE_TS
    if _MACRO_CACHE and time.time() - _MACRO_CACHE_TS < MACRO_CACHE_TTL:
        return _MACRO_CACHE

    articles = fetch_market_news(max_articles=15)
    default_resp = {"benefiting": [], "at_risk": [], "mood": "Neutral"}
    
    if not articles:
        return default_resp

    headlines = [a.get("title", "") for a in articles]
    text = "\n".join(f"- {h}" for h in headlines if h)

    sys_prompt = (
        "You are a macro-economic analyst. Analyze these recent global business "
        "headlines and identify which 3 stock sectors are most likely to benefit, "
        "and which 3 are at highest risk. Output pure JSON."
    )
    
    usr_prompt = f"""Headlines:
{text}

Output format:
{{
  "benefiting": ["Technology", "Energy", "Financials"],
  "at_risk": ["Real Estate", "Utilities", "Consumer Discretionary"],
  "mood": "Bullish"
}}"""

    result = _call_openai(
        model="gpt-4o-mini",
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        max_tokens=200,
        temperature=0.2
    )

    if not result:
        return default_resp

    try:
        content = result["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lower().startswith("json"):
                content = content[4:]
        
        parsed = json.loads(content.strip())
        
        _MACRO_CACHE = {
            "benefiting": [s.lower() for s in parsed.get("benefiting", [])],
            "at_risk": [s.lower() for s in parsed.get("at_risk", [])],
            "mood": parsed.get("mood", "Neutral")
        }
        _MACRO_CACHE_TS = time.time()
        logger.info(f"Updated macro bias: {_MACRO_CACHE}")
        return _MACRO_CACHE
        
    except Exception as e:
        logger.error(f"Failed to parse macro bias: {e}")
        return default_resp


def apply_macro_bias(score: float, sector: str, bias: Dict[str, Any]) -> float:
    """Adjust the macro sub-score based on GPT sector bias."""
    if not sector or not bias:
        return score
    
    sec = sector.lower()
    
    # Check if the sector name roughly matches any benefiting / at_risk targets
    is_benefiting = any(b in sec or sec in b for b in bias.get("benefiting", []))
    is_at_risk = any(r in sec or sec in r for r in bias.get("at_risk", []))

    if is_benefiting:
        score += 15.0
    elif is_at_risk:
        score -= 15.0

    return max(0.0, min(100.0, score))
