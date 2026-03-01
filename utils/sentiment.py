"""
Sentiment analysis pipeline — VADER + NewsAPI + GPT-4o.

Provides headline sentiment scoring (VADER, free and local) with
optional GPT-4o-mini enhancement for deeper analysis.  Includes a
``CostTracker`` that enforces a hard $5/month cap on OpenAI spend.

Rate limits
-----------
- **NewsAPI free tier:** 100 requests/day  →  bot uses ~21/day.
- **OpenAI:** $5/month hard cap enforced by ``CostTracker``.

Fallback chain
--------------
1. VADER (always available, zero cost)
2. GPT-4o-mini analysis (if budget remains)
3. GPT-4o for weekly briefing only (if budget remains)
4. Template strings if all else fails
"""

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEWSAPI_KEY = os.getenv("STOCKS_NEWSAPI_KEY", "")
OPENAI_API_KEY = os.getenv("STOCKS_OPENAI_API_KEY", "")

ENABLE_NEWS = os.getenv("STOCKS_ENABLE_NEWS", "true").lower() == "true"
ENABLE_GPT = os.getenv("STOCKS_ENABLE_GPT", "true").lower() == "true"

NEWSAPI_BASE = "https://newsapi.org/v2"
OPENAI_BASE = "https://api.openai.com/v1"

# Cost limits
MONTHLY_BUDGET_USD = float(os.getenv("STOCKS_GPT_MONTHLY_BUDGET", "5.00"))
KILL_SWITCH_USD = float(os.getenv("STOCKS_GPT_KILL_SWITCH", "4.00"))

# Pricing (per 1M tokens) — GPT-4o-mini & GPT-4o as of 2025
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# News cache TTL
NEWS_CACHE_TTL = 6 * 3600  # 6 hours

# ---------------------------------------------------------------------------
# News cache (in-memory, 6-hour TTL)
# ---------------------------------------------------------------------------
_news_cache: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
_newsapi_calls_today = 0
_newsapi_calls_date: Optional[str] = None
NEWSAPI_DAILY_LIMIT = 90  # stop at 90 to leave 10 buffer


# ═══════════════════════════════════════════════════════════════════════════
# CostTracker
# ═══════════════════════════════════════════════════════════════════════════

class CostTracker:
    """Tracks OpenAI API costs and enforces a monthly budget cap.

    All costs are tracked in memory and reset on the 1st of each month.
    The tracker is intentionally conservative — it rounds UP token counts.

    Usage::

        tracker = CostTracker()
        if tracker.can_spend(estimated_cost=0.01):
            response = call_openai(...)
            tracker.record(model, prompt_tokens, completion_tokens)
    """

    def __init__(
        self,
        monthly_budget: float = MONTHLY_BUDGET_USD,
        kill_switch: float = KILL_SWITCH_USD,
    ) -> None:
        self.monthly_budget = monthly_budget
        self.kill_switch = kill_switch

        self._month: str = date.today().strftime("%Y-%m")
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._cost_by_model: Dict[str, float] = {}
        self._tokens_by_model: Dict[str, Dict[str, int]] = {}
        self._killed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_spend(self, estimated_cost: float = 0.01) -> bool:
        """Check if the estimated cost is within budget."""
        self._maybe_reset_month()

        if self._killed:
            return False

        if self._total_cost + estimated_cost > self.monthly_budget:
            logger.warning(
                f"CostTracker: budget exhausted "
                f"(${self._total_cost:.4f} / ${self.monthly_budget:.2f})"
            )
            return False

        if self._total_cost >= self.kill_switch:
            logger.warning(
                f"CostTracker: kill switch triggered at "
                f"${self._total_cost:.4f}"
            )
            self._killed = True
            return False

        return True

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Record an API call and return its cost in USD."""
        self._maybe_reset_month()

        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        call_cost = input_cost + output_cost

        self._total_cost += call_cost
        self._call_count += 1
        self._cost_by_model[model] = (
            self._cost_by_model.get(model, 0.0) + call_cost
        )

        model_tokens = self._tokens_by_model.setdefault(
            model, {"input": 0, "output": 0}
        )
        model_tokens["input"] += prompt_tokens
        model_tokens["output"] += completion_tokens

        logger.debug(
            f"CostTracker: ${call_cost:.6f} "
            f"({model}, {prompt_tokens}+{completion_tokens} tok) "
            f"total=${self._total_cost:.4f}"
        )

        return call_cost

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of current month's spending."""
        self._maybe_reset_month()
        return {
            "month": self._month,
            "total_cost_usd": round(self._total_cost, 6),
            "budget_usd": self.monthly_budget,
            "remaining_usd": round(
                self.monthly_budget - self._total_cost, 6
            ),
            "utilization_pct": round(
                (self._total_cost / self.monthly_budget) * 100, 2
            )
            if self.monthly_budget > 0
            else 0,
            "call_count": self._call_count,
            "cost_by_model": dict(self._cost_by_model),
            "tokens_by_model": dict(self._tokens_by_model),
            "killed": self._killed,
        }

    @property
    def total_cost(self) -> float:
        self._maybe_reset_month()
        return self._total_cost

    @property
    def budget_remaining(self) -> float:
        self._maybe_reset_month()
        return max(0, self.monthly_budget - self._total_cost)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_reset_month(self) -> None:
        """Reset counters on the 1st of a new month."""
        current_month = date.today().strftime("%Y-%m")
        if current_month != self._month:
            logger.info(
                f"CostTracker: new month {current_month} — "
                f"resetting (was ${self._total_cost:.4f})"
            )
            self._month = current_month
            self._total_cost = 0.0
            self._call_count = 0
            self._cost_by_model.clear()
            self._tokens_by_model.clear()
            self._killed = False


# Module-level singleton
cost_tracker = CostTracker()


# ═══════════════════════════════════════════════════════════════════════════
# NewsAPI
# ═══════════════════════════════════════════════════════════════════════════

def _reset_daily_counter() -> None:
    """Reset the NewsAPI daily counter if the date has changed."""
    global _newsapi_calls_today, _newsapi_calls_date
    today = date.today().isoformat()
    if _newsapi_calls_date != today:
        _newsapi_calls_today = 0
        _newsapi_calls_date = today


def fetch_news(
    ticker: str,
    max_articles: int = 10,
) -> List[Dict[str, str]]:
    """Fetch recent news headlines for a ticker from NewsAPI.

    Parameters
    ----------
    ticker : Stock ticker symbol (e.g. ``'AAPL'``).
    max_articles : Maximum number of articles to return.

    Returns
    -------
    List of dicts with keys: ``title``, ``description``, ``source``,
    ``url``, ``published_at``.
    """
    global _newsapi_calls_today

    if not ENABLE_NEWS or not NEWSAPI_KEY:
        logger.debug("NewsAPI disabled or no API key")
        return []

    # Check cache
    cache_key = ticker.upper()
    if cache_key in _news_cache:
        ts, articles = _news_cache[cache_key]
        if time.time() - ts < NEWS_CACHE_TTL:
            return articles

    # Check daily limit
    _reset_daily_counter()
    if _newsapi_calls_today >= NEWSAPI_DAILY_LIMIT:
        logger.warning(
            f"NewsAPI daily limit reached ({_newsapi_calls_today}/"
            f"{NEWSAPI_DAILY_LIMIT}) — skipping {ticker}"
        )
        return []

    try:
        # Map ticker to company name for better search results
        query = _ticker_to_query(ticker)

        resp = httpx.get(
            f"{NEWSAPI_BASE}/everything",
            params={
                "q": query,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": max_articles,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        _newsapi_calls_today += 1

        if resp.status_code != 200:
            logger.warning(
                f"NewsAPI error for {ticker}: {resp.status_code}"
            )
            return []

        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })

        # Cache
        _news_cache[cache_key] = (time.time(), articles)
        logger.debug(
            f"NewsAPI: {len(articles)} articles for {ticker} "
            f"(call #{_newsapi_calls_today})"
        )
        return articles

    except Exception as e:
        logger.error(f"NewsAPI fetch failed for {ticker}: {e}")
        return []


def fetch_market_news(max_articles: int = 10) -> List[Dict[str, str]]:
    """Fetch general market/business news headlines."""
    global _newsapi_calls_today

    if not ENABLE_NEWS or not NEWSAPI_KEY:
        return []

    _reset_daily_counter()
    if _newsapi_calls_today >= NEWSAPI_DAILY_LIMIT:
        return []

    try:
        resp = httpx.get(
            f"{NEWSAPI_BASE}/top-headlines",
            params={
                "category": "business",
                "language": "en",
                "pageSize": max_articles,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        _newsapi_calls_today += 1

        if resp.status_code != 200:
            return []

        data = resp.json()
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in data.get("articles", [])
        ]

    except Exception as e:
        logger.error(f"Market news fetch failed: {e}")
        return []


def _ticker_to_query(ticker: str) -> str:
    """Convert ticker to a search query (add company name if known)."""
    # Common tickers → better search terms
    _TICKER_MAP = {
        "AAPL": "Apple AAPL stock",
        "MSFT": "Microsoft MSFT stock",
        "GOOGL": "Google Alphabet GOOGL stock",
        "AMZN": "Amazon AMZN stock",
        "NVDA": "NVIDIA NVDA stock",
        "META": "Meta Facebook META stock",
        "TSLA": "Tesla TSLA stock",
        "JPM": "JPMorgan JPM stock",
        "V": "Visa stock",
        "UNH": "UnitedHealth UNH stock",
    }
    return _TICKER_MAP.get(ticker.upper(), f"{ticker} stock")


def get_newsapi_usage() -> Dict[str, Any]:
    """Return current NewsAPI usage stats."""
    _reset_daily_counter()
    return {
        "calls_today": _newsapi_calls_today,
        "daily_limit": NEWSAPI_DAILY_LIMIT,
        "remaining": NEWSAPI_DAILY_LIMIT - _newsapi_calls_today,
        "date": _newsapi_calls_date,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VADER Sentiment
# ═══════════════════════════════════════════════════════════════════════════

def vader_score(headlines: List[str]) -> Dict[str, Any]:
    """Score a list of headlines using VADER sentiment analyzer.

    Parameters
    ----------
    headlines : List of headline strings.

    Returns
    -------
    Dict with ``score`` (-10 to +10), ``positive``, ``negative``,
    ``neutral`` counts, and ``raw_scores`` list.
    """
    if not headlines:
        return {
            "score": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "raw_scores": [],
            "headline_count": 0,
        }

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("vaderSentiment not installed — returning neutral")
        return {
            "score": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": len(headlines),
            "raw_scores": [],
            "headline_count": len(headlines),
        }

    raw_scores: List[float] = []
    positive = 0
    negative = 0
    neutral = 0

    for headline in headlines:
        if not headline or not headline.strip():
            continue

        scores = analyzer.polarity_scores(headline)
        compound = scores["compound"]
        raw_scores.append(compound)

        if compound >= 0.05:
            positive += 1
        elif compound <= -0.05:
            negative += 1
        else:
            neutral += 1

    if not raw_scores:
        return {
            "score": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "raw_scores": [],
            "headline_count": 0,
        }

    # VADER compound is [-1, 1] → scale to [-10, +10]
    avg_compound = sum(raw_scores) / len(raw_scores)
    sentiment_score = round(avg_compound * 10, 2)

    return {
        "score": sentiment_score,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "raw_scores": raw_scores,
        "headline_count": len(raw_scores),
    }


# ═══════════════════════════════════════════════════════════════════════════
# GPT-4o Integration
# ═══════════════════════════════════════════════════════════════════════════

def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """Call the OpenAI Chat Completions API.

    Returns dict with ``content``, ``prompt_tokens``,
    ``completion_tokens``, ``model``, or None on failure.
    """
    if not ENABLE_GPT or not OPENAI_API_KEY:
        logger.debug("GPT disabled or no API key")
        return None

    # Estimate cost before calling
    est_input_tokens = len(system_prompt.split()) + len(user_prompt.split())
    est_input_tokens = int(est_input_tokens * 1.3)  # rough token estimate
    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
    est_cost = (
        (est_input_tokens / 1_000_000) * pricing["input"]
        + (max_tokens / 1_000_000) * pricing["output"]
    )

    if not cost_tracker.can_spend(est_cost):
        logger.warning("GPT budget exhausted — using fallback")
        return None

    try:
        resp = httpx.post(
            f"{OPENAI_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            logger.error(f"OpenAI API error: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", est_input_tokens)
        completion_tokens = usage.get("completion_tokens", 0)

        # Record actual cost
        cost_tracker.record(model, prompt_tokens, completion_tokens)

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model,
        }

    except httpx.TimeoutException:
        logger.error("OpenAI API timeout")
        return None
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None


def gpt4o_mini_analyze(
    headlines: List[str],
    ticker: str,
) -> Optional[Dict[str, Any]]:
    """Use GPT-4o-mini to analyze news headlines for a stock.

    Returns dict with ``summary`` (plain English), ``sentiment``
    (-10 to +10), ``key_themes`` (list), ``risk_flags`` (list).
    """
    if not headlines:
        return None

    # Limit to top 10 headlines to control token count
    headlines_text = "\n".join(
        f"- {h}" for h in headlines[:10] if h.strip()
    )

    system_prompt = (
        "You are a stock market analyst assistant. Analyze news headlines "
        "and provide a brief sentiment assessment. Be concise and specific. "
        "Respond in valid JSON only."
    )

    user_prompt = f"""Analyze these recent news headlines for {ticker}:

{headlines_text}

Respond with JSON:
{{
  "summary": "1-2 sentence summary of the news sentiment",
  "sentiment": <number from -10 to +10>,
  "key_themes": ["theme1", "theme2"],
  "risk_flags": ["flag1"] or []
}}"""

    result = _call_openai(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=300,
        temperature=0.2,
    )

    if result is None:
        return None

    try:
        content = result["content"].strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        return {
            "summary": parsed.get("summary", ""),
            "sentiment": max(-10, min(10, float(parsed.get("sentiment", 0)))),
            "key_themes": parsed.get("key_themes", []),
            "risk_flags": parsed.get("risk_flags", []),
            "model": "gpt-4o-mini",
            "tokens_used": (
                result["prompt_tokens"] + result["completion_tokens"]
            ),
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse GPT response for {ticker}: {e}")
        return None


def gpt4o_earnings_analysis(
    ticker: str,
    earnings_data: Dict[str, Any],
    headlines: Optional[List[str]] = None,
) -> Optional[str]:
    """Use GPT-4o to produce a plain-English earnings analysis.

    Parameters
    ----------
    ticker : Stock ticker.
    earnings_data : Dict with keys like ``beat_streak``,
                    ``last_surprise_pct``, ``revenue_growth``, etc.
    headlines : Optional recent headlines about this earnings.

    Returns
    -------
    Plain English analysis string, or None.
    """
    system_prompt = (
        "You are a stock market analyst. Write a brief, plain-English "
        "earnings analysis for a swing trader. Focus on what matters "
        "for the next 1-6 weeks. Be concise (3-4 sentences max)."
    )

    earnings_text = json.dumps(earnings_data, indent=2, default=str)

    user_prompt = f"Analyze {ticker}'s recent earnings:\n\n{earnings_text}"

    if headlines:
        news_text = "\n".join(f"- {h}" for h in headlines[:5])
        user_prompt += f"\n\nRelated headlines:\n{news_text}"

    result = _call_openai(
        model="gpt-4o",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=300,
        temperature=0.3,
    )

    if result is None:
        return None

    return result["content"].strip()


def gpt4o_weekly_briefing(
    data: Dict[str, Any],
) -> Optional[str]:
    """Use GPT-4o to write the weekly market briefing.

    Parameters
    ----------
    data : Dict with keys:
        - ``opportunities`` : top picks list
        - ``market_mood`` : str
        - ``hot_sectors`` : list
        - ``regime`` : market regime dict
        - ``events`` : recent events list

    Returns
    -------
    Plain English briefing string, or None.
    """
    system_prompt = (
        "You are a stock market analyst writing a weekly briefing for a "
        "swing trader. Write in clear, conversational English. Be specific "
        "about tickers, levels, and actionable takeaways. Keep it under "
        "300 words. Use bullet points where helpful."
    )

    # Build a compact summary for the prompt
    opps = data.get("opportunities", [])
    opp_summary = []
    for o in opps[:5]:
        opp_summary.append(
            f"  {o.get('ticker')}: confidence {o.get('confidence')}, "
            f"setup={o.get('setup_type')}, "
            f"entry=${o.get('entry_price_low')}-${o.get('entry_price_high')}, "
            f"stop=${o.get('stop_loss')}, target=${o.get('target_price')}"
        )

    mood = data.get("market_mood", "Neutral")
    hot = data.get("hot_sectors", [])
    events = data.get("events", [])

    user_prompt = f"""Write a weekly stock market briefing:

Market mood: {mood}
Hot sectors: {', '.join(hot) if hot else 'None standout'}

Top opportunities:
{chr(10).join(opp_summary) if opp_summary else '  No high-conviction picks this week'}

Recent events: {', '.join(str(e) for e in events[:5]) if events else 'None notable'}

Write the briefing now."""

    result = _call_openai(
        model="gpt-4o",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=800,
        temperature=0.4,
    )

    if result is None:
        return None

    return result["content"].strip()


def gpt4o_exit_analysis(
    position: Dict[str, Any],
    current_indicators: Dict[str, Any],
    news_summary: Optional[str] = None,
) -> Optional[str]:
    """Use GPT-4o-mini to help with exit decision on a position.

    Returns a plain-English recommendation string.
    """
    system_prompt = (
        "You are a swing trading assistant. Analyze the current position "
        "and provide a brief hold/sell recommendation with reasoning. "
        "Be specific about price levels. 2-3 sentences max."
    )

    pos_info = (
        f"Ticker: {position.get('ticker')}\n"
        f"Entry: ${position.get('entry_price')}\n"
        f"Current: ${position.get('current_price')}\n"
        f"Stop: ${position.get('stop_loss')}\n"
        f"Target: ${position.get('target_price')}\n"
        f"Days held: {position.get('days_held')}\n"
        f"P&L: {position.get('unrealized_pnl_pct', 0):.1f}%"
    )

    indicators_info = (
        f"RSI: {current_indicators.get('rsi_14')}\n"
        f"MACD: {current_indicators.get('macd_signal')}\n"
        f"Volume ratio: {current_indicators.get('volume_ratio')}"
    )

    user_prompt = f"Position:\n{pos_info}\n\nCurrent technicals:\n{indicators_info}"

    if news_summary:
        user_prompt += f"\n\nRecent news: {news_summary}"

    result = _call_openai(
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=200,
        temperature=0.3,
    )

    if result is None:
        return None

    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_sentiment(
    ticker: str,
    use_gpt: bool = True,
) -> Dict[str, Any]:
    """Full sentiment pipeline for a single ticker.

    Steps:
    1. Fetch news from NewsAPI
    2. Score headlines with VADER
    3. Optionally enhance with GPT-4o-mini analysis
    4. Return aggregated sentiment

    Parameters
    ----------
    ticker : Stock ticker symbol.
    use_gpt : Whether to attempt GPT enhancement.

    Returns
    -------
    Dict with:
    - ``score`` : -10 to +10 sentiment score
    - ``vader_score`` : VADER-only score
    - ``gpt_score`` : GPT score (if available)
    - ``headlines`` : list of headline strings
    - ``gpt_summary`` : GPT analysis summary (if available)
    - ``key_themes`` : list of themes (if available)
    - ``risk_flags`` : list of risk flags (if available)
    - ``source`` : 'vader+gpt' or 'vader' or 'none'
    """
    # 1. Fetch news
    articles = fetch_news(ticker)
    headlines = [a["title"] for a in articles if a.get("title")]

    if not headlines:
        return {
            "score": 0.0,
            "vader_score": 0.0,
            "gpt_score": None,
            "headlines": [],
            "gpt_summary": None,
            "key_themes": [],
            "risk_flags": [],
            "source": "none",
        }

    # 2. VADER score
    vader = vader_score(headlines)
    v_score = vader["score"]

    # 3. GPT enhancement (if enabled + budget remains)
    gpt_score = None
    gpt_summary = None
    key_themes: List[str] = []
    risk_flags: List[str] = []
    source = "vader"

    if use_gpt and ENABLE_GPT:
        gpt_result = gpt4o_mini_analyze(headlines, ticker)
        if gpt_result:
            gpt_score = gpt_result["sentiment"]
            gpt_summary = gpt_result["summary"]
            key_themes = gpt_result.get("key_themes", [])
            risk_flags = gpt_result.get("risk_flags", [])
            source = "vader+gpt"

    # 4. Aggregate: blend VADER and GPT if both available
    if gpt_score is not None:
        # Weight GPT more heavily — it understands context better
        final_score = round(v_score * 0.3 + gpt_score * 0.7, 2)
    else:
        final_score = v_score

    return {
        "score": final_score,
        "vader_score": v_score,
        "gpt_score": gpt_score,
        "headlines": headlines,
        "gpt_summary": gpt_summary,
        "key_themes": key_themes,
        "risk_flags": risk_flags,
        "source": source,
    }


def batch_sentiment(
    tickers: List[str],
    use_gpt: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run sentiment analysis for a batch of tickers.

    Returns ``{ticker: sentiment_dict}`` for each ticker.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for ticker in tickers:
        try:
            results[ticker] = aggregate_sentiment(ticker, use_gpt=use_gpt)
        except Exception as e:
            logger.warning(f"Sentiment failed for {ticker}: {e}")
            results[ticker] = {
                "score": 0.0,
                "vader_score": 0.0,
                "gpt_score": None,
                "headlines": [],
                "gpt_summary": None,
                "key_themes": [],
                "risk_flags": [],
                "source": "error",
            }

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return results
