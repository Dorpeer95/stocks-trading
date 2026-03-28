"""
Market breadth gate and distribution top detector.

Breadth gate: blocks new entries when fewer than 40% of S&P 500 stocks
are above their 200-day MA (^SPXA200R free yfinance index).

Market top detector: flags potential distribution when SPY accumulates
≥5 distribution days (down ≥0.2% on above-average volume) in 25 sessions,
or when SPY breaks below its 50-day MA.

All data via free yfinance tickers — zero additional API costs.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache — valid for one trading day each
# ---------------------------------------------------------------------------
_breadth_cache: Optional[Dict[str, Any]] = None
_breadth_cached_at: Optional[datetime] = None
_BREADTH_CACHE_HOURS = 8

_top_cache: Optional[Dict[str, Any]] = None
_top_cached_at: Optional[datetime] = None
_TOP_CACHE_HOURS = 8


def _is_fresh(cached_at: Optional[datetime], hours: int) -> bool:
    if cached_at is None:
        return False
    return (datetime.now(timezone.utc) - cached_at) < timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Breadth gate
# ---------------------------------------------------------------------------

def check_breadth_gate() -> Dict[str, Any]:
    """Check market breadth via % of S&P 500 stocks above 200-day MA.

    Uses CBOE indices available via yfinance:
      ^SPXA200R  — % of S&P 500 above 200 MA
      ^SPXA50R   — % of S&P 500 above 50 MA

    Returns
    -------
    Dict with ``allow_entries``, ``pct_above_200``, ``pct_above_50``,
    ``breadth_signal``, ``reason``.
    """
    global _breadth_cache, _breadth_cached_at

    if _is_fresh(_breadth_cached_at, _BREADTH_CACHE_HOURS) and _breadth_cache:
        return _breadth_cache

    pct_above_200 = 50.0
    pct_above_50 = 50.0

    try:
        data = yf.download(
            "^SPXA200R ^SPXA50R",
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10,
        )
        if data is not None and not data.empty:
            close = data["Close"] if "Close" in data.columns else data
            if hasattr(close, "columns"):
                if "^SPXA200R" in close.columns:
                    series = close["^SPXA200R"].dropna()
                    if not series.empty:
                        pct_above_200 = float(series.iloc[-1])
                if "^SPXA50R" in close.columns:
                    series = close["^SPXA50R"].dropna()
                    if not series.empty:
                        pct_above_50 = float(series.iloc[-1])
    except Exception as e:
        logger.warning(f"Breadth gate fetch failed: {e}")

    if pct_above_200 >= 55 and pct_above_50 >= 55:
        signal = "healthy"
        allow = True
        reason = f"{pct_above_200:.0f}% above 200MA — broad participation"
    elif pct_above_200 >= 40:
        signal = "weakening"
        allow = True  # allow but warn; scorer/sizer will reduce allocation
        reason = f"{pct_above_200:.0f}% above 200MA — breadth weakening"
    else:
        signal = "poor"
        allow = False
        reason = f"{pct_above_200:.0f}% above 200MA — too few stocks participating"

    result: Dict[str, Any] = {
        "allow_entries": allow,
        "pct_above_200": pct_above_200,
        "pct_above_50": pct_above_50,
        "breadth_signal": signal,
        "reason": reason,
    }
    _breadth_cache = result
    _breadth_cached_at = datetime.now(timezone.utc)
    logger.info(f"Breadth gate: {signal} — {reason}")
    return result


# ---------------------------------------------------------------------------
# Distribution / market top detector
# ---------------------------------------------------------------------------

def detect_market_top() -> Dict[str, Any]:
    """Detect distribution via SPY price/volume analysis.

    A distribution day = SPY closes down ≥0.2% on volume above its 20-day average.
    ≥5 distribution days in 25 sessions = high risk.

    Also checks whether SPY is below its 50-day MA.

    Returns
    -------
    Dict with ``top_risk``, ``distribution_days``, ``spy_above_50ma``,
    ``size_penalty``, ``reason``.
    """
    global _top_cache, _top_cached_at

    if _is_fresh(_top_cached_at, _TOP_CACHE_HOURS) and _top_cache:
        return _top_cache

    distribution_days = 0
    spy_above_50ma = True
    size_penalty = 0.0
    reason = "Normal market conditions"
    top_risk = "low"

    try:
        df = yf.download(
            "SPY",
            period="60d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10,
        )
        if df is not None and not df.empty and len(df) >= 20:
            close = df["Close"].values.astype(float).flatten()
            volume = df["Volume"].values.astype(float).flatten()

            # 50-day MA
            ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else float(np.mean(close))
            spy_above_50ma = float(close[-1]) > ma50

            # 20-day avg volume reference
            avg_vol_20 = float(np.mean(volume[-20:]))

            # Count distribution days in last 25 sessions
            last_n = min(25, len(close) - 1)
            for i in range(len(close) - last_n, len(close)):
                prev_close = close[i - 1]
                if prev_close == 0:
                    continue
                day_chg = (close[i] - prev_close) / prev_close
                if day_chg <= -0.002 and volume[i] > avg_vol_20:
                    distribution_days += 1

            if distribution_days >= 5 or not spy_above_50ma:
                top_risk = "high"
                size_penalty = 0.25
                parts = [f"{distribution_days} distribution days"]
                if not spy_above_50ma:
                    parts.append("SPY below 50MA")
                reason = " | ".join(parts)
            elif distribution_days >= 3:
                top_risk = "medium"
                size_penalty = 0.10
                reason = f"{distribution_days} distribution days — elevated caution"
            else:
                top_risk = "low"
                size_penalty = 0.0
                reason = f"{distribution_days} distribution days — trend intact"

    except Exception as e:
        logger.warning(f"Market top detection failed: {e}")

    result: Dict[str, Any] = {
        "top_risk": top_risk,
        "distribution_days": distribution_days,
        "spy_above_50ma": spy_above_50ma,
        "size_penalty": size_penalty,
        "reason": reason,
    }
    _top_cache = result
    _top_cached_at = datetime.now(timezone.utc)
    logger.info(f"Market top: {top_risk} — {reason}")
    return result


# ---------------------------------------------------------------------------
# Combined safety assessment
# ---------------------------------------------------------------------------

def get_market_safety() -> Dict[str, Any]:
    """Combine breadth gate + distribution top into a single safety call.

    Returns
    -------
    Dict with:
      ``safe_to_enter``  — bool: False = block new buys
      ``size_modifier``  — float 0.0–1.0: multiply against regime modifier
      ``breadth``        — raw breadth dict
      ``top``            — raw top dict
      ``summary``        — human-readable combined reason
    """
    breadth = check_breadth_gate()
    top = detect_market_top()

    safe_to_enter = breadth["allow_entries"] and top["top_risk"] != "high"

    # Build composite size modifier
    size_mod = 1.0
    if breadth["breadth_signal"] == "weakening":
        size_mod *= 0.85
    elif breadth["breadth_signal"] == "poor":
        size_mod *= 0.50
    size_mod = max(0.0, size_mod - top["size_penalty"])

    parts: list = []
    if not breadth["allow_entries"]:
        parts.append(breadth["reason"])
    if top["top_risk"] != "low":
        parts.append(top["reason"])
    summary = " | ".join(parts) if parts else "Market conditions healthy"

    return {
        "safe_to_enter": safe_to_enter,
        "size_modifier": round(size_mod, 2),
        "breadth": breadth,
        "top": top,
        "summary": summary,
    }
