"""
Earnings calendar and history utilities.

Fetches earnings dates, beat/miss history, and days-to-earnings
using yfinance. Results are cached.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_earnings_dates(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch upcoming and recent earnings dates for a ticker.

    Returns
    -------
    Dict with keys: ``next_earnings_date``, ``days_to_earnings``,
    ``previous_earnings_date``, ``is_within_2_weeks``.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # Try calendar first
        next_date = None
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    next_date = pd.Timestamp(val).date()
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    dates = cal["Earnings Date"]
                    if dates:
                        next_date = pd.Timestamp(dates[0]).date()
        except Exception:
            pass

        # Fall back to earnings_dates attribute
        prev_date = None
        try:
            earnings_hist = t.earnings_dates
            if earnings_hist is not None and not earnings_hist.empty:
                today = date.today()
                for idx in earnings_hist.index:
                    d = pd.Timestamp(idx).date()
                    if d >= today and next_date is None:
                        next_date = d
                    elif d < today and prev_date is None:
                        prev_date = d
                    if next_date and prev_date:
                        break
        except Exception:
            pass

        today = date.today()
        days_to = (next_date - today).days if next_date else None
        within_2w = days_to is not None and 0 <= days_to <= 14

        return {
            "ticker": ticker,
            "next_earnings_date": str(next_date) if next_date else None,
            "days_to_earnings": days_to,
            "previous_earnings_date": str(prev_date) if prev_date else None,
            "is_within_2_weeks": within_2w,
        }

    except Exception as e:
        logger.warning(f"fetch_earnings_dates failed for {ticker}: {e}")
        return None


def fetch_earnings_history(ticker: str, quarters: int = 8) -> Optional[List[Dict[str, Any]]]:
    """Fetch recent earnings beat/miss history.

    Returns
    -------
    List of dicts with keys: ``date``, ``estimated_eps``,
    ``actual_eps``, ``surprise_pct``, ``beat``.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        earnings_hist = t.earnings_dates

        if earnings_hist is None or earnings_hist.empty:
            logger.debug(f"No earnings history for {ticker}")
            return None

        records: List[Dict[str, Any]] = []
        today = date.today()
        count = 0

        for idx, row in earnings_hist.iterrows():
            d = pd.Timestamp(idx).date()
            if d >= today:
                continue  # skip future dates

            estimated = row.get("EPS Estimate")
            actual = row.get("Reported EPS")

            if pd.isna(estimated) or pd.isna(actual):
                continue

            estimated = float(estimated)
            actual = float(actual)
            surprise_pct = (
                round(((actual - estimated) / abs(estimated)) * 100, 2)
                if estimated != 0
                else 0.0
            )

            records.append({
                "date": str(d),
                "estimated_eps": estimated,
                "actual_eps": actual,
                "surprise_pct": surprise_pct,
                "beat": actual > estimated,
            })

            count += 1
            if count >= quarters:
                break

        return records if records else None

    except Exception as e:
        logger.warning(f"fetch_earnings_history failed for {ticker}: {e}")
        return None


def calc_beat_streak(history: Optional[List[Dict[str, Any]]]) -> int:
    """Count consecutive earnings beats starting from most recent.

    Returns 0 if no history or latest was a miss. Negative means
    consecutive misses.
    """
    if not history:
        return 0

    streak = 0
    direction = None

    for record in history:
        beat = record.get("beat", False)
        if direction is None:
            direction = beat
        if beat != direction:
            break
        streak += 1

    return streak if direction else -streak


def earnings_risk_flag(ticker: str) -> Dict[str, Any]:
    """Check if a ticker has upcoming earnings risk.

    Returns
    -------
    Dict with ``has_risk``, ``days_to_earnings``, ``recommendation``.
    """
    dates = fetch_earnings_dates(ticker)
    if dates is None:
        return {
            "has_risk": False,
            "days_to_earnings": None,
            "recommendation": "No earnings data — proceed with caution",
        }

    days = dates.get("days_to_earnings")
    if days is None:
        return {
            "has_risk": False,
            "days_to_earnings": None,
            "recommendation": "Earnings date unknown",
        }

    if days <= 3:
        return {
            "has_risk": True,
            "days_to_earnings": days,
            "recommendation": "AVOID — earnings imminent",
        }
    elif days <= 7:
        return {
            "has_risk": True,
            "days_to_earnings": days,
            "recommendation": "HIGH RISK — earnings within 1 week",
        }
    elif days <= 14:
        return {
            "has_risk": True,
            "days_to_earnings": days,
            "recommendation": "CAUTION — earnings within 2 weeks, reduce size",
        }
    else:
        return {
            "has_risk": False,
            "days_to_earnings": days,
            "recommendation": "OK — earnings are distant",
        }


def batch_earnings_check(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Check earnings risk for a batch of tickers.

    Returns dict mapping ticker → earnings_risk_flag result.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        try:
            results[ticker] = earnings_risk_flag(ticker)
        except Exception as e:
            logger.warning(f"Earnings check failed for {ticker}: {e}")
            results[ticker] = {
                "has_risk": False,
                "days_to_earnings": None,
                "recommendation": f"Error: {e}",
            }
    return results
