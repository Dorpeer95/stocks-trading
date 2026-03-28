"""
Enhanced earnings intelligence.

Extends utils/earnings.py with:
  1. More aggressive confidence penalties for imminent earnings
     (< 7 days → reduce confidence 15 pts; already was -10 in score_earnings_risk)
  2. PEAD (Post-Earnings Announcement Drift) detection:
     If a stock gapped up > 3% on volume > 2× average after its last earnings,
     it is fast-tracked — set ``pead_detected=True`` so portfolio_manager
     can skip the normal 2-week consecutive-weeks requirement.

All data comes from yfinance (already cached) — zero FMP API calls.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, Optional

from utils.earnings import earnings_risk_flag, fetch_earnings_history, calc_beat_streak

logger = logging.getLogger(__name__)

# PEAD thresholds
PEAD_GAP_MIN_PCT = 3.0      # stock must have gapped up ≥ 3% on earnings day
PEAD_VOL_MULT    = 1.8      # volume must be ≥ 1.8× the prior 20-day average


def enhanced_earnings_score(
    ticker: str,
    stock: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute earnings-aware score with PEAD detection.

    Parameters
    ----------
    ticker : Stock ticker.
    stock  : Scanner candidate dict (used for PEAD volume check if provided).

    Returns
    -------
    Dict with:
        ``score``          — 0-100 earnings safety score
        ``pead_detected``  — bool (True if PEAD fast-track qualifies)
        ``beat_streak``    — int (consecutive earnings beats)
        ``days_to_earnings`` — int or None
        ``has_risk``       — bool
    """
    result: Dict[str, Any] = {
        "score": 60.0,
        "pead_detected": False,
        "beat_streak": 0,
        "days_to_earnings": None,
        "has_risk": False,
    }

    try:
        risk = earnings_risk_flag(ticker)
        days = risk.get("days_to_earnings")
        has_risk = risk.get("has_risk", False)

        result["has_risk"] = has_risk
        result["days_to_earnings"] = days

        # ── Earnings proximity penalty (more aggressive than base scorer) ────
        if days is None:
            result["score"] = 60.0
        elif days <= 3:
            result["score"] = 5.0    # imminent — nearly always skip
        elif days <= 7:
            result["score"] = 20.0   # within 1 week — aggressive penalty
        elif days <= 14:
            result["score"] = 45.0   # within 2 weeks — moderate penalty
        elif days <= 30:
            result["score"] = 70.0   # within 1 month — mild caution
        else:
            result["score"] = 85.0   # far out — earnings safety is a positive

        # ── Beat streak bonus ─────────────────────────────────────────────────
        # Consistent beaters are safer to hold through earnings gaps
        try:
            history = fetch_earnings_history(ticker, quarters=6)
            streak = calc_beat_streak(history)
            result["beat_streak"] = streak
            if streak >= 4:
                result["score"] = min(100, result["score"] + 10)
            elif streak >= 2:
                result["score"] = min(100, result["score"] + 5)
            elif streak <= -2:
                result["score"] = max(0, result["score"] - 10)
        except Exception:
            pass

        # ── PEAD detection ────────────────────────────────────────────────────
        # If the stock recently gapped up strongly after earnings, it may be
        # in a PEAD drift — a confirmed buying opportunity.
        result["pead_detected"] = _detect_pead(ticker, stock)

    except Exception as e:
        logger.debug(f"enhanced_earnings_score failed for {ticker}: {e}")

    return result


def _detect_pead(
    ticker: str,
    stock: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if stock shows post-earnings announcement drift pattern.

    Criteria:
    - Has price history in scanner data (stock dict with recent OHLCV)
    - OR can fetch from yfinance
    - Last earnings was within 21 days
    - On the earnings day (or day after), stock gapped up ≥ PEAD_GAP_MIN_PCT
    - Volume on that day was ≥ PEAD_VOL_MULT × 20-day average
    """
    try:
        from utils.data_loader import fetch_price_data
        from utils.earnings import fetch_earnings_dates

        # Check if earnings were recent (within 21 days)
        dates = fetch_earnings_dates(ticker)
        if not dates:
            return False

        prev_earnings_str = dates.get("previous_earnings_date")
        if not prev_earnings_str:
            return False

        try:
            from datetime import datetime
            prev_date = datetime.fromisoformat(prev_earnings_str).date()
        except Exception:
            return False

        days_since = (date.today() - prev_date).days
        if days_since > 21 or days_since < 0:
            return False

        # Fetch recent price data
        df = fetch_price_data(ticker, period="30d", interval="1d")
        if df is None or df.empty or len(df) < 5:
            return False

        # Find the earnings day row (or next trading day)
        earnings_idx = None
        for i, idx in enumerate(df.index):
            row_date = idx.date() if hasattr(idx, "date") else idx
            if row_date >= prev_date:
                earnings_idx = i
                break

        if earnings_idx is None or earnings_idx == 0:
            return False

        # Gap up: today's open vs prior close
        prior_close = float(df["Close"].iloc[earnings_idx - 1])
        earn_open   = float(df["Open"].iloc[earnings_idx])
        earn_volume = float(df["Volume"].iloc[earnings_idx])

        if prior_close <= 0:
            return False

        gap_pct = (earn_open - prior_close) / prior_close * 100

        # Average volume over previous 20 days
        vol_window = df["Volume"].iloc[max(0, earnings_idx - 20): earnings_idx]
        avg_vol = vol_window.mean() if len(vol_window) > 0 else 0

        if avg_vol <= 0:
            return False

        vol_ratio = earn_volume / avg_vol

        pead = gap_pct >= PEAD_GAP_MIN_PCT and vol_ratio >= PEAD_VOL_MULT

        if pead:
            logger.info(
                f"PEAD detected: {ticker} — gap {gap_pct:.1f}% on "
                f"{vol_ratio:.1f}× volume after earnings {prev_earnings_str}"
            )

        return pead

    except Exception as e:
        logger.debug(f"PEAD detection failed for {ticker}: {e}")
        return False
