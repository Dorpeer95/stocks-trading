"""
Position stagnation detector (Layer 4).

A stagnant position is one that has stopped making progress — it hasn't moved
meaningfully in either direction, consuming a slot that could be used for a
better opportunity.

Stagnation criteria (all must be true)
───────────────────────────────────────
  • Days held ≥ MIN_STAGNANT_DAYS (default 10)
  • Price range over last N days < RANGE_THRESHOLD_PCT (default 3%)
  • Not within BUFFER_DAYS of the time-stop limit

On detection the bot sends an action alert so the trader can decide whether
to exit manually, raise the stop, or add a tighter time stop.
"""

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np

from utils.helpers import safe_float, safe_int

logger = logging.getLogger(__name__)

MIN_STAGNANT_DAYS = int(os.getenv("STOCKS_STAGNANT_MIN_DAYS", "10"))
RANGE_THRESHOLD_PCT = float(os.getenv("STOCKS_STAGNANT_RANGE_PCT", "3.0"))
BUFFER_DAYS = int(os.getenv("STOCKS_STAGNANT_BUFFER_DAYS", "5"))
MAX_HOLD_DAYS = int(os.getenv("STOCKS_MAX_HOLD_DAYS", "42"))


def check_stagnation(position: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate whether a single position is stagnating.

    Parameters
    ----------
    position : Open position row from Supabase.

    Returns
    -------
    Dict with:
      ``stagnant``        — bool
      ``days_flat``       — int (days since last meaningful move)
      ``range_pct``       — float (high-low % range over hold period)
      ``days_to_time_stop`` — int
      ``recommendation``  — "EXIT" | "RAISE_STOP" | "HOLD" | "WATCH"
      ``reason``          — str
    """
    ticker = position.get("ticker", "?")
    days_held = safe_int(position.get("days_held"))
    pnl_pct = safe_float(position.get("unrealized_pnl_pct"))

    # Days to time stop
    days_to_time_stop = max(0, MAX_HOLD_DAYS - days_held)

    # Not enough time held yet
    if days_held < MIN_STAGNANT_DAYS:
        return _not_stagnant(days_held, days_to_time_stop, "too early to judge")

    # Near time stop — let time-stop handle it
    if days_to_time_stop <= BUFFER_DAYS:
        return _not_stagnant(days_held, days_to_time_stop, "near time stop")

    # Estimate price range from water marks
    entry = safe_float(position.get("entry_price"))
    raw_high = position.get("high_water_mark")
    raw_low = position.get("low_water_mark")

    # Skip stagnation check if water marks not yet populated (first intraday tick)
    if raw_high is None or raw_low is None:
        return _not_stagnant(days_held, days_to_time_stop, "water marks not yet set")

    high = safe_float(raw_high, entry)
    low = safe_float(raw_low, entry)

    if entry <= 0:
        return _not_stagnant(days_held, days_to_time_stop, "no entry price")

    # Range = (high - low) / entry * 100
    range_pct = (high - low) / entry * 100 if entry > 0 else 0.0
    stagnant = range_pct < RANGE_THRESHOLD_PCT

    if not stagnant:
        return {
            "stagnant": False,
            "days_flat": days_held,
            "range_pct": round(range_pct, 2),
            "days_to_time_stop": days_to_time_stop,
            "recommendation": "HOLD",
            "reason": f"Range {range_pct:.1f}% — position moving",
        }

    # Stagnant — recommend action based on P&L
    if pnl_pct < 0:
        rec = "EXIT"
        reason = (
            f"Stagnant {days_held}d, range {range_pct:.1f}%, "
            f"P&L {pnl_pct:+.1f}% — cut the loss"
        )
    elif pnl_pct > 1.0:
        rec = "RAISE_STOP"
        reason = (
            f"Stagnant {days_held}d, range {range_pct:.1f}%, "
            f"P&L {pnl_pct:+.1f}% — lock in gains with tighter stop"
        )
    else:
        rec = "WATCH"
        reason = (
            f"Stagnant {days_held}d, range {range_pct:.1f}% — "
            f"near breakeven, monitor"
        )

    logger.info(f"Stagnation detected: {ticker} — {reason}")

    return {
        "stagnant": True,
        "days_flat": days_held,
        "range_pct": round(range_pct, 2),
        "days_to_time_stop": days_to_time_stop,
        "recommendation": rec,
        "reason": reason,
    }


def scan_for_stagnation(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check all positions and return those flagged as stagnant.

    Returns a list of dicts combining position data with stagnation result.
    """
    flagged = []
    for pos in positions:
        result = check_stagnation(pos)
        if result.get("stagnant"):
            flagged.append({
                "ticker": pos.get("ticker"),
                "action": f"REVIEW — {result['recommendation']}",
                "reason": result["reason"],
                "current_price": safe_float(pos.get("current_price")),
                "pnl": safe_float(pos.get("unrealized_pnl")),
                "pnl_pct": safe_float(pos.get("unrealized_pnl_pct")),
                "urgent": "medium" if result["recommendation"] == "EXIT" else "low",
                "stagnation": result,
            })
    return flagged


def _not_stagnant(
    days_held: int,
    days_to_time_stop: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "stagnant": False,
        "days_flat": days_held,
        "range_pct": 0.0,
        "days_to_time_stop": days_to_time_stop,
        "recommendation": "HOLD",
        "reason": reason,
    }
