"""
Macro event detection and market regime analysis.

Monitors VIX, oil, dollar, treasury yields, gold, and SPY for
significant moves that affect swing trading decisions.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.persistence import insert_event, get_recent_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds for "significant" moves
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, Dict[str, float]] = {
    "vix": {
        "spike": 20.0,       # absolute level
        "surge_pct": 15.0,   # single-day % increase
        "extreme": 30.0,     # extreme fear
    },
    "oil": {
        "big_move_pct": 5.0,
    },
    "gold": {
        "big_move_pct": 3.0,
    },
    "dollar": {
        "big_move_pct": 1.5,
    },
    "treasury_10y": {
        "high_yield": 5.0,       # absolute yield %
        "big_move_pct": 5.0,     # % change in yield
    },
    "spy": {
        "big_drop_pct": -2.0,
        "big_rally_pct": 2.0,
    },
}

# How macro events correlate with stock signals
CORRELATION_MAP: Dict[str, str] = {
    "vix_spike": "negative",         # high VIX → risk-off
    "vix_extreme": "very_negative",  # extreme VIX → danger
    "oil_surge": "mixed",            # hurts consumers, helps energy
    "oil_crash": "mixed",
    "gold_surge": "negative",        # flight to safety
    "dollar_strong": "negative",     # hurts multinationals
    "dollar_weak": "positive",       # helps exporters
    "treasury_surge": "negative",    # rising rates hurt growth
    "spy_crash": "very_negative",
    "spy_rally": "positive",
    "vix_calm": "positive",          # low VIX → complacent/bullish
}

# Severity mapping for persistence
SEVERITY_MAP: Dict[str, str] = {
    "very_negative": "high",
    "negative": "medium",
    "positive": "low",
    "very_positive": "low",
    "mixed": "medium",
}


# ---------------------------------------------------------------------------
# Event detection from macro data
# ---------------------------------------------------------------------------

def detect_events(macro_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Scan macro data for significant events.

    Parameters
    ----------
    macro_data : Output from ``data_loader.fetch_macro_data()``.
        Each key (vix, oil, gold, etc.) maps to a dict with
        ``current``, ``prev_close``, ``change_pct``.

    Returns
    -------
    List of event dicts ready for ``persistence.insert_event()``:
    ``event_type``, ``severity``, ``description``, ``affected_sectors``,
    ``event_date``, ``raw_data``.
    """
    events: List[Dict[str, Any]] = []
    today = date.today().isoformat()

    # --- VIX ---
    vix = macro_data.get("vix")
    if vix:
        level = vix["current"]
        change = vix.get("change_pct", 0)

        if level >= THRESHOLDS["vix"]["extreme"]:
            events.append(_make_event(
                "vix_extreme",
                "high",
                f"VIX at extreme level: {level:.1f} ({change:+.1f}%)",
                ["all"],
                today,
                vix,
            ))
        elif level >= THRESHOLDS["vix"]["spike"]:
            events.append(_make_event(
                "vix_spike",
                "medium",
                f"VIX elevated: {level:.1f} ({change:+.1f}%)",
                ["all"],
                today,
                vix,
            ))
        elif change >= THRESHOLDS["vix"]["surge_pct"]:
            events.append(_make_event(
                "vix_surge",
                "medium",
                f"VIX surged {change:+.1f}% to {level:.1f}",
                ["all"],
                today,
                vix,
            ))
        elif level < 15:
            events.append(_make_event(
                "vix_calm",
                "low",
                f"VIX calm at {level:.1f}",
                ["all"],
                today,
                vix,
            ))

    # --- Oil ---
    oil = macro_data.get("oil")
    if oil:
        change = oil.get("change_pct", 0)
        if abs(change) >= THRESHOLDS["oil"]["big_move_pct"]:
            direction = "surge" if change > 0 else "crash"
            events.append(_make_event(
                f"oil_{direction}",
                "medium",
                f"Oil {direction}: {change:+.1f}% to ${oil['current']:.2f}",
                ["Energy", "Industrials", "Consumer Discretionary"],
                today,
                oil,
            ))

    # --- Gold ---
    gold = macro_data.get("gold")
    if gold:
        change = gold.get("change_pct", 0)
        if abs(change) >= THRESHOLDS["gold"]["big_move_pct"]:
            direction = "surge" if change > 0 else "drop"
            events.append(_make_event(
                f"gold_{direction}",
                "medium" if change > 0 else "low",
                f"Gold {direction}: {change:+.1f}% to ${gold['current']:.2f}",
                ["Materials"],
                today,
                gold,
            ))

    # --- Dollar ---
    dollar = macro_data.get("dollar")
    if dollar:
        change = dollar.get("change_pct", 0)
        if abs(change) >= THRESHOLDS["dollar"]["big_move_pct"]:
            direction = "strong" if change > 0 else "weak"
            events.append(_make_event(
                f"dollar_{direction}",
                "medium",
                f"Dollar {direction}: {change:+.1f}%",
                ["Technology", "Industrials", "Consumer Staples"],
                today,
                dollar,
            ))

    # --- Treasury 10Y ---
    treasury = macro_data.get("treasury_10y")
    if treasury:
        level = treasury["current"]
        change = treasury.get("change_pct", 0)

        if level >= THRESHOLDS["treasury_10y"]["high_yield"]:
            events.append(_make_event(
                "treasury_high",
                "high",
                f"10Y yield at {level:.2f}%",
                ["Real Estate", "Utilities", "Financials"],
                today,
                treasury,
            ))
        elif abs(change) >= THRESHOLDS["treasury_10y"]["big_move_pct"]:
            direction = "surge" if change > 0 else "drop"
            events.append(_make_event(
                f"treasury_{direction}",
                "medium",
                f"10Y yield {direction}: {change:+.1f}% to {level:.2f}%",
                ["Real Estate", "Utilities", "Financials", "Technology"],
                today,
                treasury,
            ))

    # --- SPY ---
    spy = macro_data.get("spy")
    if spy:
        change = spy.get("change_pct", 0)
        if change <= THRESHOLDS["spy"]["big_drop_pct"]:
            events.append(_make_event(
                "spy_crash",
                "high",
                f"SPY dropped {change:.1f}%",
                ["all"],
                today,
                spy,
            ))
        elif change >= THRESHOLDS["spy"]["big_rally_pct"]:
            events.append(_make_event(
                "spy_rally",
                "low",
                f"SPY rallied +{change:.1f}%",
                ["all"],
                today,
                spy,
            ))

    logger.info(f"Detected {len(events)} macro events")
    return events


def _make_event(
    event_type: str,
    severity: str,
    description: str,
    affected_sectors: List[str],
    event_date: str,
    raw_data: Dict[str, float],
) -> Dict[str, Any]:
    """Create an event dict for persistence."""
    return {
        "event_type": event_type,
        "severity": severity,
        "description": description,
        "affected_sectors": affected_sectors,
        "event_date": event_date,
        "raw_data": raw_data,
        "correlation": CORRELATION_MAP.get(event_type, "neutral"),
    }


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def assess_market_regime(
    macro_data: Dict[str, Dict[str, float]],
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assess the current market regime / mood.

    Returns
    -------
    Dict with ``regime``, ``mood``, ``risk_level``,
    ``position_size_modifier``, ``description``.
    """
    vix = macro_data.get("vix", {}).get("current", 15)
    spy_change = macro_data.get("spy", {}).get("change_pct", 0)

    # Count recent high-severity events
    high_events = 0
    if events:
        high_events = sum(1 for e in events if e.get("severity") == "high")

    # Regime classification
    if vix >= 30 or high_events >= 2:
        regime = "crisis"
        mood = "Bearish"
        risk_level = "high"
        size_mod = 0.25   # quarter size
        desc = f"Crisis mode — VIX={vix:.0f}, {high_events} severe events"
    elif vix >= 20 or high_events >= 1:
        regime = "volatile"
        mood = "Bearish"
        risk_level = "elevated"
        size_mod = 0.5    # half size
        desc = f"Elevated volatility — VIX={vix:.0f}"
    elif vix >= 15:
        regime = "normal"
        mood = "Neutral"
        risk_level = "normal"
        size_mod = 1.0    # full size
        desc = f"Normal conditions — VIX={vix:.0f}"
    else:
        regime = "calm"
        mood = "Bullish"
        risk_level = "low"
        size_mod = 1.0    # full size (could argue for more)
        desc = f"Low volatility — VIX={vix:.0f}"

    # Adjust for SPY trend
    if spy_change <= -3:
        size_mod *= 0.5
        desc += f" | SPY crashed {spy_change:.1f}%"
    elif spy_change >= 3:
        desc += f" | SPY rallied +{spy_change:.1f}%"

    return {
        "regime": regime,
        "mood": mood,
        "risk_level": risk_level,
        "position_size_modifier": round(size_mod, 2),
        "vix": vix,
        "spy_change_pct": spy_change,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Persist detected events
# ---------------------------------------------------------------------------

def detect_and_persist_events(
    macro_data: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Detect macro events, persist them, and assess regime.

    Returns
    -------
    (events, regime) tuple.
    """
    events = detect_events(macro_data)

    # Persist each event
    for event in events:
        try:
            insert_event(event)
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")

    # Assess regime considering recent events from DB too
    recent_db_events = []
    try:
        recent_db_events = get_recent_events(days=3)
    except Exception:
        pass

    all_recent = events + recent_db_events
    regime = assess_market_regime(macro_data, all_recent)

    return events, regime
