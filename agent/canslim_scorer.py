"""
CANSLIM methodology scorer.

Evaluates a stock against the C-A-N-S-L criteria using already-fetched
fundamentals (from data_loader.fetch_fundamentals) and scanner data.
Zero additional API calls required.

C — Current quarterly earnings growth > 25%
A — Annual revenue growth > 20% (proxy for sustained earnings)
N — Near new 52-week high (breakout candidate)
S — Small/mid market cap with institutional growth potential
L — Leader: RS percentile > 80
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def score_canslim(
    stock: Dict[str, Any],
    fundamentals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a stock against CANSLIM criteria.

    Parameters
    ----------
    stock : Scanner candidate dict (must have rs_percentile, distance_52w_high).
    fundamentals : Output from ``data_loader.fetch_fundamentals()`` — pre-fetched,
                   so no extra API call needed.

    Returns
    -------
    Dict with:
        ``score``      — 0-100 composite CANSLIM score
        ``components`` — dict of per-criterion ratings
        ``criteria_met`` — count of criteria rated "strong"
    """
    score = 50.0
    components: Dict[str, str] = {}

    if fundamentals is None:
        return {"score": score, "components": components, "criteria_met": 0}

    # ── C: Current quarterly earnings growth ─────────────────────────────────
    qtr_growth = fundamentals.get("earnings_quarterly_growth")
    if qtr_growth is not None:
        if qtr_growth >= 0.25:
            score += 12
            components["C"] = "strong"       # > 25% — CANSLIM ideal
        elif qtr_growth >= 0.10:
            score += 6
            components["C"] = "moderate"
        elif qtr_growth >= 0:
            score += 0
            components["C"] = "weak"
        else:
            score -= 8
            components["C"] = "declining"
    else:
        components["C"] = "unknown"

    # ── A: Annual earnings acceleration (revenue_growth as proxy) ─────────────
    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth >= 0.20:
            score += 12
            components["A"] = "strong"
        elif rev_growth >= 0.10:
            score += 6
            components["A"] = "moderate"
        elif rev_growth >= 0:
            score += 0
            components["A"] = "flat"
        else:
            score -= 6
            components["A"] = "declining"
    else:
        components["A"] = "unknown"

    # ── N: Near new high (price breakout candidate) ───────────────────────────
    dist_high = stock.get("distance_52w_high")
    adx = stock.get("adx")
    if dist_high is not None:
        if dist_high >= -3 and adx and adx > 25:
            score += 15
            components["N"] = "breakout_ready"
        elif dist_high >= -8:
            score += 8
            components["N"] = "near_high"
        elif dist_high >= -20:
            score += 2
            components["N"] = "moderate"
        else:
            score -= 5
            components["N"] = "far_from_high"
    else:
        components["N"] = "unknown"

    # ── S: Supply/demand — smaller float favours bigger moves ────────────────
    market_cap_b = fundamentals.get("market_cap_b")
    if market_cap_b is not None:
        if 0.5 <= market_cap_b <= 20:
            score += 8   # small cap — biggest potential moves
            components["S"] = "small_cap"
        elif 20 < market_cap_b <= 100:
            score += 5   # mid cap — good balance
            components["S"] = "mid_cap"
        elif market_cap_b > 100:
            score += 0   # large cap — moves more slowly
            components["S"] = "large_cap"
        else:
            components["S"] = "micro"
    else:
        components["S"] = "unknown"

    # ── L: Leader — RS percentile ─────────────────────────────────────────────
    rs_pct = stock.get("rs_percentile", 0)
    if rs_pct >= 80:
        score += 15
        components["L"] = "leader"
    elif rs_pct >= 70:
        score += 8
        components["L"] = "above_avg"
    elif rs_pct >= 50:
        score += 0
        components["L"] = "average"
    else:
        score -= 8
        components["L"] = "laggard"

    final_score = round(max(0, min(100, score)), 1)
    criteria_met = sum(1 for v in components.values() if v in ("strong", "breakout_ready", "leader"))

    logger.debug(
        f"CANSLIM: {stock.get('ticker')} score={final_score} "
        f"criteria_met={criteria_met} {components}"
    )

    return {
        "score": final_score,
        "components": components,
        "criteria_met": criteria_met,
    }
