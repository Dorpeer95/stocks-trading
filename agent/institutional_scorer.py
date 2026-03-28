"""
Institutional flow scorer.

Uses yfinance ``heldPercentInstitutions`` (already pulled in
fetch_fundamentals) to assess whether a stock is in the institutional
sweet spot — enough coverage for liquidity, room left for more buying.

Adds a score_adjustment to the insider sub-score:
  +8  : 20-60% institutional ownership (prime zone: room to grow)
  +4  : 60-75% (well-covered, limited additional inflow headroom)
  +0  : >75% (fully owned — incremental institutional demand exhausted)
  -5  : <10% (either undiscovered gem or actively avoided — risky)
  +3  : insider ownership > 10% (skin in the game bonus)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_institutional_score(
    ticker: str,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assess institutional ownership quality.

    Parameters
    ----------
    ticker : Stock ticker (used for logging).
    fundamentals : Output from ``data_loader.fetch_fundamentals()`` — must
                   contain ``held_pct_institutions`` and ``held_pct_insiders``.

    Returns
    -------
    Dict with:
        ``score_adjustment`` — int to add to insider sub-score (-8 to +10)
        ``institutional_pct`` — float (0-100) or None
        ``insider_pct``       — float (0-100) or None
        ``signal``            — descriptive signal string
    """
    if fundamentals is None:
        return _unknown(ticker)

    raw_inst = fundamentals.get("held_pct_institutions")
    raw_ins  = fundamentals.get("held_pct_insiders")

    if raw_inst is None:
        return _unknown(ticker)

    # yfinance returns fractions (0.0–1.0) or percentages — normalise to 0-100
    inst_pct = raw_inst * 100 if raw_inst <= 1.0 else float(raw_inst)
    ins_pct  = (raw_ins * 100 if raw_ins <= 1.0 else float(raw_ins)) if raw_ins is not None else None

    # ── Institutional ownership band ─────────────────────────────────────────
    if inst_pct >= 75:
        adj = 0
        signal = "fully_owned"          # maxed out — no more institutional buying headroom
    elif inst_pct >= 60:
        adj = 4
        signal = "well_covered"
    elif inst_pct >= 20:
        adj = 8
        signal = "prime_zone"           # sweet spot: established but room to grow
    elif inst_pct >= 5:
        adj = -2
        signal = "under_owned"
    else:
        adj = -5
        signal = "avoided"              # institutions are passing — red flag

    # ── Insider ownership bonus (skin in the game) ───────────────────────────
    if ins_pct is not None and ins_pct >= 10:
        adj += 3
        signal += "+insider_aligned"

    logger.debug(
        f"Institutional: {ticker} inst={inst_pct:.1f}% "
        f"ins={ins_pct:.1f}% adj={adj:+d} signal={signal}"
    )

    return {
        "score_adjustment": adj,
        "institutional_pct": round(inst_pct, 1),
        "insider_pct": round(ins_pct, 1) if ins_pct is not None else None,
        "signal": signal,
    }


def _unknown(ticker: str) -> Dict[str, Any]:
    logger.debug(f"Institutional: no data for {ticker}")
    return {
        "score_adjustment": 0,
        "institutional_pct": None,
        "insider_pct": None,
        "signal": "unknown",
    }
