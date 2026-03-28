"""
Trade postmortem recorder and pattern analyzer (Layer 4).

Records every closed trade to ``trade_postmortems`` with outcome classification,
MAE/MFE context, sub-scores at entry, and regime conditions.

Outcome classes
───────────────
  TRUE_POSITIVE    — hit target at ≥+5%
  FALSE_POSITIVE   — stopped out at ≤-3%
  SCRATCH          — closed between -3% and +5% (no conviction)
  REGIME_MISMATCH  — loss in a regime that should have been filtered out

Pattern analysis surfaces the sub-scores that distinguish wins from losses.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from agent.persistence import _table
from agent.regime_engine import get_cached_regime
from utils.helpers import safe_float, safe_int

logger = logging.getLogger(__name__)

# Outcome thresholds
_TP_THRESHOLD = 5.0    # pnl_pct >= this → TRUE_POSITIVE
_FP_THRESHOLD = -3.0   # pnl_pct <= this → FALSE_POSITIVE


def _classify_outcome(
    pnl_pct: float,
    exit_reason: str,
    regime_at_entry: str,
    regime_at_exit: str,
) -> str:
    """Classify trade outcome."""
    if pnl_pct >= _TP_THRESHOLD:
        return "TRUE_POSITIVE"
    if pnl_pct <= _FP_THRESHOLD:
        # Check if regime turned bearish during the hold — different lesson
        bearish_regimes = {"contraction", "inflationary", "crisis", "volatile"}
        if regime_at_exit in bearish_regimes and regime_at_entry not in bearish_regimes:
            return "REGIME_MISMATCH"
        return "FALSE_POSITIVE"
    return "SCRATCH"


def record_postmortem(
    trade: Dict[str, Any],
    position: Dict[str, Any],
    holding_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a postmortem record for a closed trade.

    Called by portfolio._record_trade() immediately after inserting a trade row.

    Parameters
    ----------
    trade       : The trade dict just inserted (from _record_trade).
    position    : The open position row from Supabase.
    holding_data: portfolio_holdings row (has sub_scores, entry_confidence, etc.).
    """
    try:
        entry_price = safe_float(trade.get("entry_price"))
        exit_price = safe_float(trade.get("exit_price"))
        pnl_pct = safe_float(trade.get("realized_pnl_pct"))
        mae_pct = safe_float(trade.get("mae_pct"))
        mfe_pct = safe_float(trade.get("mfe_pct"))
        hold_days = safe_int(trade.get("hold_days"))
        ticker = trade.get("ticker", "")
        exit_reason = trade.get("exit_reason", "")

        # Regime context
        cross = get_cached_regime()
        regime_at_exit = cross.get("regime", "unknown") if cross else "unknown"

        # Try to get entry regime from holding_data or position
        regime_at_entry = "unknown"
        if holding_data:
            regime_at_entry = holding_data.get("regime_at_entry", regime_at_entry)
        if regime_at_entry == "unknown" and cross:
            regime_at_entry = regime_at_exit  # best approximation

        outcome = _classify_outcome(pnl_pct, exit_reason, regime_at_entry, regime_at_exit)

        # Sub scores from holding_data
        sub_scores = None
        entry_confidence = None
        setup_type = None
        if holding_data:
            sub_scores = holding_data.get("sub_scores")
            entry_confidence = safe_float(holding_data.get("entry_confidence"))
            setup_type = holding_data.get("setup_type")

        # Build auto-notes
        notes_parts = []
        if mae_pct > 0:
            notes_parts.append(f"MAE {mae_pct:.1f}%")
        if mfe_pct > 0:
            notes_parts.append(f"MFE {mfe_pct:.1f}%")
        if exit_reason:
            notes_parts.append(exit_reason)
        if cross:
            notes_parts.append(f"Regime@exit: {regime_at_exit}")
        notes = " | ".join(notes_parts)

        record = {
            "ticker": ticker,
            "outcome": outcome,
            "entry_date": trade.get("entry_date"),
            "exit_date": trade.get("exit_date") or date.today().isoformat(),
            "entry_confidence": entry_confidence,
            "sub_scores_at_entry": sub_scores,
            "regime_at_entry": regime_at_entry,
            "regime_at_exit": regime_at_exit,
            "pnl": trade.get("realized_pnl"),
            "pnl_pct": pnl_pct,
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "hold_days": hold_days,
            "setup_type": setup_type,
            "notes": notes,
        }

        _table("trade_postmortems").insert(record).execute()
        logger.info(
            f"Postmortem recorded: {ticker} {outcome} "
            f"pnl={pnl_pct:+.1f}% mae={mae_pct:.1f}% mfe={mfe_pct:.1f}%"
        )

    except Exception as e:
        logger.warning(f"record_postmortem failed for {trade.get('ticker')}: {e}")


# ---------------------------------------------------------------------------
# Pattern analysis helpers
# ---------------------------------------------------------------------------

def get_recent_postmortems(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent postmortems ordered by exit_date desc."""
    try:
        resp = (
            _table("trade_postmortems")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"get_recent_postmortems failed: {e}")
        return []


def analyze_signal_quality(min_sample: int = 10) -> Dict[str, Any]:
    """Identify which sub-scores predict wins vs losses.

    Returns
    -------
    Dict with ``by_outcome`` breakdown, ``weakest_signal``, ``strongest_signal``,
    ``sample_size``.
    """
    postmortems = get_recent_postmortems(limit=200)
    if len(postmortems) < min_sample:
        return {"sample_size": len(postmortems), "insufficient_data": True}

    wins = [p for p in postmortems if p.get("outcome") == "TRUE_POSITIVE"]
    losses = [p for p in postmortems if p.get("outcome") == "FALSE_POSITIVE"]

    def avg_sub_score(records: List[Dict], key: str) -> Optional[float]:
        vals = []
        for r in records:
            ss = r.get("sub_scores_at_entry") or {}
            v = ss.get(key)
            if v is not None:
                vals.append(safe_float(v))
        return round(sum(vals) / len(vals), 1) if vals else None

    signal_keys = ["technical", "relative_strength", "fundamental", "canslim", "sentiment", "macro"]
    win_avgs: Dict[str, Optional[float]] = {}
    loss_avgs: Dict[str, Optional[float]] = {}
    gaps: Dict[str, float] = {}

    for key in signal_keys:
        wa = avg_sub_score(wins, key)
        la = avg_sub_score(losses, key)
        win_avgs[key] = wa
        loss_avgs[key] = la
        if wa is not None and la is not None:
            gaps[key] = round(wa - la, 1)  # positive = wins score higher here

    strongest = max(gaps, key=lambda k: gaps[k]) if gaps else None
    weakest = min(gaps, key=lambda k: gaps[k]) if gaps else None

    return {
        "sample_size": len(postmortems),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_avg_scores": win_avgs,
        "loss_avg_scores": loss_avgs,
        "score_gaps": gaps,
        "strongest_signal": strongest,
        "weakest_signal": weakest,
        "win_rate": round(len(wins) / (len(wins) + len(losses)) * 100, 1) if (wins or losses) else 0,
    }
