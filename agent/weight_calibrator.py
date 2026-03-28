"""
Auto-tuning signal weight calibrator (Layer 4).

Analyses closed trade postmortems to identify which scoring sub-signals
have the strongest predictive power, then persists calibrated weights to
``calibrated_weights`` for use by scorer.py.

How it works
────────────
1. Fetch recent postmortems (min 20 samples required).
2. For each signal dimension (technical, rs, fundamental, canslim, sentiment,
   macro, insider), compute the mean sub-score for wins vs losses.
3. A signal's weight is proportional to its win/loss gap → signals that
   strongly separate wins from losses get higher weight.
4. Weights are normalised to sum to 1.0, clipped to [0.05, 0.40].
5. The new weights are stored in ``calibrated_weights`` and returned.

The scorer reads calibrated weights (if any) at the start of each weekly
scan when ``STOCKS_USE_CALIBRATED_WEIGHTS=true``.
"""

import logging
import os
from datetime import date
from typing import Any, Dict, Optional

from agent.persistence import _table
from agent.postmortem import analyze_signal_quality
from utils.helpers import safe_float

logger = logging.getLogger(__name__)

USE_CALIBRATED = os.getenv("STOCKS_USE_CALIBRATED_WEIGHTS", "false").lower() == "true"

# Hard floors and ceilings per weight
_MIN_WEIGHT = 0.05
_MAX_WEIGHT = 0.40

# Default starting weights (baseline if no calibration yet)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical":   0.30,
    "rs":          0.20,
    "fundamental": 0.15,
    "canslim":     0.10,
    "sentiment":   0.08,
    "macro":       0.07,
    "insider":     0.05,
    "ml":          0.05,
}


def get_active_weights() -> Dict[str, float]:
    """Return the most recently calibrated weights, or defaults.

    Reads ``calibrated_weights`` table sorted by calibration_date desc.
    Falls back to ``_DEFAULT_WEIGHTS`` on any error or if table is empty.
    """
    if not USE_CALIBRATED:
        return _DEFAULT_WEIGHTS.copy()

    try:
        resp = (
            _table("calibrated_weights")
            .select("weights")
            .eq("is_active", True)
            .order("calibration_date", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if rows and rows[0].get("weights"):
            stored = rows[0]["weights"]
            # Ensure all expected keys are present
            merged = _DEFAULT_WEIGHTS.copy()
            merged.update({k: safe_float(v, _DEFAULT_WEIGHTS.get(k, 0.05)) for k, v in stored.items()})
            return merged
    except Exception as e:
        logger.warning(f"get_active_weights DB read failed: {e}")

    return _DEFAULT_WEIGHTS.copy()


def calibrate_weights(min_sample: int = 20) -> Optional[Dict[str, Any]]:
    """Run the calibration and persist new weights.

    Parameters
    ----------
    min_sample : Minimum postmortems needed to proceed.

    Returns
    -------
    The calibration result dict, or None if insufficient data.
    """
    analysis = analyze_signal_quality(min_sample=min_sample)

    if analysis.get("insufficient_data"):
        logger.info(
            f"Weight calibration skipped: only {analysis.get('sample_size', 0)} "
            f"samples (need {min_sample})"
        )
        return None

    gaps = analysis.get("score_gaps", {})
    if not gaps:
        logger.warning("No score gaps available for calibration")
        return None

    # Convert gaps to positive weights (floor at _MIN_WEIGHT for negative gaps)
    raw: Dict[str, float] = {}
    for key in _DEFAULT_WEIGHTS:
        gap = gaps.get(key, 0.0)
        # Shift all values positive; add default weight as base to avoid zeroing
        raw[key] = max(_MIN_WEIGHT, _DEFAULT_WEIGHTS[key] + gap * 0.005)

    # Clip to [min, max]
    clipped = {k: max(_MIN_WEIGHT, min(_MAX_WEIGHT, v)) for k, v in raw.items()}

    # Normalise to sum = 1.0
    total = sum(clipped.values())
    normalised: Dict[str, float] = {
        k: round(v / total, 4) for k, v in clipped.items()
    } if total > 0 else _DEFAULT_WEIGHTS.copy()

    # Persist — first deactivate old active record
    try:
        _table("calibrated_weights").update({"is_active": False}).eq("is_active", True).execute()
    except Exception as e:
        logger.warning(f"Failed to deactivate old weights: {e}")

    record = {
        "weights": normalised,
        "sample_size": analysis["sample_size"],
        "win_rate": analysis.get("win_rate"),
        "avg_pnl": None,  # could add if needed
        "calibration_date": date.today().isoformat(),
        "notes": (
            f"auto-calibrated | strongest={analysis.get('strongest_signal')} "
            f"weakest={analysis.get('weakest_signal')}"
        ),
        "is_active": True,
    }

    try:
        _table("calibrated_weights").insert(record).execute()
        logger.info(
            f"Weights calibrated: {normalised} "
            f"(sample={analysis['sample_size']} win_rate={analysis.get('win_rate')}%)"
        )
    except Exception as e:
        logger.error(f"Failed to persist calibrated weights: {e}")

    return {
        "weights": normalised,
        "analysis": analysis,
    }
