"""
Portfolio Manager — persistent model portfolio with state machine.

Transforms the bot from a weekly "fresh picks" screener into a proper
managed portfolio that holds stocks until signals break, not until the
next scan runs.

State machine per holding:
    NEW CANDIDATE
        → (confidence ≥ ENTRY_THRESHOLD for CONSECUTIVE_NEEDED weeks)
    ACTIVE  — in portfolio, signal strong
        → (confidence drops below STAY_THRESHOLD for 1 week)
    WATCH   — in portfolio, signal weakening; monitor closely
        → (confidence recovers above STAY_THRESHOLD)  → back to ACTIVE
        → (confidence stays below STAY_THRESHOLD 2nd week)
    EXITING — signal confirmed broken; send SELL alert
        → removed from portfolio after alert

Displacement:
    If portfolio is full AND a new candidate scores DISPLACEMENT_GAP
    points above the weakest active/watch holding, flag it as a
    DISPLACEMENT_CANDIDATE for the user to consider.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from agent.regime_engine import get_cached_regime
from agent.persistence import (
    get_portfolio_holdings,
    get_portfolio_holding,
    upsert_portfolio_holding,
    remove_portfolio_holding,
    log_portfolio_action,
    get_consecutive_strong_weeks,
)
from utils.helpers import safe_float

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (env-overridable via agent.py if needed)
# ---------------------------------------------------------------------------
ENTRY_THRESHOLD = 72        # confidence needed to be a candidate
STAY_THRESHOLD = 58         # once in portfolio, stay until below this...
EXIT_WEAK_WEEKS = 2         # ...for this many consecutive weeks
CONSECUTIVE_NEEDED = 2      # must hit ENTRY_THRESHOLD N weeks in a row to enter
DISPLACEMENT_GAP = 10       # new stock needs this many points above weakest hold
MAX_SLOTS = 8               # maximum portfolio size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_portfolio_cycle(
    scored_candidates: List[Dict[str, Any]],
    scan_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Main weekly entry point. Call after score_candidates().

    Parameters
    ----------
    scored_candidates : Output of ``scorer.score_candidates()`` — already
        filtered to ≥ MIN_CONFIDENCE, sorted by confidence descending.
    scan_date : ISO date string for this scan (default: today).

    Returns
    -------
    Portfolio diff dict with keys:
        holdings       — all current active/watch holdings (updated)
        new_entries    — tickers added to portfolio this week
        exits          — tickers removed this week (SELL signal)
        watch_flags    — tickers newly flagged as weakening
        watch_cleared  — tickers recovered from watch back to active
        displacements  — (new_ticker, weak_ticker) swap suggestions
        open_slots     — how many empty portfolio slots remain
    """
    scan_date = scan_date or date.today().isoformat()
    scored_map = {c["ticker"]: c for c in scored_candidates if c.get("ticker")}

    # ── Step 1: update all existing holdings with fresh scores ─────────────
    current_holdings = get_portfolio_holdings()
    exits, watch_flags, watch_cleared = _update_existing_holdings(
        current_holdings, scored_map, scan_date
    )

    # ── Step 2: reload after updates ──────────────────────────────────────
    current_holdings = get_portfolio_holdings()
    active_tickers = {h["ticker"] for h in current_holdings}
    active_count = len([h for h in current_holdings if h["status"] in ("active", "watch")])
    open_slots = MAX_SLOTS - active_count

    # ── Step 3: find new entries ───────────────────────────────────────────
    new_entries = []
    if open_slots > 0:
        new_entries = _find_entries(scored_map, active_tickers, open_slots, scan_date)

    # ── Step 4: find displacement candidates ──────────────────────────────
    displacements = []
    if open_slots == 0:
        displacements = _find_displacements(scored_map, current_holdings)

    # ── Step 5: reload final state ─────────────────────────────────────────
    final_holdings = get_portfolio_holdings()
    active_count_final = len([h for h in final_holdings if h["status"] in ("active", "watch")])

    logger.info(
        f"Portfolio cycle complete: {active_count_final} holdings | "
        f"{len(new_entries)} new | {len(exits)} exits | "
        f"{len(watch_flags)} watch | {len(displacements)} displace"
    )

    return {
        "holdings": final_holdings,
        "new_entries": new_entries,
        "exits": exits,
        "watch_flags": watch_flags,
        "watch_cleared": watch_cleared,
        "displacements": displacements,
        "open_slots": MAX_SLOTS - active_count_final,
    }


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _update_existing_holdings(
    holdings: List[Dict[str, Any]],
    scored_map: Dict[str, Any],
    scan_date: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Re-score existing holdings and apply state transitions.

    Returns (exits, watch_flags, watch_cleared).
    """
    exits: List[str] = []
    watch_flags: List[str] = []
    watch_cleared: List[str] = []

    for holding in holdings:
        ticker = holding["ticker"]
        prev_status = holding["status"]
        prev_conf = holding.get("current_confidence", 0) or 0

        # Get fresh confidence — use scored_map if available, else keep prev
        if ticker in scored_map:
            new_conf = safe_float(scored_map[ticker].get("confidence"))
            sub_scores = scored_map[ticker].get("sub_scores", {})
            setup_type = scored_map[ticker].get("setup_type", "")
            gpt_vetoed = scored_map[ticker].get("gpt_vetoed", False)
        else:
            # Stock didn't pass scanner this week — treat as weakened signal
            new_conf = max(0, prev_conf - 8)  # decay when absent from scan
            sub_scores = holding.get("sub_scores", {})
            setup_type = holding.get("setup_type", "")
            gpt_vetoed = False

        weeks_held = (holding.get("weeks_held") or 0) + 1

        # ── GPT hard veto — immediate exit regardless of status ────────────
        if gpt_vetoed:
            _do_exit(holding, ticker, new_conf, "GPT risk flag veto", scan_date)
            exits.append(ticker)
            log_portfolio_action(
                ticker, "GPT_VETO",
                reason=scored_map.get(ticker, {}).get("gpt_veto_reason", "GPT veto"),
                confidence=new_conf,
                prev_status=prev_status, new_status="exiting",
            )
            continue

        # ── State transitions ──────────────────────────────────────────────
        if prev_status == "active":
            if new_conf < STAY_THRESHOLD:
                # First weak week → move to watch
                updates = {
                    "ticker": ticker,
                    "status": "watch",
                    "prev_confidence": prev_conf,
                    "current_confidence": new_conf,
                    "consecutive_weak_weeks": 1,
                    "weeks_held": weeks_held,
                    "sub_scores": sub_scores,
                    "setup_type": setup_type,
                    "last_scored_at": scan_date,
                }
                upsert_portfolio_holding(updates)
                watch_flags.append(ticker)
                log_portfolio_action(
                    ticker, "WATCH_FLAG",
                    reason=f"Confidence dropped {prev_conf:.0f}→{new_conf:.0f} (below {STAY_THRESHOLD})",
                    confidence=new_conf,
                    prev_status="active", new_status="watch",
                )
            else:
                # Still strong — confirm hold
                updates = {
                    "ticker": ticker,
                    "status": "active",
                    "prev_confidence": prev_conf,
                    "current_confidence": new_conf,
                    "consecutive_weak_weeks": 0,
                    "weeks_held": weeks_held,
                    "sub_scores": sub_scores,
                    "setup_type": setup_type,
                    "last_scored_at": scan_date,
                }
                upsert_portfolio_holding(updates)
                log_portfolio_action(
                    ticker, "HOLD_CONFIRMED",
                    reason=f"Confidence {new_conf:.0f} above stay threshold",
                    confidence=new_conf,
                    prev_status="active", new_status="active",
                )

        elif prev_status == "watch":
            weak_weeks = (holding.get("consecutive_weak_weeks") or 1) + 1

            if new_conf >= STAY_THRESHOLD:
                # Signal recovered — back to active
                updates = {
                    "ticker": ticker,
                    "status": "active",
                    "prev_confidence": prev_conf,
                    "current_confidence": new_conf,
                    "consecutive_weak_weeks": 0,
                    "weeks_held": weeks_held,
                    "sub_scores": sub_scores,
                    "setup_type": setup_type,
                    "last_scored_at": scan_date,
                }
                upsert_portfolio_holding(updates)
                watch_cleared.append(ticker)
                log_portfolio_action(
                    ticker, "WATCH_CLEARED",
                    reason=f"Signal recovered {prev_conf:.0f}→{new_conf:.0f}",
                    confidence=new_conf,
                    prev_status="watch", new_status="active",
                )

            elif weak_weeks >= EXIT_WEAK_WEEKS:
                # Second weak week — exit
                _do_exit(holding, ticker, new_conf,
                         f"Signal weak {weak_weeks} consecutive weeks "
                         f"(conf {new_conf:.0f} < {STAY_THRESHOLD})",
                         scan_date)
                exits.append(ticker)
                log_portfolio_action(
                    ticker, "REMOVED",
                    reason=f"Exit: {weak_weeks} consecutive weak weeks, conf={new_conf:.0f}",
                    confidence=new_conf,
                    prev_status="watch", new_status="exiting",
                )

            else:
                # Still watching
                updates = {
                    "ticker": ticker,
                    "status": "watch",
                    "prev_confidence": prev_conf,
                    "current_confidence": new_conf,
                    "consecutive_weak_weeks": weak_weeks,
                    "weeks_held": weeks_held,
                    "sub_scores": sub_scores,
                    "setup_type": setup_type,
                    "last_scored_at": scan_date,
                }
                upsert_portfolio_holding(updates)

    return exits, watch_flags, watch_cleared


def _find_entries(
    scored_map: Dict[str, Any],
    active_tickers: set,
    open_slots: int,
    scan_date: str,
) -> List[Dict[str, Any]]:
    """Find candidates ready to enter the portfolio.

    A candidate enters when it has scored ≥ ENTRY_THRESHOLD for
    CONSECUTIVE_NEEDED consecutive weeks.
    """
    new_entries: List[Dict[str, Any]] = []

    # Sort by confidence descending, skip already-held tickers
    ranked = sorted(
        [c for c in scored_map.values() if c["ticker"] not in active_tickers],
        key=lambda x: x.get("confidence", 0),
        reverse=True,
    )

    for candidate in ranked:
        if len(new_entries) >= open_slots:
            break

        ticker = candidate["ticker"]
        confidence = safe_float(candidate.get("confidence"))

        if confidence < ENTRY_THRESHOLD:
            continue  # already sorted, no need to check lower entries

        # Check consecutive strong weeks in signal_history
        consec = get_consecutive_strong_weeks(ticker, ENTRY_THRESHOLD)

        if consec < CONSECUTIVE_NEEDED:
            # First strong week — update their consecutive count in signal history
            # (signal_history is written by agent.py before this runs)
            logger.debug(
                f"{ticker}: conf {confidence:.0f} — week {consec}/{CONSECUTIVE_NEEDED} "
                f"(need {CONSECUTIVE_NEEDED} consecutive)"
            )
            continue

        # ✅ Qualifies — add to portfolio
        _regime = get_cached_regime()
        holding = {
            "ticker": ticker,
            "status": "active",
            "entry_confidence": confidence,
            "current_confidence": confidence,
            "prev_confidence": None,
            "consecutive_strong_weeks": consec,
            "consecutive_weak_weeks": 0,
            "weeks_held": 0,
            "added_at": scan_date,
            "last_scored_at": scan_date,
            "sector": candidate.get("sector"),
            "regime_at_entry": _regime.get("regime", "unknown") if _regime else "unknown",
            "entry_price": candidate.get("entry_price_low") or candidate.get("close"),
            "stop_loss": candidate.get("stop_loss"),
            "target_price": candidate.get("target_price"),
            "sub_scores": candidate.get("sub_scores", {}),
            "setup_type": candidate.get("setup_type", ""),
            "gpt_risk_flag": candidate.get("gpt_vetoed", False),
            "notes": "; ".join(candidate.get("reasons", [])[:3]),
        }
        upsert_portfolio_holding(holding)
        new_entries.append(candidate)
        log_portfolio_action(
            ticker, "ADDED",
            reason=f"{consec} consecutive strong weeks, conf={confidence:.0f}",
            confidence=confidence,
            prev_status=None, new_status="active",
        )
        logger.info(
            f"Portfolio ENTRY: {ticker} conf={confidence:.0f} "
            f"({consec} consecutive strong weeks)"
        )

    return new_entries


def _find_displacements(
    scored_map: Dict[str, Any],
    current_holdings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest swaps: new candidate with much higher confidence than weakest hold.

    Returns list of dicts: {new_ticker, new_conf, weak_ticker, weak_conf, gap}.
    """
    if not current_holdings:
        return []

    active_tickers = {h["ticker"] for h in current_holdings}

    # Find weakest current holding
    weakest = min(
        current_holdings,
        key=lambda h: h.get("current_confidence") or 0,
    )
    weak_conf = safe_float(weakest.get("current_confidence"))

    displacements = []
    for candidate in scored_map.values():
        ticker = candidate["ticker"]
        if ticker in active_tickers:
            continue
        new_conf = safe_float(candidate.get("confidence"))
        gap = new_conf - weak_conf
        if gap >= DISPLACEMENT_GAP:
            displacements.append({
                "new_ticker": ticker,
                "new_conf": new_conf,
                "new_setup": candidate.get("setup_type", ""),
                "weak_ticker": weakest["ticker"],
                "weak_conf": weak_conf,
                "gap": round(gap, 1),
            })
            log_portfolio_action(
                ticker, "DISPLACEMENT_CANDIDATE",
                reason=f"Conf {new_conf:.0f} vs weakest hold {weakest['ticker']} "
                       f"({weak_conf:.0f}) — gap={gap:.1f}",
                confidence=new_conf,
            )

    # Return only the best displacement suggestion
    displacements.sort(key=lambda x: x["gap"], reverse=True)
    return displacements[:1]


def _do_exit(
    holding: Dict[str, Any],
    ticker: str,
    confidence: float,
    reason: str,
    scan_date: str,
) -> None:
    """Mark holding as exiting then remove it from the portfolio."""
    # Mark exiting first (so alert formatter can read the reason)
    upsert_portfolio_holding({
        "ticker": ticker,
        "status": "exiting",
        "current_confidence": confidence,
        "last_scored_at": scan_date,
        "notes": reason,
    })
    # Hard delete — the trade exit is handled by portfolio.py stop/target logic
    remove_portfolio_holding(ticker)
    logger.info(f"Portfolio EXIT: {ticker} — {reason}")
