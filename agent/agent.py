"""
Agent loop — wires all modules into the scheduled callbacks.

This is the brain of the bot. Each method corresponds to a scheduled
event from ``MarketScheduler`` and orchestrates the correct sequence
of data fetching, analysis, scoring, and alerting.

Phase 3: ML models are loaded once on startup / model_check and
used during the weekly scan for pre-filtering and scoring.

Phase 4: VADER + GPT-4o sentiment pipeline provides news-based
scoring and human-readable briefings.
"""

import logging
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional

from agent.ai_model import ModelManager
from agent.events import detect_and_persist_events, assess_market_regime
from agent.persistence import (
    get_open_positions,
    get_pending_opportunities,
    get_recent_events,
    insert_daily_scores,
    insert_opportunities,
    purge_old_scores,
    insert_gpt_briefing,
)
from agent.portfolio import (
    PORTFOLIO_VALUE,
    can_open_position,
    generate_eod_summary,
    get_portfolio_summary,
    take_equity_snapshot,
    update_positions_intraday,
)
from agent.scanner import scan_universe, filter_by_ml_direction
from agent.scorer import build_opportunity, score_candidates, ENABLE_ML
from health import update_status, get_status, check_memory
from utils.data_loader import fetch_macro_data, clear_cache
from utils.sentiment import (
    aggregate_sentiment,
    batch_sentiment,
    gpt4o_weekly_briefing,
    gpt4o_exit_analysis,
    fetch_news,
    vader_score,
    cost_tracker,
    get_newsapi_usage,
)
from utils.telegram_bot import send_alert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
DRY_RUN = os.getenv("STOCKS_DRY_RUN", "false").lower() == "true"
ENABLE_INSIDER = os.getenv("STOCKS_ENABLE_INSIDER", "true").lower() == "true"
ENABLE_EARNINGS = os.getenv("STOCKS_ENABLE_EARNINGS", "true").lower() == "true"
ENABLE_SENTIMENT = os.getenv("STOCKS_ENABLE_NEWS", "true").lower() == "true"
ENABLE_GPT = os.getenv("STOCKS_ENABLE_GPT", "true").lower() == "true"


class AgentLoop:
    """Orchestrates all scheduled agent tasks."""

    def __init__(self) -> None:
        self._last_scan_result: Optional[Dict[str, Any]] = None
        self._last_regime: Optional[Dict[str, Any]] = None
        self._last_events: List[Dict[str, Any]] = []

        # ML model manager (Phase 3)
        self._model_manager = ModelManager()
        self._models_loaded = False

    # ------------------------------------------------------------------
    # 1. WEEKLY SCAN (Sunday 8 PM ET)
    # ------------------------------------------------------------------
    def weekly_scan(self) -> None:
        """Full weekly universe scan → score → build opportunities.

        Steps:
        1. Clear data cache
        2. Scan universe (download prices, compute indicators, RS)
        3. Fetch macro data and assess regime
        4. Score candidates
        5. Build and persist opportunities
        6. Send Telegram summary
        """
        logger.info("=" * 60)
        logger.info("  WEEKLY SCAN — START")
        logger.info("=" * 60)
        start = time.time()

        try:
            # Pre-flight: memory check
            mem = check_memory()
            if mem.get("emergency_mode"):
                logger.warning(
                    "Emergency mode active — running lightweight scan "
                    "(no ML, no sentiment, top 100 only)"
                )
                update_status("last_error", "Emergency mode: memory pressure")

            # Clear stale cache for fresh data
            clear_cache()

            # 1. Scan universe
            scan = scan_universe()
            self._last_scan_result = scan
            candidates = scan.get("candidates", [])

            if not candidates:
                logger.warning("No candidates from scan")
                send_alert("weekly_summary", {
                    "opportunities": [],
                    "market_mood": "Neutral",
                    "hot_sectors": scan.get("hot_sectors", []),
                })
                update_status("last_scan", time.strftime("%Y-%m-%dT%H:%M:%S"))
                return

            # 2. Macro data + regime
            macro = fetch_macro_data()
            events, regime = detect_and_persist_events(macro) if macro else ([], None)
            self._last_regime = regime
            self._last_events = events

            # 3. ML pre-filter (Phase 3) — skip in emergency mode
            emergency = mem.get("emergency_mode", False)
            ml_predictions_map: Dict[str, Dict[str, Any]] = {}
            if ENABLE_ML and self._models_loaded and not emergency:
                candidates, ml_predictions_map = filter_by_ml_direction(
                    candidates, self._model_manager
                )
                if not candidates:
                    logger.warning("All candidates rejected by ML filter")
                    send_alert("weekly_summary", {
                        "opportunities": [],
                        "market_mood": regime.get("mood", "Neutral") if regime else "Neutral",
                        "hot_sectors": scan.get("hot_sectors", []),
                    })
                    update_status("last_scan", time.strftime("%Y-%m-%dT%H:%M:%S"))
                    return

            # 4. Sentiment analysis (Phase 4) — skip in emergency mode
            sentiment_map: Dict[str, Dict[str, Any]] = {}
            if ENABLE_SENTIMENT and not emergency:
                tickers = [c.get("ticker") for c in candidates if c.get("ticker")]
                try:
                    sentiment_map = batch_sentiment(
                        tickers[:15],  # cap to control API usage
                        use_gpt=ENABLE_GPT,
                    )
                    logger.info(
                        f"Sentiment fetched for {len(sentiment_map)} tickers"
                    )
                except Exception as e:
                    logger.warning(f"Batch sentiment failed: {e}")

            # 5. Score candidates (with fundamentals/insider + ML + sentiment)
            fetch_extras = ENABLE_INSIDER or ENABLE_EARNINGS
            scored = score_candidates(
                candidates,
                regime=regime,
                fetch_extras=fetch_extras,
                ml_predictions_map=ml_predictions_map if ENABLE_ML else None,
                sentiment_map=sentiment_map if ENABLE_SENTIMENT else None,
            )

            # 6. Build opportunities
            opportunities: List[Dict[str, Any]] = []
            for stock in scored[:10]:  # top 10
                opp = build_opportunity(
                    stock,
                    portfolio_value=PORTFOLIO_VALUE,
                    regime=regime,
                )
                opportunities.append(opp)

            # 7. Persist & Execute
            if opportunities and not DRY_RUN:
                insert_opportunities(opportunities)
                logger.info(f"Persisted {len(opportunities)} opportunities")
                
                # logger.info("Triggering autonomous Alpaca buy execution for new opportunities...")
                # from agent.portfolio import execute_buy_opportunities
                # execute_buy_opportunities(opportunities)

            # 8. Persist daily scores (all scanned stocks)
            daily_scores = _build_daily_scores(
                scan.get("candidates", []) + [
                    c for c in scored if c not in scan.get("candidates", [])
                ]
            )
            if daily_scores and not DRY_RUN:
                insert_daily_scores(daily_scores)

            # 9. GPT-4o weekly briefing (Phase 4) — skip in emergency mode
            gpt_briefing: Optional[str] = None
            if ENABLE_GPT and opportunities and not emergency:
                try:
                    gpt_briefing = gpt4o_weekly_briefing({
                        "opportunities": opportunities,
                        "market_mood": regime.get("mood", "Neutral") if regime else "Neutral",
                        "hot_sectors": scan.get("hot_sectors", []),
                        "regime": regime,
                        "events": self._last_events,
                    })
                    if gpt_briefing:
                        logger.info(
                            f"GPT weekly briefing generated "
                            f"({len(gpt_briefing)} chars)"
                        )
                        if not DRY_RUN:
                            insert_gpt_briefing(
                                gpt_briefing, 
                                regime.get("mood", "Neutral") if regime else "Neutral"
                            )
                except Exception as e:
                    logger.warning(f"GPT weekly briefing failed: {e}")

            # 10. Send Telegram alert
            mood = regime.get("mood", "Neutral") if regime else "Neutral"
            alert_data: Dict[str, Any] = {
                "opportunities": opportunities,
                "market_mood": mood,
                "hot_sectors": scan.get("hot_sectors", []),
            }
            if gpt_briefing:
                alert_data["gpt_briefing"] = gpt_briefing

            send_alert("weekly_summary", alert_data)

            # 11. Take equity snapshot
            if not DRY_RUN:
                take_equity_snapshot()

            # 12. Purge old scores (monthly cleanup)
            if date.today().day == 1:
                purge_old_scores(days=180)

            elapsed = time.time() - start
            update_status("last_scan", time.strftime("%Y-%m-%dT%H:%M:%S"))
            update_status("scan_duration_s", f"{elapsed:.0f}")
            update_status("opportunities_count", str(len(opportunities)))

            logger.info(
                f"Weekly scan complete in {elapsed:.0f}s — "
                f"{len(opportunities)} opportunities"
            )

        except Exception as e:
            logger.error(f"Weekly scan failed: {e}", exc_info=True)
            try:
                send_alert("action_needed", {
                    "ticker": "SYSTEM",
                    "action": "Weekly scan failed",
                    "reason": str(e),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 2. MORNING BRIEFING (Mon-Fri 8:30 AM ET)
    # ------------------------------------------------------------------
    def morning_briefing(self) -> None:
        """Pre-market briefing with positions and macro context."""
        logger.info("Morning briefing — start")

        try:
            # Macro data
            macro = fetch_macro_data()
            events = []
            regime = self._last_regime

            if macro:
                events, regime = detect_and_persist_events(macro)
                self._last_regime = regime
                self._last_events = events

            # Open positions
            positions = get_open_positions()
            pos_details = []
            for p in positions:
                pos_details.append({
                    "ticker": p.get("ticker"),
                    "entry_price": p.get("entry_price"),
                    "current_price": p.get("current_price"),
                    "unrealized_pnl": p.get("unrealized_pnl", 0),
                    "stop_loss": p.get("stop_loss"),
                    "target_price": p.get("target_price"),
                    "days_held": p.get("days_held", 0),
                })

            # Event descriptions
            event_texts = [e.get("description", "") for e in events]

            # Macro summary
            macro_summary = {}
            if macro:
                for key, val in macro.items():
                    macro_summary[key] = val.get("current", 0)

            send_alert("morning_briefing", {
                "events": event_texts,
                "positions": pos_details,
                "macro": macro_summary,
            })

            update_status(
                "last_morning_briefing",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            logger.info("Morning briefing sent")

        except Exception as e:
            logger.error(f"Morning briefing failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # 3. INTRADAY MONITOR (every 15 min during market hours)
    # ------------------------------------------------------------------
    def intraday_monitor(self) -> None:
        """Check open positions for exit signals."""
        logger.debug("Intraday monitor tick")

        try:
            actions = update_positions_intraday()

            for action in actions:
                urgency = action.get("urgency", "medium")
                if urgency in ("high", "medium"):
                    send_alert("action_needed", action)
                    logger.info(
                        f"Action alert: {action['ticker']} — "
                        f"{action['action']}"
                    )

        except Exception as e:
            logger.error(f"Intraday monitor failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # 4. AFTER-MARKET REVIEW (Mon-Fri 4:30 PM ET)
    # ------------------------------------------------------------------
    def after_market_review(self) -> None:
        """End-of-day portfolio summary."""
        logger.info("After-market review — start")

        try:
            # Final position update for the day
            update_positions_intraday()

            # Generate and send EOD summary
            eod = generate_eod_summary()
            send_alert("eod_summary", eod)

            update_status(
                "last_eod_review",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            logger.info("After-market review sent")

        except Exception as e:
            logger.error(f"After-market review failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # 5. MODEL CHECK (Mon-Fri 6 AM ET)
    # ------------------------------------------------------------------
    def model_check(self) -> None:
        """Check for ML model updates and load/reload from Supabase.

        Called daily at 6 AM ET.  Downloads the latest active model
        versions from Supabase Storage and loads them into memory.
        On first run, this is what initially loads the models.
        """
        logger.info("Model check — start")

        try:
            loaded = self._model_manager.load_from_supabase()

            if loaded:
                self._models_loaded = True
                model_names = list(self._model_manager.models.keys())
                logger.info(f"Models loaded: {model_names}")
                update_status("models_loaded", ",".join(model_names))
            else:
                # Fall back to local disk models
                logger.info("Supabase load failed — trying local disk")
                loaded_local = self._model_manager.load_from_disk()
                if loaded_local:
                    self._models_loaded = True
                    model_names = list(self._model_manager.models.keys())
                    logger.info(f"Models loaded from disk: {model_names}")
                    update_status("models_loaded", ",".join(model_names))
                else:
                    logger.warning("No ML models available — running without ML")
                    self._models_loaded = False
                    update_status("models_loaded", "none")

            update_status(
                "last_model_check",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        except Exception as e:
            logger.error(f"Model check failed: {e}", exc_info=True)
            # Don't crash — agent continues without ML
            self._models_loaded = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_daily_scores(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build daily_scores records from scored stock data."""
    today = date.today().isoformat()
    scores: List[Dict[str, Any]] = []

    for s in stocks:
        ticker = s.get("ticker")
        if not ticker:
            continue

        scores.append({
            "ticker": ticker,
            "score_date": today,
            "confidence": s.get("confidence", 0),
            "technical_score": s.get("sub_scores", {}).get("technical"),
            "rs_score": s.get("sub_scores", {}).get("relative_strength"),
            "fundamental_score": s.get("sub_scores", {}).get("fundamental"),
            "sentiment_score": s.get("sub_scores", {}).get("sentiment"),
            "sub_scores": s.get("sub_scores", {}),
            "setup_type": s.get("setup_type"),
        })

    return scores
