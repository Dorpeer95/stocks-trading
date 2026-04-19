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
from agent.regime_engine import assess_regime
from agent.market_safety import get_market_safety
from agent.stagnation_detector import scan_for_stagnation
from agent.persistence import (
    get_open_positions,
    get_pending_opportunities,
    get_recent_events,
    insert_opportunities,
    insert_gpt_briefing,
    insert_signal_history,
    get_consecutive_strong_weeks,
)
from agent.portfolio import (
    PORTFOLIO_VALUE,
    can_open_position,
    generate_eod_summary,
    get_portfolio_summary,
    take_equity_snapshot,
    update_positions_intraday,
)
from agent.portfolio_manager import run_portfolio_cycle, ENTRY_THRESHOLD
from agent.scanner import scan_universe, filter_by_ml_direction
from agent.scorer import build_opportunity, score_candidates, ENABLE_ML, MIN_CONFIDENCE
from health import update_status, get_status, check_memory
from utils.data_loader import fetch_macro_data, clear_cache
from utils.helpers import safe_float
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
from utils.telegram_bot import send_alert, is_paused

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
        self._entry_alert_cache: Dict[str, float] = {}  # ticker → last alert timestamp

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

            # 2. Cross-asset regime (force refresh on weekly scan)
            cross_regime = assess_regime(force_refresh=True)
            logger.info(
                f"Cross-asset regime: {cross_regime['regime']} "
                f"score={cross_regime['score']:.0f} "
                f"size_mod={cross_regime['position_size_modifier']:.2f}"
            )

            # 3. Macro data + VIX regime (now blended with cross-asset via events.py)
            macro = fetch_macro_data()
            events, regime = detect_and_persist_events(macro) if macro else ([], None)
            self._last_regime = regime
            self._last_events = events

            # 3b. Market safety gate (breadth + distribution)
            safety = get_market_safety()
            logger.info(
                f"Market safety: safe={safety['safe_to_enter']} "
                f"size_mod={safety['size_modifier']:.2f} — {safety['summary']}"
            )
            if not safety["safe_to_enter"]:
                logger.warning(f"Market safety GATE TRIGGERED — {safety['summary']}")
                send_alert("action_needed", {
                    "ticker": "MARKET",
                    "action": "Entry gate active — new entries paused",
                    "reason": safety["summary"],
                })

            # 4. ML pre-filter (Phase 3) — skip in emergency mode
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

            # 5. Build signal_age_map from signal history (for scoring bonus)
            today_str = date.today().isoformat()
            signal_age_map: Dict[str, int] = {}
            for c in candidates:
                t = c.get("ticker")
                if t:
                    signal_age_map[t] = get_consecutive_strong_weeks(t, MIN_CONFIDENCE)

            # 6. Score candidates (with fundamentals/insider + ML + sentiment + signal age)
            fetch_extras = ENABLE_INSIDER or ENABLE_EARNINGS
            scored = score_candidates(
                candidates,
                regime=regime,
                fetch_extras=fetch_extras,
                ml_predictions_map=ml_predictions_map if ENABLE_ML else None,
                sentiment_map=sentiment_map if ENABLE_SENTIMENT else None,
                signal_age_map=signal_age_map,
            )

            # 7. Persist signal history for ALL scored candidates this week
            if not DRY_RUN:
                try:
                    sig_records = [
                        {
                            "ticker": s["ticker"],
                            "scan_date": today_str,
                            "confidence": s.get("confidence", 0),
                            "sub_scores": s.get("sub_scores", {}),
                            "in_portfolio": False,   # updated by portfolio_manager
                            "setup_type": s.get("setup_type", ""),
                            "passed_scan": True,
                        }
                        for s in scored if s.get("ticker")
                    ]
                    if sig_records:
                        insert_signal_history(sig_records)
                except Exception as e:
                    logger.warning(f"signal_history insert failed: {e}")

            # 7b. Auto-calibrate signal weights (once we have ≥20 postmortems)
            if not DRY_RUN:
                try:
                    from agent.weight_calibrator import calibrate_weights
                    calibration = calibrate_weights(min_sample=20)
                    if calibration:
                        logger.info(
                            f"Weight calibration complete: "
                            f"{calibration['weights']}"
                        )
                except Exception as e:
                    logger.warning(f"Weight calibration failed: {e}")

            # 8. Run portfolio state machine — the core managed-portfolio logic
            portfolio_diff: Dict[str, Any] = {}
            try:
                portfolio_diff = run_portfolio_cycle(scored, scan_date=today_str)
            except Exception as e:
                logger.error(f"portfolio_cycle failed: {e}", exc_info=True)

            # 9. Build opportunities from portfolio new_entries + active holdings
            #    (for Alpaca execution and legacy persistence)
            opportunities: List[Dict[str, Any]] = []

            # Compute combined size modifier: regime × safety breadth
            regime_size_mod = regime.get("position_size_modifier", 1.0) if regime else 1.0
            combined_size_mod = round(regime_size_mod * safety["size_modifier"], 2)

            # Gate: skip new entries entirely if safety gate triggered
            #        OR if user has paused the bot via /pause.
            allow_new_entries = safety["safe_to_enter"] and not is_paused()
            if is_paused():
                logger.warning("Bot is /paused — suppressing new entries this scan")

            entry_tickers = {c["ticker"] for c in portfolio_diff.get("new_entries", [])}
            scored_map = {s["ticker"]: s for s in scored if s.get("ticker")}
            for stock in portfolio_diff.get("new_entries", []):
                if not allow_new_entries:
                    logger.warning(
                        f"Skipping new entry {stock.get('ticker')} — "
                        "market safety gate active"
                    )
                    continue
                # Pass combined modifier so position sizer respects regime + breadth
                opp_regime = dict(regime) if regime else {}
                opp_regime["position_size_modifier"] = combined_size_mod
                opp = build_opportunity(
                    stock, portfolio_value=PORTFOLIO_VALUE, regime=opp_regime
                )
                opportunities.append(opp)

            if opportunities and not DRY_RUN:
                try:
                    from agent.persistence import _table
                    existing = _table("opportunities").select("ticker").eq("scan_date", today_str).execute().data
                    existing_tickers = {row["ticker"] for row in existing} if existing else set()
                except Exception as e:
                    logger.warning(f"Dedup check failed: {e}")
                    existing_tickers = set()

                new_opps = [o for o in opportunities if o.get("ticker") not in existing_tickers]
                if new_opps:
                    insert_opportunities(new_opps)
                    logger.info(f"Persisted {len(new_opps)} new opportunities")

                logger.info("Triggering Alpaca buy execution for new portfolio entries...")
                from agent.portfolio import execute_buy_opportunities
                execute_buy_opportunities(opportunities)

            # 10. GPT-4o weekly briefing
            gpt_briefing: Optional[str] = None
            if ENABLE_GPT and not emergency:
                try:
                    gpt_briefing = gpt4o_weekly_briefing({
                        "opportunities": opportunities,
                        "market_mood": regime.get("mood", "Neutral") if regime else "Neutral",
                        "hot_sectors": scan.get("hot_sectors", []),
                        "regime": regime,
                        "events": self._last_events,
                        "portfolio_diff": portfolio_diff,
                    })
                    if gpt_briefing and not DRY_RUN:
                        insert_gpt_briefing(
                            gpt_briefing,
                            regime.get("mood", "Neutral") if regime else "Neutral",
                        )
                except Exception as e:
                    logger.warning(f"GPT weekly briefing failed: {e}")

            # 11. Send portfolio update alert (new format)
            mood = regime.get("mood", "Neutral") if regime else "Neutral"
            alert_data: Dict[str, Any] = {
                "portfolio_diff": portfolio_diff,
                "market_mood": mood,
                "hot_sectors": scan.get("hot_sectors", []),
                "regime": regime,
                "cross_asset_regime": cross_regime,
                "safety": safety,
                "combined_size_mod": combined_size_mod,
            }
            if gpt_briefing:
                alert_data["gpt_briefing"] = gpt_briefing

            send_alert("portfolio_update", alert_data)

            # 11. Take equity snapshot
            if not DRY_RUN:
                take_equity_snapshot()

            # 12. Monthly cleanup (Removed daily_scores purge)
            pass

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
        """Pre-market briefing with action plan: enter now / hold / watch."""
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
            open_tickers = {p.get("ticker") for p in positions}
            pos_details = []
            for p in positions:
                pos_details.append({
                    "ticker": p.get("ticker"),
                    "entry_price": p.get("entry_price"),
                    "current_price": p.get("current_price"),
                    "unrealized_pnl": p.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": p.get("unrealized_pnl_pct", 0),
                    "stop_loss": p.get("stop_loss"),
                    "target_price": p.get("target_price"),
                    "days_held": p.get("days_held", 0),
                })

            # Watchlist: check pending opportunities against current prices
            enter_now: list = []
            watchlist: list = []
            try:
                pending = get_pending_opportunities()
                for opp in pending[:8]:  # check top 8 by confidence
                    ticker = opp.get("ticker")
                    # entry_price may be null on old rows — fall back to low/high avg
                    entry_price = safe_float(opp.get("entry_price"))
                    if not entry_price:
                        low = safe_float(opp.get("entry_price_low"))
                        high = safe_float(opp.get("entry_price_high"))
                        entry_price = round((low + high) / 2, 2) if (low or high) else 0
                    if not ticker or not entry_price or ticker in open_tickers:
                        continue
                    try:
                        from utils.data_loader import fetch_price_data
                        df = fetch_price_data(ticker, period="5d", interval="1d")
                        if df is None or df.empty:
                            continue
                        current = float(df["Close"].iloc[-1])
                        # distance_pct: negative means below trigger (still waiting)
                        distance_pct = (current - entry_price) / entry_price * 100
                        item = {
                            "ticker": ticker,
                            "current_price": current,
                            "entry_price": entry_price,
                            "stop_loss": safe_float(opp.get("stop_loss")),
                            "target_price": safe_float(opp.get("target_price")),
                            "confidence": opp.get("confidence", 0),
                            "distance_pct": distance_pct,
                            "notes": opp.get("notes", ""),
                        }
                        # Within 2% below or any amount above entry = enter now
                        if distance_pct >= -2.0:
                            enter_now.append(item)
                        else:
                            watchlist.append(item)
                    except Exception as e:
                        logger.warning(f"Failed to check price for {ticker}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load pending opportunities: {e}")

            # Sort watchlist closest to entry first
            watchlist.sort(key=lambda x: x.get("distance_pct", -999), reverse=True)

            # Event descriptions
            event_texts = [
                {"severity": e.get("severity", "low"), "event_detail": e.get("description", "")}
                for e in events
            ]

            # Macro summary (pass full dicts so formatter can read change_pct)
            macro_summary = macro or {}

            # Breadth / safety snapshot for morning context
            morning_safety = get_market_safety()

            send_alert("morning_briefing", {
                "events": event_texts,
                "positions": pos_details,
                "macro": macro_summary,
                "enter_now": enter_now,
                "watchlist": watchlist,
                "safety": morning_safety,
            })

            update_status(
                "last_morning_briefing",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            logger.info(
                f"Morning briefing sent — {len(enter_now)} enter now, "
                f"{len(watchlist)} watching"
            )

        except Exception as e:
            logger.error(f"Morning briefing failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # 3. INTRADAY MONITOR (every 15 min during market hours)
    # ------------------------------------------------------------------
    def intraday_monitor(self) -> None:
        """Check open positions for exit signals and watchlist for entry triggers."""
        logger.debug("Intraday monitor tick")

        try:
            # 1. Exit checks on open positions
            actions = update_positions_intraday()

            # 1b. Stagnation scan — reuse positions already fetched by update_positions_intraday
            try:
                _positions_for_stagnation = get_open_positions()
                stagnant = scan_for_stagnation(_positions_for_stagnation)
                for sa in stagnant:
                    cache_key = f"stagnant_{sa['ticker']}"
                    if time.time() - self._entry_alert_cache.get(cache_key, 0) > 86400:
                        actions.append(sa)
                        self._entry_alert_cache[cache_key] = time.time()
            except Exception as e:
                logger.debug(f"Stagnation scan failed: {e}")

            for action in actions:
                urgency = action.get("urgent", "medium")
                if urgency in ("high", "medium"):
                    send_alert("action_needed", action)
                    logger.info(
                        f"Action alert: {action['ticker']} — "
                        f"{action['action']}"
                    )

            # 2. Entry trigger checks on watchlist — skipped if user has paused
            if is_paused():
                logger.debug("Bot paused — skipping intraday entry checks")
                return

            try:
                open_tickers = {p.get("ticker") for p in _positions_for_stagnation}
                pending = get_pending_opportunities()

                for opp in pending[:8]:
                    ticker = opp.get("ticker")
                    entry_price = safe_float(opp.get("entry_price"))
                    if not entry_price:
                        low = safe_float(opp.get("entry_price_low"))
                        high = safe_float(opp.get("entry_price_high"))
                        entry_price = round((low + high) / 2, 2) if (low or high) else 0
                    if not ticker or not entry_price or ticker in open_tickers:
                        continue

                    # Skip if we already alerted this ticker recently
                    alerted_key = f"entry_alerted_{ticker}"
                    last_alert = self._entry_alert_cache.get(alerted_key, 0)
                    if time.time() - last_alert < 3600:  # 1h cooldown
                        continue

                    try:
                        # Prefer Alpaca realtime; fall back to yfinance 5m if unavailable.
                        current: Optional[float] = None
                        try:
                            from utils.alpaca_broker import get_latest_price
                            current = get_latest_price(ticker)
                        except Exception:
                            current = None

                        if current is None:
                            from utils.data_loader import fetch_price_data
                            df = fetch_price_data(ticker, period="1d", interval="5m")
                            if df is None or df.empty:
                                continue
                            current = float(df["Close"].iloc[-1])

                        if current >= entry_price:
                            send_alert("entry_signal", {
                                "ticker": ticker,
                                "current_price": current,
                                "entry_price": entry_price,
                                "stop_loss": safe_float(opp.get("stop_loss")),
                                "target_price": safe_float(opp.get("target_price")),
                                "confidence": opp.get("confidence", 0),
                                "notes": opp.get("notes", ""),
                            })
                            self._entry_alert_cache[alerted_key] = time.time()
                            logger.info(f"Entry signal fired: {ticker} at ${current:.2f}")
                    except Exception as e:
                        logger.warning(f"Entry check failed for {ticker}: {e}")

            except Exception as e:
                logger.warning(f"Watchlist entry check failed: {e}")

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


    # ------------------------------------------------------------------
    # 6. DAILY HEALTH CHECK (Daily 12:00 PM ET)
    # ------------------------------------------------------------------
    def daily_health_check(self) -> None:
        """Send a daily heartbeat to Telegram so the user knows the bot is alive."""
        logger.info("Daily health check — start")
        try:
            from health import check_memory
            from utils.telegram_bot import send_message
            
            mem_info = check_memory()
            mem_pct = mem_info.get("percent", 0.0)
            
            msg = f"🟢 Daily Health Check: Bot is running smoothly.\n"
            msg += f"Memory Usage: {mem_pct:.1f}%\n"
            msg += f"Open Positions: {len(get_open_positions())}"
            
            send_message(msg)
            
            update_status(
                "last_health_check",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            logger.info("Daily health check sent")
            
        except Exception as e:
            logger.error(f"Daily health check failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
