"""
stocks-agent — main entry point.

Initialises all components, starts the scheduler and health server,
and handles graceful shutdown.
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment FIRST
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("STOCKS_LOG_LEVEL", "INFO").upper()
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "stocks-agent.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=3,
        ),
    ],
)

logger = logging.getLogger("stocks-agent")

# ---------------------------------------------------------------------------
# Import project modules AFTER env + logging are ready
# ---------------------------------------------------------------------------
from agent.agent import AgentLoop  # noqa: E402
from agent.persistence import init_supabase  # noqa: E402
from health import run_health_server, update_status  # noqa: E402
from utils.scheduler import MarketScheduler  # noqa: E402
from utils.telegram_bot import init_bot, send_alert  # noqa: E402

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
scheduler: MarketScheduler | None = None
agent: AgentLoop | None = None
shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

def _handle_shutdown(signum: int, frame) -> None:
    """Handle SIGTERM / SIGINT for graceful shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} — shutting down gracefully")
    shutdown_event.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def startup() -> None:
    """Initialise all components.

    Each component initialises independently — a failure in one does not
    block the others.  The bot can run in degraded mode (e.g. without
    Supabase persistence or Telegram alerts).
    """
    global scheduler, agent

    logger.info("=" * 60)
    logger.info("  stocks-agent starting up")
    logger.info("=" * 60)

    env = os.getenv("STOCKS_ENV", "development")
    dry_run = os.getenv("STOCKS_DRY_RUN", "false").lower() == "true"
    logger.info(f"Environment: {env} | Dry run: {dry_run}")

    # Track init status
    init_ok = {"supabase": False, "telegram": False}

    # 1. Supabase
    try:
        init_supabase()
        init_ok["supabase"] = True
        logger.info("✅ Supabase connected")
    except Exception as e:
        logger.error(f"❌ Supabase init failed (will retry on first write): {e}")

    # 2. Telegram
    try:
        init_bot()
        init_ok["telegram"] = True
        logger.info("✅ Telegram bot connected")
        from utils.telegram_bot import start_command_listener
        start_command_listener()
    except Exception as e:
        logger.error(f"❌ Telegram init failed (alerts will be logged): {e}")

    # 3. Health server (background thread)
    health_thread = threading.Thread(
        target=run_health_server,
        daemon=True,
        name="health-server",
    )
    health_thread.start()
    logger.info("✅ Health server started")

    # 4. Agent loop
    agent = AgentLoop()
    logger.info("✅ Agent loop initialised")

    # 5. Scheduler — wire to real agent methods
    scheduler = MarketScheduler()
    scheduler.schedule_weekly_scan(agent.weekly_scan)
    scheduler.schedule_morning_briefing(agent.morning_briefing)
    scheduler.schedule_intraday_monitor(agent.intraday_monitor)
    scheduler.schedule_after_market(agent.after_market_review)
    scheduler.schedule_model_check(agent.model_check)
    scheduler.schedule_daily_health_check(agent.daily_health_check)
    scheduler.start()
    logger.info("✅ Scheduler started")

    # Update status for health endpoint
    next_runs = scheduler.get_next_run_times()
    update_status("next_weekly_scan", next_runs.get("weekly_scan", ""))
    update_status("init_supabase", str(init_ok["supabase"]))
    update_status("init_telegram", str(init_ok["telegram"]))

    logger.info("=" * 60)
    logger.info("  stocks-agent ready")
    if not all(init_ok.values()):
        failed = [k for k, v in init_ok.items() if not v]
        logger.warning(f"  ⚠️  Running in degraded mode — failed: {failed}")
    logger.info("=" * 60)

    # Send startup alert with full system status
    if init_ok["telegram"]:
        try:
            send_alert("bot_start", {
                "env": env,
                "dry_run": dry_run,
                "init_ok": init_ok,
                "next_weekly_scan": next_runs.get("weekly_scan", "unknown"),
                "next_morning_briefing": next_runs.get("morning_briefing", "unknown"),
            })
        except Exception as e:
            logger.warning(f"Startup alert failed: {e}")


def shutdown() -> None:
    """Clean up all components."""
    logger.info("Shutting down stocks-agent...")

    if scheduler and scheduler.is_running:
        scheduler.stop()
        logger.info("Scheduler stopped")

    try:
        send_alert("bot_stop", {})
    except Exception:
        pass  # Best effort

    logger.info("stocks-agent stopped")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="stocks-agent main entry point")
    parser.add_argument("--force-scan", action="store_true", help="Force run the weekly scan immediately and exit")
    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    try:
        startup()
        
        if args.force_scan:
            logger.info("🚀 --force-scan flag detected. Running scan immediately...")
            if agent is not None:
                agent.weekly_scan()
                logger.info("✅ Force scan completed.")
            else:
                logger.error("Agent failed to initialize. Cannot run force scan.")
            shutdown()
            sys.exit(0)

        # Keep main thread alive until shutdown signal
        logger.info("Main thread waiting for shutdown signal...")
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        try:
            from utils.telegram_bot import send_alert
            send_alert("action_needed", {
                "ticker": "SYSTEM",
                "reason": f"🛑 FATAL BOT CRASH: {str(e)}",
                "recommendation": "Check server logs for traceback."
            })
        except Exception:
            pass
    finally:
        shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
