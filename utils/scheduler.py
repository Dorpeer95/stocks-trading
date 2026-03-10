"""
Market-hours-aware scheduler for stocks-agent.

Handles US market hours, holidays, DST transitions, and
all scheduled tasks (weekly scan, morning briefing, intraday monitor,
after-market review).
"""

import logging
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# US market holidays for 2026 (NYSE observed)
# Update annually — add 2027 when available
# ---------------------------------------------------------------------------
US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Jr. Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

# Combine all years
US_MARKET_HOLIDAYS = US_MARKET_HOLIDAYS_2026


def _now_et() -> datetime:
    """Current time in US Eastern."""
    return datetime.now(ET)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_trading_day(d: Optional[date] = None) -> bool:
    """Check if a date is a US trading day (Mon-Fri, not a holiday).

    Parameters
    ----------
    d : Date to check. Defaults to today in ET.
    """
    if d is None:
        d = _now_et().date()
    # Weekday: 0=Mon … 4=Fri, 5=Sat, 6=Sun
    if d.weekday() >= 5:
        return False
    if d in US_MARKET_HOLIDAYS:
        return False
    return True


def is_market_open() -> bool:
    """Check if the US stock market is currently open.

    Regular hours: 9:30 AM – 4:00 PM ET, Mon-Fri, excluding holidays.
    """
    now = _now_et()
    if not is_trading_day(now.date()):
        return False
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() < market_close


def is_premarket() -> bool:
    """Check if we're in the pre-market window (8:00-9:30 AM ET)."""
    now = _now_et()
    if not is_trading_day(now.date()):
        return False
    return time(8, 0) <= now.time() < time(9, 30)


def is_after_hours() -> bool:
    """Check if we're in the after-hours window (4:00-5:00 PM ET)."""
    now = _now_et()
    if not is_trading_day(now.date()):
        return False
    return time(16, 0) <= now.time() < time(17, 0)


def get_next_trading_day(d: Optional[date] = None) -> date:
    """Return the next trading day after *d* (or today)."""
    if d is None:
        d = _now_et().date()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def minutes_to_market_close() -> Optional[int]:
    """Minutes remaining until market close, or ``None`` if market is closed."""
    if not is_market_open():
        return None
    now = _now_et()
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = close - now
    return max(0, int(delta.total_seconds() / 60))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class MarketScheduler:
    """APScheduler wrapper with market-hours awareness.

    All times are in US Eastern and respect holidays/weekends.

    Usage
    -----
    >>> scheduler = MarketScheduler()
    >>> scheduler.schedule_weekly_scan(my_weekly_scan_func)
    >>> scheduler.schedule_morning_briefing(my_morning_func)
    >>> scheduler.start()
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone=ET)
        self._running = False
        logger.info("MarketScheduler created")

    # -- Schedule registration ------------------------------------------------

    def schedule_weekly_scan(self, callback: Callable) -> None:
        """Schedule the weekly scan for Sunday 8:00 PM ET."""
        self._scheduler.add_job(
            self._wrap(callback, "weekly_scan"),
            CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=ET),
            id="weekly_scan",
            name="Weekly stock scan",
            replace_existing=True,
            misfire_grace_time=3600,  # 1 hour grace
        )
        logger.info("Scheduled: weekly_scan — Sunday 20:00 ET")

    def schedule_morning_briefing(self, callback: Callable) -> None:
        """Schedule the morning briefing for Mon-Fri 8:30 AM ET."""
        self._scheduler.add_job(
            self._wrap(callback, "morning_briefing"),
            CronTrigger(
                day_of_week="mon-fri", hour=8, minute=30, timezone=ET
            ),
            id="morning_briefing",
            name="Morning briefing",
            replace_existing=True,
            misfire_grace_time=1800,
        )
        logger.info("Scheduled: morning_briefing — Mon-Fri 08:30 ET")

    def schedule_intraday_monitor(self, callback: Callable) -> None:
        """Schedule intraday position monitoring every 15 min during market hours.

        Only runs Mon-Fri, 9:30 AM – 4:00 PM ET.
        """
        self._scheduler.add_job(
            self._wrap_market_hours(callback, "intraday_monitor"),
            IntervalTrigger(minutes=15, timezone=ET),
            id="intraday_monitor",
            name="Intraday position monitor",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("Scheduled: intraday_monitor — every 15 min (market hours)")

    def schedule_after_market(self, callback: Callable) -> None:
        """Schedule the after-market review for Mon-Fri 4:30 PM ET."""
        self._scheduler.add_job(
            self._wrap(callback, "after_market"),
            CronTrigger(
                day_of_week="mon-fri", hour=16, minute=30, timezone=ET
            ),
            id="after_market",
            name="After-market review",
            replace_existing=True,
            misfire_grace_time=1800,
        )
        logger.info("Scheduled: after_market — Mon-Fri 16:30 ET")

    def schedule_model_check(self, callback: Callable) -> None:
        """Schedule ML model update check for Mon-Fri 6:00 AM ET."""
        self._scheduler.add_job(
            self._wrap(callback, "model_check"),
            CronTrigger(
                day_of_week="mon-fri", hour=6, minute=0, timezone=ET
            ),
            id="model_check",
            name="ML model update check",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info("Scheduled: model_check — Mon-Fri 06:00 ET")

    def schedule_daily_health_check(self, callback: Callable) -> None:
        """Schedule the daily health check for 12:00 PM ET every day."""
        self._scheduler.add_job(
            self._wrap(callback, "daily_health_check"),
            CronTrigger(hour=12, minute=0, timezone=ET),  # Runs 7 days a week
            id="daily_health_check",
            name="Daily health check",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info("Scheduled: daily_health_check — Daily 12:00 ET")

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        self._scheduler.start()
        self._running = True
        logger.info("MarketScheduler started")
        self._log_next_runs()

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._running:
            return
        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info("MarketScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_next_run_times(self) -> dict:
        """Return the next fire time for each scheduled job."""
        result = {}
        for job in self._scheduler.get_jobs():
            result[job.id] = (
                job.next_run_time.isoformat() if job.next_run_time else None
            )
        return result

    # -- Internal helpers -----------------------------------------------------

    def _wrap(self, callback: Callable, name: str) -> Callable:
        """Wrap a callback with logging, holiday check, and error handling."""

        def wrapper() -> None:
            # Skip holidays for weekday-only jobs
            if name not in ("weekly_scan",) and not is_trading_day():
                logger.info(
                    f"Skipping {name}: not a trading day "
                    f"({_now_et().date()})"
                )
                return
            logger.info(f"=== {name} START ===")
            try:
                callback()
            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)
            finally:
                logger.info(f"=== {name} END ===")

        return wrapper

    def _wrap_market_hours(self, callback: Callable, name: str) -> Callable:
        """Wrap a callback that only runs during market hours."""

        def wrapper() -> None:
            if not is_market_open():
                return  # Silently skip — runs every 15 min, no need to log
            logger.debug(f"{name} tick")
            try:
                callback()
            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)

        return wrapper

    def _log_next_runs(self) -> None:
        """Log the next scheduled run time for each job."""
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            if next_run:
                logger.info(
                    f"  Next {job.id}: {next_run.strftime('%Y-%m-%d %H:%M %Z')}"
                )
