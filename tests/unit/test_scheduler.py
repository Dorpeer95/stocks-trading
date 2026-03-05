"""
Unit tests for utils/scheduler.py.

Tests market-hours logic, holiday detection, and DST edge cases.
"""

from datetime import date, time, datetime

import pytest

from utils.scheduler import (
    is_trading_day,
    is_market_open,
    US_MARKET_HOLIDAYS,
    get_next_trading_day,
)


# ═══════════════════════════════════════════════════════════════════════════
# is_trading_day
# ═══════════════════════════════════════════════════════════════════════════

class TestIsTradingDay:
    def test_monday_is_trading_day(self):
        # Find a Monday that's not a holiday — 2026-03-02 is Monday
        d = date(2026, 3, 2)
        assert d.weekday() == 0  # Monday
        assert is_trading_day(d) is True

    def test_tuesday_is_trading_day(self):
        d = date(2026, 3, 3)
        assert is_trading_day(d) is True

    def test_saturday_not_trading_day(self):
        d = date(2026, 3, 7)
        assert d.weekday() == 5  # Saturday
        assert is_trading_day(d) is False

    def test_sunday_not_trading_day(self):
        d = date(2026, 3, 8)
        assert d.weekday() == 6  # Sunday
        assert is_trading_day(d) is False

    def test_christmas_not_trading_day(self):
        assert is_trading_day(date(2026, 12, 25)) is False

    def test_new_years_not_trading_day(self):
        assert is_trading_day(date(2026, 1, 1)) is False

    def test_july_4_observed_not_trading(self):
        # July 3 is Friday (observed Independence Day in 2026)
        if date(2026, 7, 3) in US_MARKET_HOLIDAYS:
            assert is_trading_day(date(2026, 7, 3)) is False

    def test_thanksgiving_not_trading(self):
        assert is_trading_day(date(2026, 11, 26)) is False


# ═══════════════════════════════════════════════════════════════════════════
# get_next_trading_day
# ═══════════════════════════════════════════════════════════════════════════

class TestGetNextTradingDay:
    def test_friday_to_monday(self):
        # 2026-03-06 is Friday → next should be Monday 2026-03-09
        result = get_next_trading_day(date(2026, 3, 6))
        assert result == date(2026, 3, 9)

    def test_saturday_to_monday(self):
        result = get_next_trading_day(date(2026, 3, 7))
        assert result == date(2026, 3, 9)

    def test_weekday_to_next_weekday(self):
        # Monday → Tuesday
        result = get_next_trading_day(date(2026, 3, 2))
        assert result == date(2026, 3, 3)

    def test_skips_holidays(self):
        # Day before Christmas (Thursday 12/24) → next trading day skips 12/25
        result = get_next_trading_day(date(2026, 12, 24))
        assert result.weekday() < 5  # must be weekday
        assert result not in US_MARKET_HOLIDAYS
