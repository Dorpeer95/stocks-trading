"""
Unit tests for agent/portfolio.py.

Tests portfolio constraint checks and helper functions.
Database-dependent functions are mocked.
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.portfolio import (
    _calc_trailing_stop,
    _record_trade,
)


# ═══════════════════════════════════════════════════════════════════════════
# _calc_trailing_stop
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcTrailingStop:
    def test_atr_based_stop(self):
        # current=110, ATR=2, mult=2 → trailing = 110 - 4 = 106
        result = _calc_trailing_stop(110.0, 100.0, atr_value=2.0)
        assert result == 106.0  # 110 - (2 * 2.0)

    def test_fallback_percentage_in_profit(self):
        # No ATR, in profit (current > entry) → 5% below current
        result = _calc_trailing_stop(110.0, 100.0, atr_value=None)
        assert abs(result - 104.5) < 0.01  # 110 * 0.95

    def test_fallback_none_not_in_profit(self):
        # No ATR, not in profit → None
        result = _calc_trailing_stop(95.0, 100.0, atr_value=None)
        assert result is None

    def test_zero_atr_uses_fallback(self):
        result = _calc_trailing_stop(110.0, 100.0, atr_value=0.0)
        assert abs(result - 104.5) < 0.01

    def test_atr_stop_below_entry(self):
        # current=102, ATR=5, mult=2 → 102 - 10 = 92
        result = _calc_trailing_stop(102.0, 100.0, atr_value=5.0)
        assert result == 92.0


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio summary (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioSummaryLogic:
    """Test the portfolio math with mocked DB calls."""

    @patch("agent.portfolio.get_open_positions")
    @patch("agent.portfolio.get_trade_history")
    def test_win_rate_calculation(self, mock_hist, mock_pos):
        mock_pos.return_value = []
        mock_hist.return_value = [
            {"realized_pnl": 100},
            {"realized_pnl": 200},
            {"realized_pnl": -50},
            {"realized_pnl": -100},
        ]

        from agent.portfolio import get_portfolio_summary
        summary = get_portfolio_summary()
        # 2 wins / 4 trades = 50%
        assert summary["win_rate"] == 50.0
        assert summary["total_trades"] == 4

    @patch("agent.portfolio.get_open_positions")
    @patch("agent.portfolio.get_trade_history")
    def test_no_trades(self, mock_hist, mock_pos):
        mock_pos.return_value = []
        mock_hist.return_value = []

        from agent.portfolio import get_portfolio_summary
        summary = get_portfolio_summary()
        assert summary["win_rate"] == 0.0
        assert summary["total_trades"] == 0

    @patch("agent.portfolio.get_open_positions")
    @patch("agent.portfolio.get_trade_history")
    def test_can_add_position(self, mock_hist, mock_pos):
        mock_pos.return_value = []
        mock_hist.return_value = []

        from agent.portfolio import get_portfolio_summary
        summary = get_portfolio_summary()
        assert summary["can_add_position"] is True

    @patch("agent.portfolio.get_open_positions")
    @patch("agent.portfolio.get_trade_history")
    def test_invested_amounts(self, mock_hist, mock_pos):
        mock_pos.return_value = [
            {"shares": 10, "entry_price": 100, "stop_loss": 95, "unrealized_pnl": 50},
            {"shares": 20, "entry_price": 50, "stop_loss": 45, "unrealized_pnl": -20},
        ]
        mock_hist.return_value = []

        from agent.portfolio import get_portfolio_summary
        summary = get_portfolio_summary()
        # 10×100 + 20×50 = 2000
        assert summary["total_invested"] == 2000.0
        # risk: 10×5 + 20×5 = 150
        assert summary["total_risk"] == 150.0
        assert summary["total_unrealized_pnl"] == 30.0


# ═══════════════════════════════════════════════════════════════════════════
# can_open_position (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestCanOpenPosition:
    @patch("agent.portfolio.get_portfolio_summary")
    def test_reaches_max_positions(self, mock_summary):
        mock_summary.return_value = {
            "open_positions": 8,
            "total_invested": 5000,
            "risk_pct": 3.0,
        }
        from agent.portfolio import can_open_position, MAX_POSITIONS
        allowed, reason = can_open_position(1000)
        assert allowed is False
        assert "Max positions" in reason
