"""
Unit tests for utils/position_sizing.py.

Tests the fixed-fractional risk model, portfolio constraints (max position,
cash reserve, total risk), Kelly criterion, and edge cases.
"""

import pytest

from utils.position_sizing import (
    PositionPlan,
    fixed_risk_size,
    calc_risk_reward,
    full_position_plan,
    max_position_check,
    cap_to_max_position,
    cash_reserve_check,
    total_risk_check,
    kelly_criterion,
    half_kelly,
)


# ═══════════════════════════════════════════════════════════════════════════
# fixed_risk_size
# ═══════════════════════════════════════════════════════════════════════════

class TestFixedRiskSize:
    """With 5bps per-side slippage: $10,000 × 2% = $200 risk, effective stop
    distance = $5 + 2×($100×0.0005) = $5.10 → 39 shares.
    """

    def test_development_plan_example(self):
        plan = fixed_risk_size(10_000, 100.0, 95.0)
        assert plan is not None
        assert plan.is_valid
        assert plan.shares == 39

    def test_risk_amount_matches(self):
        plan = fixed_risk_size(10_000, 100.0, 95.0)
        # 39 shares × $5 stop-to-entry = $195 reported risk (slippage is
        # already baked into the share count, not the displayed risk).
        assert plan.risk_amount == 195.0

    def test_position_value(self):
        plan = fixed_risk_size(10_000, 100.0, 95.0)
        assert plan.position_value == 3900.0

    def test_position_pct(self):
        plan = fixed_risk_size(10_000, 100.0, 95.0)
        assert plan.position_pct_portfolio == 0.39

    def test_custom_risk_pct(self):
        # 1% risk = $100, effective distance $5.10 → 19 shares
        plan = fixed_risk_size(10_000, 100.0, 95.0, risk_pct=0.01)
        assert plan.shares == 19

    def test_large_stop_distance(self):
        # $200 risk, entry $100, stop $10 → effective distance $90.10 → 2 shares
        plan = fixed_risk_size(10_000, 100.0, 10.0)
        assert plan.shares == 2

    def test_stop_too_far_zero_shares(self):
        # $200 risk, entry $100, stop $0.01 → effective distance ≈ $100.09 → 1 share
        plan = fixed_risk_size(10_000, 100.0, 0.01)
        assert plan.shares == 1

    def test_returns_none_on_zero_portfolio(self):
        assert fixed_risk_size(0, 100.0, 95.0) is None

    def test_returns_none_on_negative_price(self):
        assert fixed_risk_size(10_000, -100.0, 95.0) is None

    def test_returns_none_on_stop_above_entry(self):
        assert fixed_risk_size(10_000, 100.0, 105.0) is None

    def test_returns_none_on_stop_equals_entry(self):
        assert fixed_risk_size(10_000, 100.0, 100.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# calc_risk_reward
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcRiskReward:
    def test_basic_2_to_1(self):
        # Entry $100, stop $95, target $110, 40 shares
        risk, reward, ratio = calc_risk_reward(100.0, 95.0, 110.0, 40)
        assert risk == 200.0   # 40 × $5
        assert reward == 400.0  # 40 × $10
        assert ratio == 2.0

    def test_1_to_1(self):
        risk, reward, ratio = calc_risk_reward(100.0, 95.0, 105.0, 40)
        assert ratio == 1.0

    def test_zero_risk(self):
        risk, reward, ratio = calc_risk_reward(100.0, 100.0, 110.0, 40)
        assert ratio == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# full_position_plan
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPositionPlan:
    def test_complete_plan(self):
        plan = full_position_plan(10_000, 100.0, 95.0, 110.0)
        assert plan is not None
        assert plan.is_valid
        assert plan.shares == 39
        assert plan.risk_amount == 195.0
        assert plan.reward_amount == 390.0
        assert plan.risk_reward_ratio == 2.0

    def test_returns_none_on_bad_input(self):
        assert full_position_plan(0, 100.0, 95.0, 110.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio constraint checks
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxPositionCheck:
    def test_within_limit(self):
        # $2,000 position in $10,000 portfolio = 20% < 25%
        assert max_position_check(2000, 10_000) is True

    def test_at_limit(self):
        # $2,500 = exactly 25%
        assert max_position_check(2500, 10_000) is True

    def test_exceeds_limit(self):
        # $3,000 = 30% > 25%
        assert max_position_check(3000, 10_000) is False

    def test_zero_portfolio(self):
        assert max_position_check(100, 0) is False


class TestCapToMaxPosition:
    def test_caps_shares(self):
        # Max 25% of $10,000 = $2,500, at $100/share → 25 shares max
        capped = cap_to_max_position(100.0, 40, 10_000)
        assert capped == 25

    def test_no_cap_needed(self):
        # $100 × 10 = $1,000, which is 10% < 25%
        capped = cap_to_max_position(100.0, 10, 10_000)
        assert capped == 10


class TestCashReserveCheck:
    def test_enough_cash(self):
        # Invested $5,000 + new $2,000 = $7,000 → $3,000 cash = 30% > 10%
        assert cash_reserve_check(5000, 2000, 10_000) is True

    def test_violates_reserve(self):
        # Invested $8,000 + new $1,500 = $9,500 → $500 cash = 5% < 10%
        assert cash_reserve_check(8000, 1500, 10_000) is False

    def test_exactly_at_reserve(self):
        # Invested $7,000 + new $2,000 = $9,000 → $1,000 cash = 10%
        assert cash_reserve_check(7000, 2000, 10_000) is True

    def test_zero_portfolio(self):
        assert cash_reserve_check(0, 100, 0) is False


class TestTotalRiskCheck:
    def test_within_limit(self):
        assert total_risk_check(0.03, 0.02) is True   # 5% < 6%

    def test_exceeds_limit(self):
        assert total_risk_check(0.05, 0.02) is False   # 7% > 6%

    def test_at_limit(self):
        assert total_risk_check(0.04, 0.02) is True    # 6% = 6%


# ═══════════════════════════════════════════════════════════════════════════
# Kelly Criterion
# ═══════════════════════════════════════════════════════════════════════════

class TestKellyCriterion:
    def test_positive_edge(self):
        # 60% win rate, avg win $2, avg loss $1
        # Kelly = 0.6 - 0.4/2 = 0.4
        result = kelly_criterion(0.6, 2.0, 1.0)
        assert abs(result - 0.25) < 0.01  # clamped to max 0.25

    def test_no_edge(self):
        # 50% win rate, avg win $1, avg loss $1
        # Kelly = 0.5 - 0.5/1 = 0.0
        result = kelly_criterion(0.5, 1.0, 1.0)
        assert result == 0.0

    def test_negative_edge(self):
        # 40% win rate, 1:1 → Kelly = 0.4 - 0.6/1.0 = -0.2 → clamped to 0
        result = kelly_criterion(0.4, 1.0, 1.0)
        assert result == 0.0

    def test_invalid_inputs(self):
        assert kelly_criterion(0.5, 0, 1.0) == 0.0
        assert kelly_criterion(0.5, 1.0, 0) == 0.0
        assert kelly_criterion(-0.1, 1.0, 1.0) == 0.0
        assert kelly_criterion(1.1, 1.0, 1.0) == 0.0


class TestHalfKelly:
    def test_is_half(self):
        full = kelly_criterion(0.6, 2.0, 1.0)
        half = half_kelly(0.6, 2.0, 1.0)
        assert abs(half - full * 0.5) < 0.0001
