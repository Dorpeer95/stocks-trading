"""
Position sizing utilities — pure math functions.

Implements fixed-fractional risk sizing with portfolio constraints.
No API calls, no side effects.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (defaults — overridden by env vars in portfolio.py)
# ---------------------------------------------------------------------------
DEFAULT_RISK_PER_TRADE = 0.02      # 2% of portfolio
DEFAULT_MAX_POSITION_PCT = 0.25    # 25% max single position
DEFAULT_MAX_TOTAL_RISK = 0.06      # 6% max total portfolio risk
DEFAULT_CASH_RESERVE_PCT = 0.10    # 10% always in cash

# Realistic frictions — applied to every round trip so position size reflects
# the actual risk the account takes, not the idealised signal price.
# Default: 5 bps per side on Alpaca retail = 0.05% × 2 = 0.10% round trip.
SLIPPAGE_BPS_PER_SIDE = 5          # basis points; 100 bps = 1%
COMMISSION_PER_TRADE = 0.0         # Alpaca retail is zero-commission


@dataclass
class PositionPlan:
    """Result of a position sizing calculation."""
    shares: int
    position_value: float
    risk_amount: float
    risk_pct_portfolio: float
    reward_amount: float
    risk_reward_ratio: float
    position_pct_portfolio: float
    is_valid: bool
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Core sizing functions
# ---------------------------------------------------------------------------

def fixed_risk_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float = DEFAULT_RISK_PER_TRADE,
) -> Optional[PositionPlan]:
    """Calculate position size using fixed-fractional risk method.

    Parameters
    ----------
    portfolio_value : Total portfolio value in USD.
    entry_price : Planned entry price per share.
    stop_loss : Stop-loss price per share.
    risk_pct : Max risk as fraction of portfolio (default 0.02 = 2%).

    Returns
    -------
    PositionPlan with share count and risk details, or ``None`` on
    invalid inputs.

    Formula
    -------
    risk_amount = portfolio_value × risk_pct
    stop_distance = |entry_price - stop_loss|
    shares = floor(risk_amount / stop_distance)
    """
    if portfolio_value <= 0 or entry_price <= 0 or stop_loss <= 0:
        logger.warning("fixed_risk_size: invalid input (zero or negative)")
        return None

    if stop_loss >= entry_price:
        logger.warning(
            f"fixed_risk_size: stop_loss ({stop_loss}) >= entry ({entry_price})"
        )
        return None

    risk_amount = portfolio_value * risk_pct
    stop_distance = entry_price - stop_loss   # clean stop-to-entry distance

    if stop_distance <= 0:
        return None

    # Effective stop distance used for sizing includes realistic slippage on
    # entry + exit. Without this we systematically size larger than the real
    # risk budget because fills rarely land at the signal price. Commissions
    # (zero on Alpaca retail) are subtracted from the risk budget.
    slip_per_side = entry_price * (SLIPPAGE_BPS_PER_SIDE / 10_000.0)
    effective_stop_distance = stop_distance + 2.0 * slip_per_side
    effective_risk = max(risk_amount - 2.0 * COMMISSION_PER_TRADE, 0.0)
    shares = int(effective_risk / effective_stop_distance)

    if shares <= 0:
        return PositionPlan(
            shares=0,
            position_value=0,
            risk_amount=0,
            risk_pct_portfolio=0,
            reward_amount=0,
            risk_reward_ratio=0,
            position_pct_portfolio=0,
            is_valid=False,
            rejection_reason="Stop distance too large for risk budget",
        )

    position_value = shares * entry_price
    # risk_amount reported to the UI is the clean stop-to-entry risk, not the
    # slippage-inflated number — it's what the user sees on their broker.
    actual_risk = shares * stop_distance
    actual_risk_pct = actual_risk / portfolio_value
    position_pct = position_value / portfolio_value

    return PositionPlan(
        shares=shares,
        position_value=round(position_value, 2),
        risk_amount=round(actual_risk, 2),
        risk_pct_portfolio=round(actual_risk_pct, 4),
        reward_amount=0,  # set by caller with target
        risk_reward_ratio=0,  # set by caller with target
        position_pct_portfolio=round(position_pct, 4),
        is_valid=True,
    )


def calc_risk_reward(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    shares: int,
) -> Tuple[float, float, float]:
    """Calculate risk, reward, and risk/reward ratio.

    Returns
    -------
    (risk_amount, reward_amount, risk_reward_ratio)
    """
    risk_per_share = entry_price - stop_loss
    reward_per_share = target_price - entry_price

    risk_amount = risk_per_share * shares
    reward_amount = reward_per_share * shares

    if risk_amount <= 0:
        return (0.0, 0.0, 0.0)

    ratio = round(reward_amount / risk_amount, 2)
    return (round(risk_amount, 2), round(reward_amount, 2), ratio)


def full_position_plan(
    portfolio_value: float,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    risk_pct: float = DEFAULT_RISK_PER_TRADE,
) -> Optional[PositionPlan]:
    """Calculate a complete position plan with entry, stop, and target.

    Convenience wrapper that combines ``fixed_risk_size`` and
    ``calc_risk_reward``.
    """
    plan = fixed_risk_size(portfolio_value, entry_price, stop_loss, risk_pct)
    if plan is None or not plan.is_valid:
        return plan

    risk, reward, ratio = calc_risk_reward(
        entry_price, stop_loss, target_price, plan.shares
    )
    plan.risk_amount = risk
    plan.reward_amount = reward
    plan.risk_reward_ratio = ratio
    return plan


# ---------------------------------------------------------------------------
# Portfolio constraint checks
# ---------------------------------------------------------------------------

def max_position_check(
    position_value: float,
    portfolio_value: float,
    max_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> bool:
    """Check if a position exceeds the max single-position limit.

    Returns ``True`` if the position is within limits.
    """
    if portfolio_value <= 0:
        return False
    pct = position_value / portfolio_value
    return pct <= max_pct


def cap_to_max_position(
    entry_price: float,
    shares: int,
    portfolio_value: float,
    max_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> int:
    """Reduce shares if position would exceed max position size.

    Returns the (possibly reduced) share count.
    """
    max_value = portfolio_value * max_pct
    max_shares = int(max_value / entry_price) if entry_price > 0 else 0
    return min(shares, max_shares)


def cash_reserve_check(
    invested_total: float,
    new_position_value: float,
    portfolio_value: float,
    reserve_pct: float = DEFAULT_CASH_RESERVE_PCT,
) -> bool:
    """Check if adding a new position would violate the cash reserve.

    Returns ``True`` if enough cash remains.
    """
    if portfolio_value <= 0:
        return False
    after_investment = invested_total + new_position_value
    cash_remaining = portfolio_value - after_investment
    cash_pct = cash_remaining / portfolio_value
    return cash_pct >= reserve_pct


def total_risk_check(
    current_risk_pct: float,
    new_risk_pct: float,
    max_total_risk: float = DEFAULT_MAX_TOTAL_RISK,
) -> bool:
    """Check if adding a new position would exceed total portfolio risk.

    Returns ``True`` if within limits.
    """
    return (current_risk_pct + new_risk_pct) <= max_total_risk


# ---------------------------------------------------------------------------
# Kelly Criterion (optional / advanced)
# ---------------------------------------------------------------------------

def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Calculate Kelly Criterion optimal bet fraction.

    Parameters
    ----------
    win_rate : Historical win probability (0.0 - 1.0).
    avg_win : Average winning trade return (positive).
    avg_loss : Average losing trade return (positive, i.e. absolute loss).

    Returns
    -------
    Optimal fraction of portfolio to risk (0.0 - 1.0).
    Use half-Kelly (result * 0.5) for more conservative sizing.

    Notes
    -----
    Kelly = W - (1 - W) / R
    where W = win rate, R = avg_win / avg_loss
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    if not (0.0 <= win_rate <= 1.0):
        return 0.0

    r = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / r

    # Clamp to [0, 0.25] — never risk more than 25%
    return max(0.0, min(kelly, 0.25))


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly for more conservative sizing."""
    return kelly_criterion(win_rate, avg_win, avg_loss) * 0.5
