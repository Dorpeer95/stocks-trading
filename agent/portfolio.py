"""
Portfolio manager — tracks open positions, risk metrics, and exit logic.

Monitors intraday prices, updates trailing stops, detects exit conditions,
and provides portfolio-level risk summaries.
"""

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from agent.persistence import (
    get_open_positions,
    insert_position,
    update_position,
    insert_trade,
    insert_equity_snapshot,
    get_latest_snapshot,
    get_trade_history,
)
from utils.data_loader import fetch_price_data
from utils.position_sizing import (
    cash_reserve_check,
    max_position_check,
    total_risk_check,
    half_kelly,
    DEFAULT_CASH_RESERVE_PCT,
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_MAX_TOTAL_RISK,
)
from utils.alpaca_broker import execute_trade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORTFOLIO_VALUE = float(os.getenv("STOCKS_PORTFOLIO_VALUE", "10000"))
RISK_PER_TRADE = float(os.getenv("STOCKS_RISK_PER_TRADE", "0.02"))
MAX_POSITIONS = int(os.getenv("STOCKS_MAX_POSITIONS", "8"))
TRAILING_STOP_ATR_MULT = float(os.getenv("STOCKS_TRAILING_ATR_MULT", "2.0"))
MAX_HOLD_DAYS = int(os.getenv("STOCKS_MAX_HOLD_DAYS", "42"))  # 6 weeks
ENABLE_TRADING = os.getenv("STOCKS_ENABLE_TRADING", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Position monitoring
# ---------------------------------------------------------------------------

def update_positions_intraday() -> List[Dict[str, Any]]:
    """Update all open positions with current prices.

    Checks for exit signals (stop hit, target hit, time stop).

    Returns
    -------
    List of action dicts:
    ``{'ticker', 'action', 'reason', 'current_price', 'pnl', 'pnl_pct'}``.
    """
    positions = get_open_positions()
    if not positions:
        logger.debug("No open positions to monitor")
        return []

    actions: List[Dict[str, Any]] = []

    for pos in positions:
        ticker = pos.get("ticker", "???")
        try:
            df = fetch_price_data(ticker, period="1d", interval="5m")
            if df is None or df.empty:
                # Fallback to daily if intraday unavailable (pre/post market)
                df = fetch_price_data(ticker, period="5d", interval="1d")
            if df is None or df.empty:
                logger.warning(f"No price data for position {ticker}")
                continue

            current = float(df["Close"].iloc[-1])
            entry = pos.get("entry_price", current)
            stop = pos.get("stop_loss", 0)
            target = pos.get("target_price", 0)
            shares = pos.get("shares", 0)
            trailing = pos.get("trailing_stop", stop)

            # Calculate PnL
            pnl = (current - entry) * shares
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0

            # Days held
            entry_date = pos.get("entry_date")
            days_held = 0
            if entry_date:
                try:
                    ed = datetime.fromisoformat(str(entry_date)).date()
                    days_held = (date.today() - ed).days
                except Exception:
                    pass

            # Update position in DB
            updates = {
                "current_price": current,
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
            }

            # --- Exit checks ---
            action = None

            # 1. Stop loss hit
            if current <= stop:
                action = {
                    "ticker": ticker,
                    "action": "SELL — STOP HIT",
                    "reason": f"Price ${current:.2f} ≤ stop ${stop:.2f}",
                    "current_price": current,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1),
                    "urgent": "high",
                }
                if ENABLE_TRADING:
                    execute_trade(ticker, "sell", shares, "market")
                updates["status"] = "stopped_out"
                updates["exit_price"] = current
                updates["exit_date"] = date.today().isoformat()

            # 2. Trailing stop hit
            elif trailing and current <= trailing:
                action = {
                    "ticker": ticker,
                    "action": "SELL — TRAILING STOP",
                    "reason": f"Price ${current:.2f} ≤ trail ${trailing:.2f}",
                    "current_price": current,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1),
                    "urgent": "high",
                }
                if ENABLE_TRADING:
                    execute_trade(ticker, "sell", shares, "market")
                updates["status"] = "trailed_out"
                updates["exit_price"] = current
                updates["exit_date"] = date.today().isoformat()

            # 3. Target hit
            elif target and current >= target:
                action = {
                    "ticker": ticker,
                    "action": "SELL — TARGET HIT",
                    "reason": f"Price ${current:.2f} ≥ target ${target:.2f}",
                    "current_price": current,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1),
                    "urgent": "medium",
                }
                if ENABLE_TRADING:
                    execute_trade(ticker, "sell", shares, "market")
                updates["status"] = "target_hit"
                updates["exit_price"] = current
                updates["exit_date"] = date.today().isoformat()

            # 4. Time stop (held too long)
            elif days_held > MAX_HOLD_DAYS:
                action = {
                    "ticker": ticker,
                    "action": "REVIEW — TIME STOP",
                    "reason": f"Held {days_held} days (max {MAX_HOLD_DAYS})",
                    "current_price": current,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1),
                    "urgency": "low",
                }

            # 5. Update trailing stop (ratchet up only)
            else:
                new_trailing = _calc_trailing_stop(
                    current, entry, pos.get("atr_value")
                )
                if new_trailing and (not trailing or new_trailing > trailing):
                    updates["trailing_stop"] = round(new_trailing, 2)
                    logger.debug(
                        f"{ticker}: trailing stop raised to ${new_trailing:.2f}"
                    )

            # Persist updates
            pos_id = pos.get("id")
            if pos_id:
                update_position(pos_id, updates)

            # Record closed trade
            if action and updates.get("status") in (
                "stopped_out", "trailed_out", "target_hit"
            ):
                _record_trade(pos, current, updates["status"])

            if action:
                actions.append(action)

        except Exception as e:
            logger.error(f"Failed to update position {ticker}: {e}")

    return actions


def _calc_trailing_stop(
    current: float,
    entry: float,
    atr_value: Optional[float] = None,
) -> Optional[float]:
    """Calculate trailing stop based on ATR or percentage fallback."""
    if atr_value and atr_value > 0:
        return current - (TRAILING_STOP_ATR_MULT * atr_value)

    # Fallback: 5% below current price (only if in profit)
    if current > entry:
        return current * 0.95
    return None


def _record_trade(
    position: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
) -> None:
    """Record a closed trade in the trades table."""
    try:
        entry_price = position.get("entry_price", exit_price)
        shares = position.get("shares", 0)
        realized_pnl = (exit_price - entry_price) * shares

        trade = {
            "ticker": position.get("ticker"),
            "entry_date": position.get("entry_date"),
            "exit_date": date.today().isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(
                ((exit_price - entry_price) / entry_price * 100)
                if entry_price > 0 else 0,
                2,
            ),
            "exit_reason": exit_reason,
            "hold_days": position.get("days_held", 0),
        }
        insert_trade(trade)
    except Exception as e:
        logger.error(f"Failed to record trade: {e}")


# ---------------------------------------------------------------------------
# Portfolio-level risk metrics
# ---------------------------------------------------------------------------

def get_portfolio_summary() -> Dict[str, Any]:
    """Compute current portfolio risk metrics.

    Returns
    -------
    Dict with ``total_invested``, ``total_risk``, ``open_positions``,
    ``can_add_position``, ``cash_available``, ``total_pnl``,
    ``win_rate``, etc.
    """
    positions = get_open_positions()
    trade_history = get_trade_history(limit=100)

    total_invested = sum(
        (p.get("shares", 0) * p.get("entry_price", 0)) for p in positions
    )
    total_risk = sum(
        (p.get("shares", 0) * (p.get("entry_price", 0) - p.get("stop_loss", 0)))
        for p in positions
    )
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)

    cash_available = PORTFOLIO_VALUE - total_invested
    cash_pct = cash_available / PORTFOLIO_VALUE if PORTFOLIO_VALUE > 0 else 0
    risk_pct = total_risk / PORTFOLIO_VALUE if PORTFOLIO_VALUE > 0 else 0

    # Win rate from history
    wins = [t for t in trade_history if (t.get("pnl") or t.get("realized_pnl") or 0) > 0]
    losses = [t for t in trade_history if (t.get("pnl") or t.get("realized_pnl") or 0) <= 0]
    total_trades = len(wins) + len(losses)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    avg_win = (
        sum(t.get("pnl") or t.get("realized_pnl") or 0 for t in wins) / len(wins)
        if wins else 0
    )
    avg_loss = (
        abs(sum(t.get("pnl") or t.get("realized_pnl") or 0 for t in losses) / len(losses))
        if losses else 0
    )

    # Can we add another position?
    can_add = (
        len(positions) < MAX_POSITIONS
        and cash_pct > DEFAULT_CASH_RESERVE_PCT
        and risk_pct < DEFAULT_MAX_TOTAL_RISK
    )

    return {
        "portfolio_value": PORTFOLIO_VALUE,
        "total_invested": round(total_invested, 2),
        "cash_available": round(cash_available, 2),
        "cash_pct": round(cash_pct * 100, 1),
        "total_risk": round(total_risk, 2),
        "risk_pct": round(risk_pct * 100, 2),
        "open_positions": len(positions),
        "max_positions": MAX_POSITIONS,
        "can_add_position": can_add,
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "kelly_fraction": round(
            half_kelly(win_rate, avg_win, avg_loss), 4
        ) if total_trades >= 10 else None,
    }


def can_open_position(position_value: float) -> Tuple[bool, str]:
    """Check if a new position can be opened given current portfolio state.

    Returns (allowed, reason).
    """
    summary = get_portfolio_summary()

    if summary["open_positions"] >= MAX_POSITIONS:
        return False, f"Max positions ({MAX_POSITIONS}) reached"

    if not max_position_check(position_value, PORTFOLIO_VALUE):
        return False, f"Position ${position_value:,.0f} exceeds max position size"

    if not cash_reserve_check(
        summary["total_invested"], position_value, PORTFOLIO_VALUE
    ):
        return False, "Would violate cash reserve requirement"

    risk_pct = summary["risk_pct"] / 100
    new_risk = RISK_PER_TRADE
    if not total_risk_check(risk_pct, new_risk):
        return False, "Would exceed total portfolio risk limit"

    return True, "OK"


def execute_buy_opportunities(opportunities: List[Dict[str, Any]]) -> None:
    """Attempt to autonomously open positions for top opportunities using Alpaca."""
    open_positions = get_open_positions()
    existing_tickers = {p.get("ticker") for p in open_positions if p.get("ticker")}

    for opp in opportunities:
        position_value = opp.get("position_size_usd", 0)
        shares = opp.get("shares", 0)
        ticker = opp.get("ticker")
        
        # Fallback if keys are missing (e.g. from older data)
        if shares <= 0:
            position_value = opp.get("suggested_position_size", 0)
            shares = opp.get("suggested_shares", 0)
        
        if not ticker or shares <= 0:
            logger.debug(f"Skipping {ticker} because shares={shares}")
            continue
            
        if ticker in existing_tickers:
            logger.info(f"Skipping auto-buy for {ticker}: Position already exists")
            continue
            
        logger.info(f"Attempting auto-buy for {ticker}: {shares} shares (${position_value:.2f})")
            
        allowed, reason = can_open_position(position_value)
        if not allowed:
            logger.info(f"Skipping auto-buy for {ticker}: {reason}")
            continue
            
        # Execute trade
        order_receipt = None
        if ENABLE_TRADING:
             order_receipt = execute_trade(ticker, "buy", shares, "market")
        
        if order_receipt or (not ENABLE_TRADING and os.getenv("STOCKS_DRY_RUN", "false").lower() == "true"):
            # Reconstruct the position dictionary for database insertion
            entry_price = opp.get("entry_price") or (
                (opp.get("entry_price_low", 0) + opp.get("entry_price_high", 0)) / 2 
                if opp.get("entry_price_low") and opp.get("entry_price_high") else 0
            )

            position = {
                "ticker": ticker,
                "entry_price": entry_price,
                "shares": int(shares),
                "stop_loss": opp.get("stop_loss"),
                "target_price": opp.get("target_price"),
                "entry_date": date.today().isoformat(),
                "status": "open",
            }
            insert_position(position)
            existing_tickers.add(ticker)
            logger.info(f"Successfully processed position for {ticker} ({shares} shares) | Trading enabled: {ENABLE_TRADING}")
        else:
            logger.error(f"Failed to submit or skip Alpaca buy order for {ticker}")



# ---------------------------------------------------------------------------
# EOD summary
# ---------------------------------------------------------------------------

def generate_eod_summary() -> Dict[str, Any]:
    """Generate end-of-day portfolio summary for Telegram."""
    positions = get_open_positions()
    summary = get_portfolio_summary()

    position_details = []
    for pos in positions:
        position_details.append({
            "ticker": pos.get("ticker"),
            "current_price": pos.get("current_price"),
            "entry_price": pos.get("entry_price"),
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
            "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
            "days_held": pos.get("days_held", 0),
            "stop_loss": pos.get("stop_loss"),
            "target_price": pos.get("target_price"),
        })

    return {
        "date": date.today().isoformat(),
        "portfolio_value": summary["portfolio_value"],
        "total_invested": summary["total_invested"],
        "cash_available": summary["cash_available"],
        "total_unrealized_pnl": summary["total_unrealized_pnl"],
        "open_positions": summary["open_positions"],
        "positions": position_details,
        "risk_pct": summary["risk_pct"],
        "win_rate": summary["win_rate"],
    }


# ---------------------------------------------------------------------------
# Equity snapshot
# ---------------------------------------------------------------------------

def take_equity_snapshot() -> bool:
    """Take a weekly equity snapshot for tracking.

    Returns True on success.
    """
    summary = get_portfolio_summary()

    snapshot = {
        "snapshot_date": date.today().isoformat(),
        "total_value": summary["portfolio_value"] + summary["total_unrealized_pnl"],
        "cash": summary["cash_available"],
        "invested": summary["total_invested"],
        "unrealized_pnl": summary["total_unrealized_pnl"],
        "positions_count": summary["open_positions"],
        "win_rate": summary["win_rate"],
    }

    return insert_equity_snapshot(snapshot)
