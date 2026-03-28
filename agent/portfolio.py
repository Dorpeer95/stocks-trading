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
    remove_portfolio_holding,
    log_portfolio_action,
    get_portfolio_holding,
)
from utils.data_loader import fetch_price_data
from utils.helpers import safe_float, safe_int
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
            entry = safe_float(pos.get("entry_price"), current)
            stop = safe_float(pos.get("stop_loss"))
            target = safe_float(pos.get("target_price"))
            shares = safe_int(pos.get("shares"))
            trailing = safe_float(pos.get("trailing_stop"), stop)

            if not stop:
                logger.warning(f"{ticker}: no stop_loss set — stop check disabled")
            if not target:
                logger.warning(f"{ticker}: no target_price set — target check disabled")

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

            # MAE/MFE water marks (ratchet only — never move backward)
            current_high = safe_float(pos.get("high_water_mark"), entry)
            current_low = safe_float(pos.get("low_water_mark"), entry)
            if current > current_high:
                updates["high_water_mark"] = round(current, 2)
            if current_low == 0 or current < current_low:
                updates["low_water_mark"] = round(current, 2)

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
                    "urgent": "low",
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

            # Record closed trade + sync portfolio_holdings
            if action and updates.get("status") in (
                "stopped_out", "trailed_out", "target_hit"
            ):
                # Fetch holding enrichment data BEFORE removal (2.4 postmortem prep)
                holding_data = get_portfolio_holding(ticker)
                _record_trade(pos, current, updates["status"], holding_data)
                # Remove from portfolio_holdings so weekly cycle doesn't
                # try to re-score a ghost holding (critical desync fix)
                remove_portfolio_holding(ticker)
                log_portfolio_action(
                    ticker, "REMOVED",
                    reason=f"Intraday exit: {updates['status']}",
                    prev_status="active", new_status="closed",
                )

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
    holding_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a closed trade in the trades table.

    Parameters
    ----------
    holding_data : portfolio_holdings row fetched before removal — provides
                   entry_confidence, sub_scores_at_entry, setup_type, and
                   regime context for Layer 4 postmortem recording.
    """
    try:
        entry_price = safe_float(position.get("entry_price"), exit_price)
        shares = safe_int(position.get("shares"))
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
            "hold_days": safe_int(position.get("days_held")),
        }

        # MAE/MFE from water marks (set by update_positions_intraday)
        high_mark = safe_float(position.get("high_water_mark"), entry_price)
        low_mark = safe_float(position.get("low_water_mark"), entry_price)
        if entry_price > 0:
            trade["mfe_pct"] = round((high_mark - entry_price) / entry_price * 100, 2)
            trade["mae_pct"] = round((entry_price - low_mark) / entry_price * 100, 2)

        # Enrich with portfolio holding context (Layer 2.4 / Layer 4 prep)
        if holding_data:
            trade["entry_confidence"] = holding_data.get("entry_confidence")
            trade["sub_scores_at_entry"] = holding_data.get("sub_scores")
            trade["setup_type"] = holding_data.get("setup_type")

        insert_trade(trade)

        # Trigger postmortem recording (non-blocking)
        try:
            from agent.postmortem import record_postmortem
            record_postmortem(trade, position, holding_data)
        except Exception as pm_e:
            logger.debug(f"Postmortem record skipped: {pm_e}")
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
        safe_float(p.get("shares")) * safe_float(p.get("entry_price")) for p in positions
    )
    total_risk = sum(
        safe_float(p.get("shares")) * (safe_float(p.get("entry_price")) - safe_float(p.get("stop_loss")))
        for p in positions
    )
    total_unrealized = sum(safe_float(p.get("unrealized_pnl")) for p in positions)

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


def get_exposure_summary() -> Dict[str, Any]:
    """Analyse portfolio concentration by sector and single-name exposure.

    Returns
    -------
    Dict with:
      ``sector_weights``       — {sector: pct_of_invested}
      ``top_name_pct``         — single largest position as % of portfolio
      ``concentrated_sectors`` — sectors >40% of invested
      ``concentration_risk``   — "low" | "medium" | "high"
      ``warnings``             — list of human-readable risk flags
    """
    positions = get_open_positions()
    if not positions:
        return {
            "sector_weights": {},
            "top_name_pct": 0.0,
            "concentrated_sectors": [],
            "concentration_risk": "low",
            "warnings": [],
        }

    total_invested = sum(
        safe_float(p.get("shares")) * safe_float(p.get("entry_price")) for p in positions
    )
    if total_invested == 0:
        return {
            "sector_weights": {},
            "top_name_pct": 0.0,
            "concentrated_sectors": [],
            "concentration_risk": "low",
            "warnings": [],
        }

    # Sector aggregation — uses sector field if available, else "Unknown"
    sector_totals: Dict[str, float] = {}
    name_weights: Dict[str, float] = {}

    for pos in positions:
        invested = safe_float(pos.get("shares")) * safe_float(pos.get("entry_price"))
        sector = pos.get("sector") or "Unknown"
        sector_totals[sector] = sector_totals.get(sector, 0.0) + invested
        name_weights[pos.get("ticker", "?")] = invested / total_invested * 100

    sector_weights = {
        k: round(v / total_invested * 100, 1)
        for k, v in sector_totals.items()
    }

    concentrated = [s for s, w in sector_weights.items() if w >= 40.0]
    top_name_pct = round(max(name_weights.values()), 1) if name_weights else 0.0

    warnings = []
    if concentrated:
        warnings.append(f"Sector concentration ≥40%: {', '.join(concentrated)}")
    if top_name_pct > 25.0:
        warnings.append(f"Single-name risk: top position = {top_name_pct:.0f}% of portfolio")
    if len(positions) <= 2 and total_invested / PORTFOLIO_VALUE > 0.3:
        warnings.append("Under-diversified: only 1-2 positions with >30% deployed")

    if len(warnings) >= 2 or (concentrated and top_name_pct > 25):
        risk_level = "high"
    elif warnings:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "sector_weights": sector_weights,
        "top_name_pct": top_name_pct,
        "concentrated_sectors": concentrated,
        "concentration_risk": risk_level,
        "warnings": warnings,
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
                if opp.get("entry_price_low") and opp.get("entry_price_high") else None
            )

            if not entry_price:
                logger.warning(f"Skipping position for {ticker}: no valid entry price")
                continue

            stop_loss = safe_float(opp.get("stop_loss")) or None
            target_price = safe_float(opp.get("target_price")) or None

            if not stop_loss or not target_price:
                logger.warning(f"{ticker}: missing stop_loss or target_price — position may not exit properly")

            # Pull sector from portfolio_holdings (written during weekly scan)
            _holding = get_portfolio_holding(ticker)
            _sector = _holding.get("sector") if _holding else None

            position = {
                "ticker": ticker,
                "entry_price": entry_price,
                "shares": int(shares),
                "stop_loss": stop_loss,
                "target_price": target_price,
                "entry_date": date.today().isoformat(),
                "status": "open",
                "sector": _sector,
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
            "current_price": safe_float(pos.get("current_price")),
            "entry_price": safe_float(pos.get("entry_price")),
            "unrealized_pnl": safe_float(pos.get("unrealized_pnl")),
            "unrealized_pnl_pct": safe_float(pos.get("unrealized_pnl_pct")),
            "days_held": safe_int(pos.get("days_held")),
            "stop_loss": safe_float(pos.get("stop_loss")),
            "target_price": safe_float(pos.get("target_price")),
        })

    return {
        "date": date.today().isoformat(),
        "portfolio_value": summary["portfolio_value"],
        "total_invested": summary["total_invested"],
        "cash_available": summary["cash_available"],
        "total_pnl": summary["total_unrealized_pnl"],
        "total_pnl_pct": round(
            summary["total_unrealized_pnl"] / summary["portfolio_value"] * 100, 1
        ) if summary["portfolio_value"] > 0 else 0,
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
