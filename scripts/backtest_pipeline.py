"""
Pipeline backtester — walks the real weekly cycle over 3+ years of history.

Imports the live ``agent.scanner``, ``agent.scorer``, and
``agent.portfolio_manager`` unmodified. Data-loader and persistence
calls are monkey-patched (see ``backtest_replay.py``) so the historical
slice at each Sunday is exactly what the live bot would have seen, and
portfolio state lives in memory.

Fills:
- ENTRY at Monday open of the week AFTER the scan, times (1 + slippage).
- EXIT from ``run_portfolio_cycle`` (confidence-based SELL) at Monday
  open of the following week, times (1 - slippage).
- EXIT on STOP hit: first daily bar whose low ≤ stop_loss; fill at stop.
- EXIT on TARGET hit: first daily bar whose high ≥ target; fill at target.
- TIME-STOP after 6 weeks with no stop/target hit: Friday close × (1 - slippage).

Mac-only. The full S&P 500 run touches 500 tickers × ~3 years of daily
bars which peaks around 2-3 GB of working memory during indicator
computation.

CLI
---
    python scripts/backtest_pipeline.py \
        --years 3 \
        --out reports/backtest.json \
        --capital 10000 \
        --cache-dir .backtest_cache

Acceptance gate (printed at the end of the markdown report):
- Profit factor ≥ 1.50
- Max drawdown ≤ 20%
- ≥ 80 simulated trades
- Beats SPY buy-and-hold on absolute return
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo root importable when run as ``python scripts/backtest_pipeline.py``
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_replay import (  # noqa: E402
    HistoricalDataCache,
    InMemoryPortfolioDB,
    assert_mac_or_ci,
    patch_data_loader,
    patch_external_noops,
    patch_persistence,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SLIPPAGE_BPS_PER_SIDE = 5       # matches utils/position_sizing.py
TIME_STOP_WEEKS = 6             # exit if still open after 6 weeks
POSITION_RISK_PCT = 0.02        # 2% risk per trade (matches live default)
MACRO_TICKERS = ["^VIX", "SPY", "^TNX", "DX-Y.NYB", "CL=F", "GC=F"]
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLY", "XLC", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB"]


# ═══════════════════════════════════════════════════════════════════════════
# Trade + equity tracking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimTrade:
    """A single simulated swing trade from entry to exit."""
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    target_price: float
    entry_confidence: float
    setup_type: str
    regime_at_entry: str
    # Filled at exit
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop' | 'target' | 'signal' | 'time_stop'
    pnl_dollars: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_days: Optional[int] = None


@dataclass
class BacktestState:
    initial_capital: float
    cash: float
    open_trades: Dict[str, SimTrade] = field(default_factory=dict)  # ticker → trade
    closed_trades: List[SimTrade] = field(default_factory=list)
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)  # (date_iso, equity)
    weekly_scan_count: int = 0

    def equity(self, marks: Dict[str, float]) -> float:
        """Mark-to-market equity = cash + sum(open position market values)."""
        mv = 0.0
        for t, tr in self.open_trades.items():
            price = marks.get(t, tr.entry_price)
            mv += price * tr.shares
        return self.cash + mv


# ═══════════════════════════════════════════════════════════════════════════
# Fill simulator — walks daily bars to find stop / target / exit triggers
# ═══════════════════════════════════════════════════════════════════════════

def _apply_slippage(price: float, side: str) -> float:
    """Buy fills worse by slippage; sell fills also worse by slippage."""
    slip = price * (SLIPPAGE_BPS_PER_SIDE / 10_000.0)
    return price + slip if side == "buy" else price - slip


def simulate_exits(
    state: BacktestState,
    cache: HistoricalDataCache,
    week_end: date,
    signal_exits: List[str],
) -> None:
    """Walk daily bars for the next ~5 trading days, close any trades that hit.

    Priority when both stop and target are in the same bar: stop fills
    first (conservative assumption since we cannot know intra-bar order).
    """
    for ticker in list(state.open_trades.keys()):
        trade = state.open_trades[ticker]
        forward = cache.get_next_bars(ticker, week_end, days=5)
        if forward is None or forward.empty:
            continue

        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None
        exit_date: Optional[date] = None

        for idx, row in forward.iterrows():
            bar_date = pd.Timestamp(idx).date() if not isinstance(idx, date) else idx
            low = float(row["Low"])
            high = float(row["High"])

            # Stop wins ties with target (conservative)
            if low <= trade.stop_loss:
                exit_price = _apply_slippage(trade.stop_loss, "sell")
                exit_reason = "stop"
                exit_date = bar_date
                break
            if high >= trade.target_price:
                exit_price = _apply_slippage(trade.target_price, "sell")
                exit_reason = "target"
                exit_date = bar_date
                break

        # Signal-based exit (confidence drop from portfolio_manager).
        # Only trigger if no intra-week stop/target hit.
        if exit_price is None and ticker in signal_exits:
            # Exit at Monday open of the following week, which is
            # forward.iloc[0] if forward has at least one bar.
            bar0 = forward.iloc[0]
            exit_price = _apply_slippage(float(bar0["Open"]), "sell")
            exit_reason = "signal"
            exit_date = pd.Timestamp(forward.index[0]).date()

        # Time stop: if we've been in the trade > TIME_STOP_WEEKS and no
        # stop/target/signal triggered, close at Friday's close this week.
        if exit_price is None:
            entry_d = datetime.fromisoformat(trade.entry_date).date()
            weeks_held = (week_end - entry_d).days / 7
            if weeks_held >= TIME_STOP_WEEKS:
                last = forward.iloc[-1]
                exit_price = _apply_slippage(float(last["Close"]), "sell")
                exit_reason = "time_stop"
                exit_date = pd.Timestamp(forward.index[-1]).date()

        if exit_price is None:
            continue  # trade remains open

        # Close the trade
        pnl_per = exit_price - trade.entry_price
        trade.exit_date = exit_date.isoformat() if exit_date else week_end.isoformat()
        trade.exit_price = round(exit_price, 2)
        trade.exit_reason = exit_reason
        trade.pnl_dollars = round(pnl_per * trade.shares, 2)
        trade.pnl_pct = round(pnl_per / trade.entry_price * 100, 2)
        entry_d = datetime.fromisoformat(trade.entry_date).date()
        trade.hold_days = (exit_date - entry_d).days if exit_date else 0

        state.cash += exit_price * trade.shares
        state.closed_trades.append(trade)
        del state.open_trades[ticker]
        logger.info(
            f"EXIT  {ticker} @ ${exit_price:.2f} ({exit_reason}) "
            f"P&L ${trade.pnl_dollars:+.0f} ({trade.pnl_pct:+.1f}%) "
            f"held {trade.hold_days}d"
        )


def simulate_entries(
    state: BacktestState,
    cache: HistoricalDataCache,
    new_entries: List[Dict[str, Any]],
    week_end: date,
    regime: str,
) -> None:
    """Open positions for ``new_entries`` at Monday's open of the next week.

    Position size uses the same fixed-fractional risk model as live.
    Skipped if we can't fit the entry under the remaining cash.
    """
    # Import live sizing so backtest & live stay in sync
    from utils.position_sizing import fixed_risk_size

    for entry in new_entries:
        ticker = entry["ticker"]
        if ticker in state.open_trades:
            continue

        forward = cache.get_next_bars(ticker, week_end, days=2)
        if forward is None or forward.empty:
            logger.debug(f"No next-week bars for {ticker}, skipping fill")
            continue

        entry_px = _apply_slippage(float(forward.iloc[0]["Open"]), "buy")
        stop = entry.get("stop_loss")
        target = entry.get("target_price")
        if not stop or not target or stop >= entry_px:
            logger.debug(f"{ticker}: bad stop/target, skipping")
            continue

        equity_now = state.equity({})
        plan = fixed_risk_size(equity_now, entry_px, float(stop), risk_pct=POSITION_RISK_PCT)
        if plan is None or not plan.is_valid or plan.shares <= 0:
            continue

        cost = plan.shares * entry_px
        if cost > state.cash:
            # Can't afford; scale down to available cash
            max_shares = int(state.cash / entry_px)
            if max_shares <= 0:
                continue
            plan.shares = max_shares
            cost = plan.shares * entry_px

        state.cash -= cost
        entry_date = pd.Timestamp(forward.index[0]).date()
        trade = SimTrade(
            ticker=ticker,
            entry_date=entry_date.isoformat(),
            entry_price=round(entry_px, 2),
            shares=plan.shares,
            stop_loss=round(float(stop), 2),
            target_price=round(float(target), 2),
            entry_confidence=float(entry.get("confidence", 0)),
            setup_type=str(entry.get("setup_type", "")),
            regime_at_entry=regime,
        )
        state.open_trades[ticker] = trade
        logger.info(
            f"ENTRY {ticker} {plan.shares}sh @ ${entry_px:.2f} "
            f"stop ${stop:.2f} tgt ${target:.2f} conf {entry.get('confidence')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Weekly loop
# ═══════════════════════════════════════════════════════════════════════════

def run_weekly_cycle(
    state: BacktestState,
    db: InMemoryPortfolioDB,
    cache: HistoricalDataCache,
    week_end: date,
) -> None:
    """Execute one simulated Sunday scan → score → portfolio cycle."""
    from agent.scanner import scan_universe  # re-import after patching
    from agent.scorer import score_candidates
    from agent.portfolio_manager import run_portfolio_cycle

    # Re-patch data_loader for THIS week's cutoff
    unpatch_dl = patch_data_loader(cache, week_end)
    try:
        try:
            scan = scan_universe()
        except Exception as e:
            logger.warning(f"scan_universe failed on {week_end}: {e}")
            return

        candidates = scan.get("candidates", [])
        if not candidates:
            logger.debug(f"No candidates on {week_end}")
        # Build a fake regime dict from SPY movement (keeps macro bias sane)
        regime = _simple_regime(cache, week_end)

        # Signal age map — our fake DB is persistent across weeks so this
        # gives portfolio_manager what it needs for consecutive-week entry.
        signal_age_map = {
            c["ticker"]: db.get_consecutive_strong_weeks(c["ticker"], 72)
            for c in candidates if c.get("ticker")
        }

        scored = score_candidates(
            candidates,
            regime=regime,
            fetch_extras=False,  # fundamentals are not point-in-time available
            ml_predictions_map=None,  # ML skipped for baseline backtest
            sentiment_map=None,
            signal_age_map=signal_age_map,
        )

        # Record signal_history so consecutive-week tracking works next week
        db.insert_signal_history([
            {
                "ticker": s["ticker"],
                "scan_date": week_end.isoformat(),
                "confidence": s.get("confidence", 0),
                "sub_scores": s.get("sub_scores", {}),
                "setup_type": s.get("setup_type", ""),
                "passed_scan": True,
            }
            for s in scored if s.get("ticker")
        ])

        diff = run_portfolio_cycle(scored, scan_date=week_end.isoformat())
        state.weekly_scan_count += 1

        # Simulate fills
        signal_exit_tickers = [e for e in diff.get("exits", []) if isinstance(e, str)]
        simulate_exits(state, cache, week_end, signal_exit_tickers)

        new_entries = diff.get("new_entries", [])
        simulate_entries(state, cache, new_entries, week_end, regime.get("regime", "unknown"))

        # Snapshot equity at Friday close (last day of next-week window)
        marks = _mark_to_market(state, cache, week_end)
        equity = state.equity(marks)
        state.equity_curve.append((week_end.isoformat(), round(equity, 2)))
    finally:
        unpatch_dl()


def _simple_regime(cache: HistoricalDataCache, as_of: date) -> Dict[str, Any]:
    """Cheap regime classifier from SPY trend + VIX level.

    Real bot uses ``agent.regime_engine`` but that pulls live macro data.
    During backtest we reconstruct a minimal regime dict good enough for
    score_macro() to produce a stable bias.
    """
    spy = cache.get_window("SPY", as_of, "3mo")
    vix = cache.get_window("^VIX", as_of, "3mo")
    if spy is None or len(spy) < 20:
        return {"regime": "unknown", "vix": None, "spy_trend": None}

    spy_now = float(spy["Close"].iloc[-1])
    spy_20d = float(spy["Close"].iloc[-20])
    trend_pct = (spy_now - spy_20d) / spy_20d * 100
    vix_now = float(vix["Close"].iloc[-1]) if vix is not None and len(vix) else None

    if trend_pct > 2 and (vix_now is None or vix_now < 20):
        label = "risk_on"
    elif trend_pct < -3 or (vix_now is not None and vix_now > 25):
        label = "risk_off"
    else:
        label = "neutral"
    return {"regime": label, "vix": vix_now, "spy_trend": round(trend_pct, 2)}


def _mark_to_market(
    state: BacktestState,
    cache: HistoricalDataCache,
    as_of: date,
) -> Dict[str, float]:
    marks: Dict[str, float] = {}
    for t in state.open_trades:
        df = cache.get_window(t, as_of + timedelta(days=7), "1mo")
        if df is not None and len(df):
            marks[t] = float(df["Close"].iloc[-1])
    return marks


def iter_sundays(start: date, end: date) -> List[date]:
    """All Sundays between ``start`` and ``end`` inclusive."""
    days = (end - start).days
    out: List[date] = []
    for offset in range(days + 1):
        d = start + timedelta(days=offset)
        if d.weekday() == 6:  # Sunday
            out.append(d)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def _profit_factor(trades: List[SimTrade]) -> float:
    wins = sum(t.pnl_dollars for t in trades if (t.pnl_dollars or 0) > 0)
    losses = -sum(t.pnl_dollars for t in trades if (t.pnl_dollars or 0) < 0)
    return round(wins / losses, 2) if losses > 0 else float("inf") if wins > 0 else 0.0


def _max_drawdown(equity_curve: List[Tuple[str, float]]) -> float:
    peak = 0.0
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    return round(max_dd * 100, 2)


def _sharpe(equity_curve: List[Tuple[str, float]]) -> float:
    """Weekly Sharpe; annualize by multiplying by sqrt(52)."""
    if len(equity_curve) < 2:
        return 0.0
    returns: List[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1][1]
        cur = equity_curve[i][1]
        if prev <= 0:
            continue
        returns.append((cur - prev) / prev)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = var ** 0.5
    if std == 0:
        return 0.0
    return round((mean / std) * (52 ** 0.5), 2)


def _group_stats(trades: List[SimTrade], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[SimTrade]] = {}
    for t in trades:
        g = getattr(t, key) or "unknown"
        groups.setdefault(g, []).append(t)
    out: Dict[str, Dict[str, Any]] = {}
    for g, trs in groups.items():
        n = len(trs)
        wins = [t for t in trs if (t.pnl_dollars or 0) > 0]
        out[g] = {
            "trades": n,
            "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0.0,
            "profit_factor": _profit_factor(trs),
            "avg_pnl_pct": round(sum((t.pnl_pct or 0) for t in trs) / n, 2) if n else 0.0,
        }
    return out


def _spy_buy_and_hold(
    cache: HistoricalDataCache,
    start: date,
    end: date,
    capital: float,
) -> Optional[float]:
    df_start = cache.get_window("SPY", start, "1mo")
    df_end = cache.get_window("SPY", end, "1mo")
    if df_start is None or df_end is None or df_start.empty or df_end.empty:
        return None
    open_px = float(df_start["Close"].iloc[-1])
    close_px = float(df_end["Close"].iloc[-1])
    shares = capital // open_px
    return round(shares * close_px + (capital - shares * open_px), 2)


def build_report(
    state: BacktestState,
    cache: HistoricalDataCache,
    start: date,
    end: date,
) -> Dict[str, Any]:
    trades = state.closed_trades
    n = len(trades)
    wins = [t for t in trades if (t.pnl_dollars or 0) > 0]
    losses = [t for t in trades if (t.pnl_dollars or 0) < 0]

    final_equity = state.equity_curve[-1][1] if state.equity_curve else state.initial_capital
    spy_final = _spy_buy_and_hold(cache, start, end, state.initial_capital)

    overall = {
        "trades": n,
        "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0.0,
        "avg_win_pct": round(sum(t.pnl_pct for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(sum(t.pnl_pct for t in losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": _profit_factor(trades),
        "expectancy_pct": round(sum((t.pnl_pct or 0) for t in trades) / n, 2) if n else 0.0,
        "avg_hold_days": round(sum((t.hold_days or 0) for t in trades) / n, 1) if n else 0.0,
        "max_drawdown_pct": _max_drawdown(state.equity_curve),
        "sharpe_annual": _sharpe(state.equity_curve),
        "final_equity": round(final_equity, 2),
        "initial_capital": state.initial_capital,
        "return_pct": round((final_equity - state.initial_capital) / state.initial_capital * 100, 2),
        "spy_buy_hold_final": spy_final,
        "beats_spy": (spy_final is not None and final_equity > spy_final),
        "weekly_scans": state.weekly_scan_count,
    }

    # Gate check
    gate = {
        "profit_factor_ge_1_5": overall["profit_factor"] >= 1.5,
        "max_drawdown_le_20": overall["max_drawdown_pct"] <= 20.0,
        "trades_ge_80": n >= 80,
        "beats_spy": overall["beats_spy"],
    }
    gate["passed"] = all(gate.values())

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "overall": overall,
        "by_regime": _group_stats(trades, "regime_at_entry"),
        "by_setup": _group_stats(trades, "setup_type"),
        "by_exit_reason": _group_stats(trades, "exit_reason"),
        "gate": gate,
    }


def write_reports(
    report: Dict[str, Any],
    state: BacktestState,
    out_json: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Wrote {out_json}")

    # Trade CSV next to the JSON
    csv_path = out_json.with_name(out_json.stem + "_trades.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry_date", "entry_price", "shares",
            "stop_loss", "target_price", "entry_confidence", "setup_type",
            "regime_at_entry", "exit_date", "exit_price", "exit_reason",
            "pnl_dollars", "pnl_pct", "hold_days",
        ])
        for t in state.closed_trades:
            w.writerow([
                t.ticker, t.entry_date, t.entry_price, t.shares,
                t.stop_loss, t.target_price, t.entry_confidence, t.setup_type,
                t.regime_at_entry, t.exit_date, t.exit_price, t.exit_reason,
                t.pnl_dollars, t.pnl_pct, t.hold_days,
            ])
    logger.info(f"Wrote {csv_path}")

    # Markdown summary
    md_path = out_json.with_suffix(".md")
    md_path.write_text(_render_markdown(report))
    logger.info(f"Wrote {md_path}")


def _render_markdown(r: Dict[str, Any]) -> str:
    o = r["overall"]
    gate = r["gate"]
    lines: List[str] = []
    lines.append(f"# Pipeline Backtest — {r['window']['start']} → {r['window']['end']}")
    lines.append("")
    lines.append(f"**Weekly scans:** {o['weekly_scans']}  |  **Initial capital:** ${o['initial_capital']:,.0f}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Trades executed: **{o['trades']}**")
    lines.append(f"- Win rate: **{o['win_rate_pct']}%**")
    lines.append(f"- Avg win: **{o['avg_win_pct']:+.2f}%**  |  Avg loss: **{o['avg_loss_pct']:+.2f}%**")
    lines.append(f"- **Profit factor: {o['profit_factor']}**   (gate: ≥ 1.50)")
    lines.append(f"- Expectancy per trade: **{o['expectancy_pct']:+.2f}%**")
    lines.append(f"- Avg hold: **{o['avg_hold_days']} days**")
    lines.append(f"- **Max drawdown: {o['max_drawdown_pct']}%**   (gate: ≤ 20%)")
    lines.append(f"- Sharpe (annualized): **{o['sharpe_annual']}**")
    lines.append(f"- Final equity: **${o['final_equity']:,.2f}** "
                 f"(return {o['return_pct']:+.2f}%)")
    if o.get("spy_buy_hold_final") is not None:
        beat = "✅" if o["beats_spy"] else "❌"
        lines.append(f"- SPY buy-and-hold: **${o['spy_buy_hold_final']:,.2f}**  {beat}")
    lines.append("")
    lines.append("## By regime")
    lines.append("")
    for name, s in r["by_regime"].items():
        lines.append(f"- **{name}**: {s['trades']} trades, WR {s['win_rate_pct']}%, "
                     f"PF {s['profit_factor']}, avg {s['avg_pnl_pct']:+.2f}%")
    lines.append("")
    lines.append("## By setup")
    lines.append("")
    for name, s in r["by_setup"].items():
        lines.append(f"- **{name or 'unspecified'}**: {s['trades']} trades, WR {s['win_rate_pct']}%, "
                     f"PF {s['profit_factor']}, avg {s['avg_pnl_pct']:+.2f}%")
    lines.append("")
    lines.append("## By exit reason")
    lines.append("")
    for name, s in r["by_exit_reason"].items():
        lines.append(f"- **{name}**: {s['trades']} ({s['win_rate_pct']}% wins, avg {s['avg_pnl_pct']:+.2f}%)")
    lines.append("")
    lines.append("## Gate status")
    lines.append("")
    check = lambda b: "✅" if b else "❌"
    lines.append(f"- {check(gate['profit_factor_ge_1_5'])} Profit factor ≥ 1.50")
    lines.append(f"- {check(gate['max_drawdown_le_20'])} Max drawdown ≤ 20%")
    lines.append(f"- {check(gate['trades_ge_80'])} Trades ≥ 80")
    lines.append(f"- {check(gate['beats_spy'])} Beats SPY buy-and-hold")
    lines.append("")
    if gate["passed"]:
        lines.append("### ✅ PASSED — proceed to Phase 2 (LLM decision layer)")
    else:
        lines.append("### ❌ FAILED — run Phase 1.5 iteration playbook before proceeding")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, default=3,
                        help="Lookback years from --end (default: 3)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date ISO (default: today)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date ISO (overrides --years)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--out", type=str, default="reports/backtest.json")
    parser.add_argument("--cache-dir", type=str, default=".backtest_cache")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--force-refetch", action="store_true",
                        help="Ignore on-disk cache and re-download data")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    assert_mac_or_ci()

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.years * 365)
    logger.info(f"Backtest window: {start} → {end}")

    cache = HistoricalDataCache(as_of_start=start, as_of_end=end)
    cache_dir = Path(args.cache_dir)

    loaded = False if args.force_refetch else cache.load_from_parquet(cache_dir)
    if not loaded:
        from utils.data_loader import fetch_sp500_list
        universe = fetch_sp500_list() or []
        if not universe:
            logger.error("Could not fetch S&P 500 universe")
            return 1
        cache.prefetch(universe, MACRO_TICKERS, SECTOR_ETFS)
        cache.save_to_parquet(cache_dir)

    db = InMemoryPortfolioDB()
    state = BacktestState(initial_capital=args.capital, cash=args.capital)

    unpatch_ext = patch_external_noops()
    unpatch_db = patch_persistence(db)
    try:
        sundays = iter_sundays(start, end)
        logger.info(f"Simulating {len(sundays)} weekly scans")
        for i, sunday in enumerate(sundays, 1):
            if i % 10 == 0:
                logger.info(f"  …week {i}/{len(sundays)}  cash=${state.cash:,.0f}  "
                           f"open={len(state.open_trades)}  closed={len(state.closed_trades)}")
            run_weekly_cycle(state, db, cache, sunday)

        # Final flush: close any still-open trades at last available price
        _flush_open(state, cache, end)

        report = build_report(state, cache, start, end)
        out_path = Path(args.out)
        write_reports(report, state, out_path)

        print()
        print(_render_markdown(report))
        return 0 if report["gate"]["passed"] else 2
    finally:
        unpatch_db()
        unpatch_ext()


def _flush_open(state: BacktestState, cache: HistoricalDataCache, end: date) -> None:
    for ticker in list(state.open_trades.keys()):
        trade = state.open_trades[ticker]
        df = cache.get_window(ticker, end, "1mo")
        if df is None or df.empty:
            del state.open_trades[ticker]
            continue
        price = _apply_slippage(float(df["Close"].iloc[-1]), "sell")
        pnl_per = price - trade.entry_price
        trade.exit_date = end.isoformat()
        trade.exit_price = round(price, 2)
        trade.exit_reason = "end_of_backtest"
        trade.pnl_dollars = round(pnl_per * trade.shares, 2)
        trade.pnl_pct = round(pnl_per / trade.entry_price * 100, 2)
        entry_d = datetime.fromisoformat(trade.entry_date).date()
        trade.hold_days = (end - entry_d).days
        state.cash += price * trade.shares
        state.closed_trades.append(trade)
        del state.open_trades[ticker]


# Imported lazily in helpers above
import pandas as pd  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
