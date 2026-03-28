"""
Simple strategy backtester — validates VCP/swing entry logic against historical data.

Usage
─────
  python scripts/backtest_strategy.py [--tickers AAPL MSFT] [--years 2]

How it works
─────────────
1. Downloads N years of daily data for each ticker via yfinance.
2. Simulates the VCP entry signal: price breaks above the 20-day high on
   above-average volume (simplified VCP proxy).
3. Applies stop = 7% below entry, target = 21% above entry (3:1 RR).
4. Tracks each trade: entry, exit (stop/target/time), P&L, hold days.
5. Reports win rate, expectancy, avg hold time, and profit factor.

Note: This is a simplified VCP proxy, not a full CANSLIM screener backtest.
The goal is to validate that the entry/exit logic produces a positive
expectancy before running on live capital.

No extra API keys required — uses yfinance only.
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "AVGO", "CRM",
]
STOP_PCT = 0.07       # 7% stop below entry
TARGET_PCT = 0.21     # 21% target above entry (3:1 RR)
MAX_HOLD_DAYS = 42    # 6 weeks time stop
MIN_RS_LOOKBACK = 63  # ~3 months for RS calculation
VOLUME_MULT = 1.3     # entry volume must be 1.3× 20-day avg


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Download adjusted OHLCV data for a ticker."""
    start = (date.today() - timedelta(days=years * 365 + 60)).isoformat()
    try:
        df = yf.download(
            ticker,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
        if df is None or len(df) < 100:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.warning(f"{ticker}: download failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Signal generation (VCP proxy)
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True on days with a VCP-like breakout.

    Signal: today's close > 20-day high AND volume > 1.3× 20-day avg volume.
    We skip the first 25 days for warmup.
    """
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    rolling_high = close.shift(1).rolling(20).max()   # high of PRIOR 20 days
    avg_vol = volume.shift(1).rolling(20).mean()

    signal = (close > rolling_high) & (volume > avg_vol * VOLUME_MULT)
    signal.iloc[:25] = False  # warmup
    return signal


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_trades(df: pd.DataFrame, signals: pd.Series) -> List[Dict[str, Any]]:
    """Walk forward through signals, open/close trades.

    Rules:
      • One trade at a time (no overlapping).
      • Entry at next open after signal day.
      • Exit: first day close hits stop OR target, or MAX_HOLD_DAYS reached.
    """
    close = df["Close"].squeeze().values.astype(float)
    open_ = df["Open"].squeeze().values.astype(float)
    dates = df.index.to_list()

    trades: List[Dict[str, Any]] = []
    in_trade = False
    entry_price = stop = target = entry_idx = 0

    for i in range(1, len(df)):
        if not in_trade and signals.iloc[i - 1]:
            # Enter at open on day i
            entry_price = float(open_[i])
            if entry_price <= 0:
                continue
            stop = entry_price * (1 - STOP_PCT)
            target = entry_price * (1 + TARGET_PCT)
            entry_idx = i
            in_trade = True
            continue

        if in_trade:
            current_close = float(close[i])
            hold_days = i - entry_idx

            # Exit conditions
            exit_price = None
            exit_reason = None

            if current_close <= stop:
                exit_price = stop          # assume fills at stop
                exit_reason = "stop"
            elif current_close >= target:
                exit_price = target        # assume fills at target
                exit_reason = "target"
            elif hold_days >= MAX_HOLD_DAYS:
                exit_price = current_close
                exit_reason = "time_stop"

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[i],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                })
                in_trade = False

    return trades


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------

def compute_stats(trades: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
    """Compute summary statistics for a ticker's trade history."""
    if not trades:
        return {"ticker": ticker, "trades": 0}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if (avg_loss and losses) else float("inf")
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    avg_hold = np.mean([t["hold_days"] for t in trades])

    return {
        "ticker": ticker,
        "trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(-avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy, 2),
        "avg_hold_days": round(avg_hold, 1),
        "stops": sum(1 for t in trades if t["exit_reason"] == "stop"),
        "targets": sum(1 for t in trades if t["exit_reason"] == "target"),
        "time_stops": sum(1 for t in trades if t["exit_reason"] == "time_stop"),
    }


def print_summary(all_stats: List[Dict[str, Any]]) -> None:
    """Print backtest results table."""
    valid = [s for s in all_stats if s.get("trades", 0) >= 3]
    if not valid:
        print("No tickers had ≥3 trades.")
        return

    print("\n" + "=" * 80)
    print(f"  BACKTEST RESULTS — stop={STOP_PCT*100:.0f}%  target={TARGET_PCT*100:.0f}%  "
          f"maxhold={MAX_HOLD_DAYS}d")
    print("=" * 80)
    header = f"{'Ticker':<8} {'Trades':>6} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} "
    header += f"{'PF':>6} {'Expect':>8} {'AvgHold':>8}"
    print(header)
    print("-" * 80)

    for s in sorted(valid, key=lambda x: x.get("expectancy_pct", 0), reverse=True):
        print(
            f"{s['ticker']:<8} {s['trades']:>6} {s['win_rate_pct']:>7.1f}% "
            f"{s['avg_win_pct']:>7.1f}% {s['avg_loss_pct']:>7.1f}% "
            f"{s['profit_factor']:>6.2f} {s['expectancy_pct']:>7.2f}% "
            f"{s['avg_hold_days']:>7.1f}d"
        )

    # Aggregate
    total_trades = sum(s["trades"] for s in valid)
    avg_wr = np.mean([s["win_rate_pct"] for s in valid])
    avg_exp = np.mean([s["expectancy_pct"] for s in valid])
    avg_pf = np.mean([s["profit_factor"] for s in valid if s["profit_factor"] < 100])
    print("-" * 80)
    print(
        f"{'AGGREGATE':<8} {total_trades:>6} {avg_wr:>7.1f}% "
        f"{'':>8} {'':>8} {avg_pf:>6.2f} {avg_exp:>7.2f}%"
    )
    print("=" * 80)
    print(f"\nPositive expectancy tickers: "
          f"{[s['ticker'] for s in valid if s['expectancy_pct'] > 0]}")
    print(f"Negative expectancy tickers: "
          f"{[s['ticker'] for s in valid if s['expectancy_pct'] <= 0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest VCP strategy")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Ticker symbols to backtest"
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of history to use (default: 2)"
    )
    args = parser.parse_args()

    print(f"Running backtest on {len(args.tickers)} tickers over {args.years}y …")
    all_stats = []

    for ticker in args.tickers:
        df = fetch_data(ticker, args.years)
        if df is None or df.empty:
            print(f"  {ticker}: no data — skipping")
            continue

        signals = generate_signals(df)
        trades = simulate_trades(df, signals)
        stats = compute_stats(trades, ticker)
        all_stats.append(stats)

        if trades:
            print(
                f"  {ticker}: {stats['trades']} trades  "
                f"WR={stats['win_rate_pct']:.0f}%  "
                f"Expect={stats['expectancy_pct']:+.1f}%"
            )
        else:
            print(f"  {ticker}: 0 trades generated")

    print_summary(all_stats)


if __name__ == "__main__":
    main()
