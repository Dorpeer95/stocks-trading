"""
Backtest replay helpers — point-in-time data cache + in-memory persistence.

Imported by ``scripts/backtest_pipeline.py``. The main goal is to let the
live ``agent.scanner`` / ``agent.scorer`` / ``agent.portfolio_manager``
code run *unmodified* during a backtest by:

1. Pre-fetching 3+ years of daily bars for the full universe once, then
   serving point-in-time slices (``get_window(ticker, end_date, period)``)
   so every call the scanner makes sees only data that was available on
   the simulated Sunday.
2. Replacing every Supabase CRUD call in ``agent.persistence`` with an
   in-memory dict-backed implementation that behaves the same way for
   the narrow set of functions the weekly cycle actually uses.

Nothing in this file is imported by the live bot — it only runs on the
dev Mac during backtests. Memory budget here is ~2-3 GB working set; do
not run on the DO $6/mo box.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Historical data cache — point-in-time slicing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HistoricalDataCache:
    """Pre-fetch 3+ years of daily bars once, serve point-in-time slices.

    The cache stores un-sliced full-history DataFrames keyed by ticker.
    ``get_window(ticker, end_date, period)`` returns a copy truncated to
    rows with index ≤ ``end_date`` and at least ``period`` days back, so
    the real scanner sees the exact bars it would have seen on that
    historical Sunday.
    """

    as_of_start: date
    as_of_end: date
    _bars: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _universe: List[Dict[str, str]] = field(default_factory=list)
    _macro_bars: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Yfinance period strings → rough day counts (matches data_loader)
    _PERIOD_DAYS = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 365 * 5,
    }

    def load_from_parquet(self, cache_dir: Path) -> bool:
        """Load pre-fetched bars from an on-disk parquet cache.

        Expected layout: ``cache_dir/{ticker}.parquet`` plus
        ``cache_dir/_universe.csv`` and ``cache_dir/_macro/{ticker}.parquet``.
        Returns False if the cache is missing — caller should then run
        ``prefetch`` to build it.
        """
        if not cache_dir.exists():
            return False

        uni_csv = cache_dir / "_universe.csv"
        if not uni_csv.exists():
            return False
        self._universe = pd.read_csv(uni_csv).to_dict(orient="records")

        for f in cache_dir.glob("*.parquet"):
            ticker = f.stem
            try:
                self._bars[ticker] = pd.read_parquet(f)
            except Exception as e:
                logger.warning(f"Skipping corrupt parquet {f}: {e}")

        macro_dir = cache_dir / "_macro"
        if macro_dir.exists():
            for f in macro_dir.glob("*.parquet"):
                self._macro_bars[f.stem] = pd.read_parquet(f)

        logger.info(
            f"Loaded historical cache: {len(self._bars)} tickers, "
            f"{len(self._universe)} universe rows, "
            f"{len(self._macro_bars)} macro series"
        )
        return len(self._bars) > 0

    def save_to_parquet(self, cache_dir: Path) -> None:
        """Persist the cache so subsequent backtest runs skip the download."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ticker, df in self._bars.items():
            try:
                df.to_parquet(cache_dir / f"{ticker}.parquet")
            except Exception as e:
                logger.warning(f"Failed to write {ticker}.parquet: {e}")

        pd.DataFrame(self._universe).to_csv(cache_dir / "_universe.csv", index=False)

        macro_dir = cache_dir / "_macro"
        macro_dir.mkdir(exist_ok=True)
        for ticker, df in self._macro_bars.items():
            df.to_parquet(macro_dir / f"{ticker}.parquet")

        logger.info(f"Saved historical cache to {cache_dir}")

    def prefetch(
        self,
        universe: List[Dict[str, str]],
        macro_tickers: List[str],
        sector_etfs: List[str],
    ) -> None:
        """Populate the cache by calling the live data loader once.

        Fetches a window large enough that the deepest historical Sunday
        still has 1y of lookback available. Imports yfinance directly to
        avoid hitting the live bot's Alpaca client during the backtest.
        """
        import yfinance as yf

        start = self.as_of_start - timedelta(days=500)  # 1y lookback buffer
        end = self.as_of_end + timedelta(days=1)

        self._universe = list(universe)
        tickers = [u["ticker"] for u in universe]
        all_symbols = tickers + macro_tickers + sector_etfs

        logger.info(
            f"Prefetching {len(all_symbols)} symbols from {start} to {end} "
            f"(this takes ~10-20 min for S&P 500)"
        )

        batch_size = 40
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i : i + batch_size]
            try:
                raw = yf.download(
                    batch, start=start.isoformat(), end=end.isoformat(),
                    group_by="ticker", auto_adjust=True, progress=False, threads=True,
                )
            except Exception as e:
                logger.warning(f"Batch {i}: yfinance download failed: {e}")
                continue

            for t in batch:
                try:
                    df = raw[t] if len(batch) > 1 else raw
                    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                    df = df[keep].dropna(subset=keep)
                    if df.empty:
                        continue
                    if t in macro_tickers:
                        self._macro_bars[t] = df
                    else:
                        self._bars[t] = df
                except Exception as e:
                    logger.debug(f"Skipping {t}: {e}")

            time.sleep(1.0)  # gentle rate limit

        logger.info(
            f"Prefetch complete: {len(self._bars)} tickers, "
            f"{len(self._macro_bars)} macro"
        )

    # ── Point-in-time slice ──────────────────────────────────────────────

    def get_window(
        self,
        ticker: str,
        end_date: date,
        period: str = "1y",
    ) -> Optional[pd.DataFrame]:
        """Return the slice of ``ticker`` bars ending on or before ``end_date``.

        Mimics ``utils.data_loader.fetch_price_data(ticker, period)`` as it
        would have run on ``end_date``.
        """
        full = self._bars.get(ticker) or self._macro_bars.get(ticker)
        if full is None:
            return None

        # pandas index is tz-aware datetime; normalize for comparison
        cutoff = pd.Timestamp(end_date)
        # Coerce index to date for comparison — handles tz-aware parquets too
        try:
            idx_dates = pd.to_datetime(full.index).tz_localize(None)
        except (AttributeError, TypeError):
            idx_dates = pd.to_datetime(full.index)

        mask = idx_dates <= cutoff
        sliced = full.loc[mask]
        if sliced.empty:
            return None

        days = self._PERIOD_DAYS.get(period, 365)
        # Keep the last ``days`` bars
        return sliced.tail(max(days // 1, 30)).copy()

    def get_batch(
        self,
        tickers: List[str],
        end_date: date,
        period: str = "1y",
    ) -> Dict[str, pd.DataFrame]:
        """Batch version of ``get_window`` for ``fetch_batch_prices``."""
        out: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = self.get_window(t, end_date, period)
            if df is not None and not df.empty:
                out[t] = df
        return out

    def universe(self) -> List[Dict[str, str]]:
        return list(self._universe)

    def get_next_bars(
        self,
        ticker: str,
        start_date: date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Bars STRICTLY AFTER ``start_date`` — used by the fill simulator."""
        full = self._bars.get(ticker)
        if full is None:
            return None
        cutoff = pd.Timestamp(start_date)
        try:
            idx_dates = pd.to_datetime(full.index).tz_localize(None)
        except (AttributeError, TypeError):
            idx_dates = pd.to_datetime(full.index)
        mask = idx_dates > cutoff
        forward = full.loc[mask].head(days)
        return forward if not forward.empty else None


# ═══════════════════════════════════════════════════════════════════════════
# In-memory portfolio DB — drop-in replacement for agent.persistence
# ═══════════════════════════════════════════════════════════════════════════

class InMemoryPortfolioDB:
    """Replaces the Supabase CRUD calls the weekly cycle actually uses.

    Only the functions exercised by ``run_portfolio_cycle`` +
    ``score_candidates`` are covered. Anything else logs a debug line
    and returns an empty/default value so the live code path doesn't
    crash when it tries to write to a table we don't care about in the
    backtest.
    """

    def __init__(self) -> None:
        self.holdings: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[Dict[str, Any]] = []
        self.portfolio_log: List[Dict[str, Any]] = []

    # ── portfolio_holdings ──────────────────────────────────────────────

    def get_portfolio_holdings(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = list(self.holdings.values())
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return rows

    def upsert_portfolio_holding(self, holding: Dict[str, Any]) -> bool:
        ticker = holding.get("ticker")
        if not ticker:
            return False
        existing = self.holdings.get(ticker, {})
        existing.update(holding)
        self.holdings[ticker] = existing
        return True

    def remove_portfolio_holding(self, ticker: str) -> bool:
        return self.holdings.pop(ticker, None) is not None

    def get_portfolio_holding(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.holdings.get(ticker)

    # ── signal_history ──────────────────────────────────────────────────

    def insert_signal_history(self, records: List[Dict[str, Any]]) -> bool:
        self.signal_history.extend(records)
        return True

    def get_signal_history(self, ticker: str, weeks: int = 4) -> List[Dict[str, Any]]:
        matches = [r for r in self.signal_history if r.get("ticker") == ticker]
        matches.sort(key=lambda r: r.get("scan_date", ""), reverse=True)
        return matches[:weeks]

    def get_consecutive_strong_weeks(self, ticker: str, min_confidence: float) -> int:
        """Count most recent consecutive weeks above ``min_confidence``."""
        matches = [r for r in self.signal_history if r.get("ticker") == ticker]
        matches.sort(key=lambda r: r.get("scan_date", ""), reverse=True)
        count = 0
        for r in matches:
            if (r.get("confidence") or 0) >= min_confidence:
                count += 1
            else:
                break
        return count

    # ── portfolio_log ───────────────────────────────────────────────────

    def log_portfolio_action(
        self,
        ticker: str,
        action: str,
        reason: str = "",
        confidence: Optional[float] = None,
        prev_status: Optional[str] = None,
        new_status: Optional[str] = None,
    ) -> bool:
        self.portfolio_log.append({
            "ticker": ticker,
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "prev_status": prev_status,
            "new_status": new_status,
        })
        return True

    # ── No-ops (tables we don't care about in backtest) ────────────────

    def upsert_universe(self, stocks: List[Dict[str, Any]]) -> bool:
        return True

    def insert_opportunity(self, opp: Dict[str, Any]) -> bool:
        return True

    def insert_opportunities(self, opps: List[Dict[str, Any]]) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Monkey-patching helpers
# ═══════════════════════════════════════════════════════════════════════════

def patch_data_loader(cache: HistoricalDataCache, as_of: date) -> Callable[[], None]:
    """Redirect every ``utils.data_loader`` read to the historical cache.

    Returns an ``unpatch`` callable the caller should run at teardown.
    """
    import utils.data_loader as dl

    originals = {
        "fetch_price_data": dl.fetch_price_data,
        "fetch_batch_prices": dl.fetch_batch_prices,
        "fetch_sp500_list": dl.fetch_sp500_list,
        "fetch_macro_data": dl.fetch_macro_data,
        "fetch_fundamentals": dl.fetch_fundamentals,
    }

    def _price(ticker: str, period: str = "1y", interval: str = "1d"):
        return cache.get_window(ticker, as_of, period)

    def _batch(tickers: List[str], period: str = "1y", interval: str = "1d"):
        return cache.get_batch(tickers, as_of, period)

    def _sp500():
        return cache.universe()

    def _macro(period: str = "3mo"):
        # Scorer's macro_bias doesn't need high fidelity in backtest;
        # derive a minimal dict from cached VIX + SPY so score_macro works.
        out: Dict[str, Dict[str, float]] = {}
        for key, sym in {"vix": "^VIX", "spy": "SPY"}.items():
            df = cache.get_window(sym, as_of, period)
            if df is None or len(df) < 2:
                continue
            current = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg = round(((current - prev) / prev) * 100, 2) if prev != 0 else 0.0
            out[key] = {"current": round(current, 2), "prev_close": round(prev, 2), "change_pct": chg}
        return out or None

    def _fundamentals(ticker: str):
        # Skip during backtest — live fundamentals don't represent history.
        # Returning None lets scorer fall back to technical-only signals.
        return None

    dl.fetch_price_data = _price
    dl.fetch_batch_prices = _batch
    dl.fetch_sp500_list = _sp500
    dl.fetch_macro_data = _macro
    dl.fetch_fundamentals = _fundamentals

    # Rebind at every module that imported these names at module top
    # (``from utils.data_loader import fetch_batch_prices`` captures the
    # reference at import time, bypassing our attribute patch on ``dl``).
    rebind_site_originals: List[Tuple[Any, str, Any]] = []

    def _rebind(module: Any) -> None:
        for name, fn in {
            "fetch_price_data": _price,
            "fetch_batch_prices": _batch,
            "fetch_sp500_list": _sp500,
            "fetch_macro_data": _macro,
            "fetch_fundamentals": _fundamentals,
        }.items():
            if hasattr(module, name):
                rebind_site_originals.append((module, name, getattr(module, name)))
                setattr(module, name, fn)

    for mod_path in [
        "agent.scanner",
        "agent.scorer",
        "agent.portfolio",
        "agent.portfolio_manager",
        "agent.events",
        "agent.regime_engine",
        "utils.sectors",
    ]:
        try:
            m = __import__(mod_path, fromlist=["_"])
            _rebind(m)
        except Exception:
            pass

    def unpatch() -> None:
        for k, v in originals.items():
            setattr(dl, k, v)
        for module, name, original in rebind_site_originals:
            setattr(module, name, original)

    return unpatch


def patch_persistence(db: InMemoryPortfolioDB) -> Callable[[], None]:
    """Redirect persistence CRUD to the in-memory fake DB.

    Patches *both* ``agent.persistence`` AND the already-imported symbols
    inside ``agent.portfolio_manager`` / ``agent.scorer``, since those
    modules bound the function names at import time.
    """
    import agent.persistence as ap
    import agent.portfolio_manager as pm

    originals: Dict[Tuple[Any, str], Any] = {}

    def _bind(module: Any, name: str, fn: Any) -> None:
        if hasattr(module, name):
            originals[(module, name)] = getattr(module, name)
            setattr(module, name, fn)

    for name, fn in {
        "get_portfolio_holdings": db.get_portfolio_holdings,
        "get_portfolio_holding": db.get_portfolio_holding,
        "upsert_portfolio_holding": db.upsert_portfolio_holding,
        "remove_portfolio_holding": db.remove_portfolio_holding,
        "insert_signal_history": db.insert_signal_history,
        "get_signal_history": db.get_signal_history,
        "get_consecutive_strong_weeks": db.get_consecutive_strong_weeks,
        "log_portfolio_action": db.log_portfolio_action,
        "upsert_universe": db.upsert_universe,
        "insert_opportunity": db.insert_opportunity,
        "insert_opportunities": db.insert_opportunities,
    }.items():
        _bind(ap, name, fn)
        _bind(pm, name, fn)

    # Scanner also calls upsert_universe directly
    import agent.scanner as scanner_mod
    _bind(scanner_mod, "upsert_universe", db.upsert_universe)

    def unpatch() -> None:
        for (module, name), original in originals.items():
            setattr(module, name, original)

    return unpatch


def patch_external_noops() -> Callable[[], None]:
    """Silence sentiment/insider/earnings external APIs during backtest.

    These don't offer point-in-time history and would pollute the result
    with live-as-of-today data. Returning neutral defaults lets the
    scorer fall back to technical + RS + ML signals only.

    Also zeroes out ``agent.scanner.BATCH_DELAY`` — the live scanner
    sleeps 5s between yfinance batches to stay under rate limits, but
    our patched ``fetch_batch_prices`` returns from memory so the sleep
    is pure wall-clock overhead (~5h per 150-week backtest).
    """
    patches: List[Tuple[Any, str, Any]] = []

    try:
        import agent.scanner as scanner_mod
        patches.append((scanner_mod, "BATCH_DELAY", scanner_mod.BATCH_DELAY))
        scanner_mod.BATCH_DELAY = 0.0
    except Exception:
        pass

    try:
        import utils.sentiment as sent
        patches.append((sent, "batch_sentiment", getattr(sent, "batch_sentiment", None)))
        sent.batch_sentiment = lambda *a, **kw: {}
    except Exception:
        pass

    try:
        import utils.insider as ins
        patches.append((ins, "get_insider_signal", getattr(ins, "get_insider_signal", None)))
        ins.get_insider_signal = lambda *a, **kw: {"score": 50.0, "net_buying": None, "recent_transactions": 0}
    except Exception:
        pass

    try:
        import utils.earnings as earn
        patches.append((earn, "earnings_risk_flag", getattr(earn, "earnings_risk_flag", None)))
        patches.append((earn, "fetch_earnings_history", getattr(earn, "fetch_earnings_history", None)))
        patches.append((earn, "calc_beat_streak", getattr(earn, "calc_beat_streak", None)))
        earn.earnings_risk_flag = lambda *a, **kw: {"flag": False, "days_to_earnings": None}
        earn.fetch_earnings_history = lambda *a, **kw: []
        earn.calc_beat_streak = lambda *a, **kw: 0
    except Exception:
        pass

    def unpatch() -> None:
        for mod, name, original in patches:
            if original is not None:
                setattr(mod, name, original)

    return unpatch


# ═══════════════════════════════════════════════════════════════════════════
# Memory guard — refuse to run on the DO server
# ═══════════════════════════════════════════════════════════════════════════

def assert_mac_or_ci() -> None:
    """Bail out if running on the DO server or anywhere else RAM-constrained.

    The backtester's working set is ~2-3 GB during feature computation,
    which would OOM the $6/mo tier instantly.
    """
    if os.getenv("STOCKS_BACKTEST_ALLOW_SERVER") == "true":
        return  # explicit override for CI
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_gb < 4.0:
            raise SystemExit(
                f"Refusing to run backtest on {total_gb:.1f} GB host. "
                "This is Mac-only per the plan — the working set will OOM "
                "small instances. Set STOCKS_BACKTEST_ALLOW_SERVER=true "
                "to override (not recommended)."
            )
    except ImportError:
        logger.warning("psutil not installed; skipping memory check")
