"""
Sector relative-strength analysis.

Computes relative strength (RS) vs SPY and vs sector ETF using
weighted multi-timeframe returns.  Pure math on DataFrames.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe weights for composite RS
# ---------------------------------------------------------------------------
#   4-week   = most recent momentum
#   13-week  = medium-term trend
#   26-week  = longer-term trend
RS_WEIGHTS: Dict[int, float] = {
    20: 0.4,    # ~4 weeks  (trading days)
    65: 0.35,   # ~13 weeks
    130: 0.25,  # ~26 weeks
}


# ---------------------------------------------------------------------------
# Core RS calculations
# ---------------------------------------------------------------------------

def _pct_return(series: pd.Series, lookback: int) -> Optional[float]:
    """Percentage return over *lookback* periods.  Returns ``None`` on failure."""
    if series is None or len(series) < lookback + 1:
        return None
    try:
        old = float(series.iloc[-(lookback + 1)])
        new = float(series.iloc[-1])
        if old == 0 or np.isnan(old) or np.isnan(new):
            return None
        return ((new - old) / old) * 100.0
    except Exception:
        return None


def calc_rs_vs_benchmark(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    weights: Optional[Dict[int, float]] = None,
) -> Optional[float]:
    """Composite relative strength of a stock vs a benchmark.

    Returns
    -------
    Weighted average of ``(stock_return - benchmark_return)``
    across multiple lookback windows.  Positive = outperforming.
    ``None`` on insufficient data.
    """
    w = weights or RS_WEIGHTS
    total_weight = 0.0
    weighted_rs = 0.0

    for lookback, weight in w.items():
        stock_ret = _pct_return(stock_close, lookback)
        bench_ret = _pct_return(benchmark_close, lookback)
        if stock_ret is None or bench_ret is None:
            continue
        weighted_rs += (stock_ret - bench_ret) * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return round(weighted_rs / total_weight, 2)


def calc_rs_percentile(
    rs_value: float,
    all_rs_values: List[float],
) -> float:
    """Convert an RS value to a percentile rank (0-100) within the universe."""
    if not all_rs_values:
        return 50.0
    below = sum(1 for v in all_rs_values if v < rs_value)
    return round((below / len(all_rs_values)) * 100, 1)


# ---------------------------------------------------------------------------
# Batch RS for the universe
# ---------------------------------------------------------------------------

def compute_universe_rs(
    price_data: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    sector_etf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ticker_sectors: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Compute RS metrics for every ticker in the universe.

    Parameters
    ----------
    price_data : ticker → OHLCV DataFrame.
    spy_df : SPY OHLCV DataFrame (benchmark).
    sector_etf_data : sector_etf_ticker → OHLCV DataFrame (optional).
    ticker_sectors : ticker → GICS sector name (optional).

    Returns
    -------
    List of dicts with keys: ``ticker``, ``rs_vs_spy``, ``rs_vs_sector``,
    ``rs_percentile``, ``momentum_4w``, ``momentum_13w``, ``momentum_26w``.
    """
    if spy_df is None or spy_df.empty:
        logger.error("compute_universe_rs: SPY data is required")
        return []

    spy_close = spy_df["Close"]
    results: List[Dict[str, Any]] = []
    all_rs: List[float] = []

    # First pass — compute RS vs SPY for all tickers
    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        stock_close = df["Close"]
        rs_spy = calc_rs_vs_benchmark(stock_close, spy_close)

        # Individual timeframe returns
        m4w = _pct_return(stock_close, 20)
        m13w = _pct_return(stock_close, 65)
        m26w = _pct_return(stock_close, 130)

        # RS vs sector ETF
        rs_sector = None
        if (
            sector_etf_data
            and ticker_sectors
            and ticker in ticker_sectors
        ):
            from utils.data_loader import get_sector_etf  # avoid circular

            sector = ticker_sectors[ticker]
            etf_ticker = get_sector_etf(sector)
            if etf_ticker and etf_ticker in sector_etf_data:
                etf_close = sector_etf_data[etf_ticker]["Close"]
                rs_sector = calc_rs_vs_benchmark(stock_close, etf_close)

        entry = {
            "ticker": ticker,
            "rs_vs_spy": rs_spy,
            "rs_vs_sector": rs_sector,
            "rs_percentile": 0.0,  # set in second pass
            "momentum_4w": round(m4w, 2) if m4w is not None else None,
            "momentum_13w": round(m13w, 2) if m13w is not None else None,
            "momentum_26w": round(m26w, 2) if m26w is not None else None,
        }
        results.append(entry)

        if rs_spy is not None:
            all_rs.append(rs_spy)

    # Second pass — compute percentile ranks
    for entry in results:
        if entry["rs_vs_spy"] is not None:
            entry["rs_percentile"] = calc_rs_percentile(
                entry["rs_vs_spy"], all_rs
            )

    logger.info(f"Computed RS for {len(results)} tickers")
    return results


# ---------------------------------------------------------------------------
# Sector rankings
# ---------------------------------------------------------------------------

def rank_sectors(
    rs_data: List[Dict[str, Any]],
    ticker_sectors: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Rank sectors by average RS of their constituents.

    Returns
    -------
    List of dicts sorted by ``avg_rs`` descending:
    ``sector``, ``avg_rs``, ``median_rs``, ``count``, ``top_stocks``.
    """
    sector_values: Dict[str, List[float]] = {}

    for entry in rs_data:
        ticker = entry["ticker"]
        rs = entry.get("rs_vs_spy")
        if rs is None:
            continue
        sector = ticker_sectors.get(ticker)
        if not sector:
            continue
        sector_values.setdefault(sector, []).append(rs)

    rankings: List[Dict[str, Any]] = []
    for sector, values in sector_values.items():
        rankings.append({
            "sector": sector,
            "avg_rs": round(float(np.mean(values)), 2),
            "median_rs": round(float(np.median(values)), 2),
            "count": len(values),
        })

    rankings.sort(key=lambda x: x["avg_rs"], reverse=True)

    # Add rank position
    for i, r in enumerate(rankings, 1):
        r["rank"] = i

    return rankings


def rank_within_sector(
    rs_data: List[Dict[str, Any]],
    ticker_sectors: Dict[str, str],
    sector: str,
) -> List[Dict[str, Any]]:
    """Rank stocks within a specific sector by RS.

    Returns list sorted by ``rs_vs_spy`` descending.
    """
    sector_stocks = [
        entry for entry in rs_data
        if ticker_sectors.get(entry["ticker"]) == sector
        and entry.get("rs_vs_spy") is not None
    ]
    sector_stocks.sort(key=lambda x: x["rs_vs_spy"], reverse=True)

    for i, s in enumerate(sector_stocks, 1):
        s["sector_rank"] = i

    return sector_stocks


def get_hot_sectors(
    sector_rankings: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[str]:
    """Return the top N sectors by average RS."""
    return [s["sector"] for s in sector_rankings[:top_n]]


def get_cold_sectors(
    sector_rankings: List[Dict[str, Any]],
    bottom_n: int = 3,
) -> List[str]:
    """Return the bottom N sectors by average RS."""
    return [s["sector"] for s in sector_rankings[-bottom_n:]]
