"""
Shared test fixtures for stocks-agent.

Provides reusable OHLCV DataFrames, mock data, and helper utilities
used across all unit and integration tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 300,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with realistic random walk data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    close = np.empty(n)
    close[0] = start_price
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + rng.normal(0.0005, 0.015))

    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_300() -> pd.DataFrame:
    """300-row OHLCV DataFrame — enough for all indicators."""
    return _make_ohlcv(300)


@pytest.fixture
def ohlcv_50() -> pd.DataFrame:
    """50-row OHLCV DataFrame — minimum for calc_all_indicators."""
    return _make_ohlcv(50)


@pytest.fixture
def ohlcv_10() -> pd.DataFrame:
    """10-row OHLCV DataFrame — too short for most indicators."""
    return _make_ohlcv(10)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def close_only_df() -> pd.DataFrame:
    """DataFrame with only a Close column (no OHLV)."""
    rng = np.random.default_rng(99)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({"Close": close}, index=dates)


# ---------------------------------------------------------------------------
# Macro data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def macro_data_normal():
    """Macro data representing normal market conditions."""
    return {
        "vix": {"current": 14.0, "prev_close": 14.5, "change_pct": -3.4},
        "oil": {"current": 72.50, "prev_close": 73.00, "change_pct": -0.7},
        "gold": {"current": 2050.0, "prev_close": 2045.0, "change_pct": 0.2},
        "dollar": {"current": 104.5, "prev_close": 104.3, "change_pct": 0.2},
        "treasury_10y": {"current": 4.25, "prev_close": 4.20, "change_pct": 1.2},
        "spy": {"current": 520.0, "prev_close": 518.0, "change_pct": 0.4},
    }


@pytest.fixture
def macro_data_crisis():
    """Macro data representing a crisis scenario."""
    return {
        "vix": {"current": 35.0, "prev_close": 25.0, "change_pct": 40.0},
        "oil": {"current": 60.0, "prev_close": 72.0, "change_pct": -16.7},
        "gold": {"current": 2150.0, "prev_close": 2050.0, "change_pct": 4.9},
        "dollar": {"current": 107.0, "prev_close": 104.0, "change_pct": 2.9},
        "treasury_10y": {"current": 5.2, "prev_close": 4.8, "change_pct": 8.3},
        "spy": {"current": 490.0, "prev_close": 520.0, "change_pct": -5.8},
    }


# ---------------------------------------------------------------------------
# Stock candidate fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bullish_stock():
    """A stock with bullish indicators."""
    return {
        "ticker": "AAPL",
        "close": 185.0,
        "rsi_14": 55.0,
        "adx": 30.0,
        "atr_pct": 1.8,
        "bb_width": 0.08,
        "volume_ratio": 1.5,
        "distance_52w_high": -3.0,
        "distance_52w_low": 45.0,
        "macd_signal": "bullish",
        "ema_cross": "bullish",
        "rs_percentile": 80,
        "momentum_4w": 8.0,
        "momentum_13w": 15.0,
        "momentum_26w": 25.0,
        "rs_vs_spy": 5.0,
        "rs_vs_sector": 3.0,
        "sector": "Technology",
    }


@pytest.fixture
def bearish_stock():
    """A stock with bearish indicators."""
    return {
        "ticker": "XOM",
        "close": 95.0,
        "rsi_14": 75.0,
        "adx": 15.0,
        "atr_pct": 3.5,
        "bb_width": 0.15,
        "volume_ratio": 0.4,
        "distance_52w_high": -25.0,
        "distance_52w_low": 10.0,
        "macd_signal": "bearish",
        "ema_cross": "death_cross",
        "rs_percentile": 20,
        "momentum_4w": -5.0,
        "momentum_13w": -10.0,
        "momentum_26w": -15.0,
        "rs_vs_spy": -8.0,
        "rs_vs_sector": -4.0,
        "sector": "Energy",
    }
