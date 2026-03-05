"""
Unit tests for utils/sectors.py.

Tests RS calculations, percentile ranking, and sector ranking logic.
All functions are pure math — no mocking needed.
"""

import numpy as np
import pandas as pd
import pytest

from utils.sectors import (
    _pct_return,
    calc_rs_vs_benchmark,
    calc_rs_percentile,
    rank_sectors,
    rank_within_sector,
    get_hot_sectors,
    get_cold_sectors,
    RS_WEIGHTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_series(start: float, end: float, length: int = 200) -> pd.Series:
    """Generate a linear price series from start to end."""
    return pd.Series(np.linspace(start, end, length))


# ═══════════════════════════════════════════════════════════════════════════
# _pct_return
# ═══════════════════════════════════════════════════════════════════════════

class TestPctReturn:
    def test_positive_return(self):
        s = pd.Series([100.0, 110.0])
        # lookback=1: old=100, new=110 → 10%
        result = _pct_return(s, 1)
        assert abs(result - 10.0) < 0.01

    def test_negative_return(self):
        s = pd.Series([100.0, 90.0])
        result = _pct_return(s, 1)
        assert abs(result - (-10.0)) < 0.01

    def test_zero_return(self):
        s = pd.Series([100.0, 100.0])
        result = _pct_return(s, 1)
        assert result == 0.0

    def test_returns_none_on_short_series(self):
        s = pd.Series([100.0])
        assert _pct_return(s, 5) is None

    def test_returns_none_on_none(self):
        assert _pct_return(None, 1) is None

    def test_returns_none_on_zero_price(self):
        s = pd.Series([0.0, 100.0])
        assert _pct_return(s, 1) is None


# ═══════════════════════════════════════════════════════════════════════════
# calc_rs_vs_benchmark
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcRSvsBenchmark:
    def test_outperformer(self):
        stock = _make_series(100, 130)   # up 30%
        bench = _make_series(100, 110)   # up 10%
        rs = calc_rs_vs_benchmark(stock, bench)
        assert rs is not None
        assert rs > 0  # outperforming

    def test_underperformer(self):
        stock = _make_series(100, 105)   # up 5%
        bench = _make_series(100, 130)   # up 30%
        rs = calc_rs_vs_benchmark(stock, bench)
        assert rs is not None
        assert rs < 0  # underperforming

    def test_equal_performance(self):
        stock = _make_series(100, 120)
        bench = _make_series(100, 120)
        rs = calc_rs_vs_benchmark(stock, bench)
        assert rs is not None
        assert abs(rs) < 0.5  # approximately zero

    def test_returns_none_on_short_data(self):
        stock = pd.Series([100.0, 101.0])
        bench = pd.Series([100.0, 101.0])
        rs = calc_rs_vs_benchmark(stock, bench)
        assert rs is None


# ═══════════════════════════════════════════════════════════════════════════
# calc_rs_percentile
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcRSPercentile:
    def test_top_percentile(self):
        all_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pct = calc_rs_percentile(10, all_vals)
        assert pct == 90.0  # 9 out of 10 are below

    def test_bottom_percentile(self):
        all_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pct = calc_rs_percentile(1, all_vals)
        assert pct == 0.0

    def test_median(self):
        all_vals = list(range(1, 101))
        pct = calc_rs_percentile(50, all_vals)
        assert 45 <= pct <= 55

    def test_empty_list(self):
        assert calc_rs_percentile(5, []) == 50.0


# ═══════════════════════════════════════════════════════════════════════════
# rank_sectors
# ═══════════════════════════════════════════════════════════════════════════

class TestRankSectors:
    def test_basic_ranking(self):
        rs_data = [
            {"ticker": "AAPL", "rs_vs_spy": 10.0},
            {"ticker": "MSFT", "rs_vs_spy": 8.0},
            {"ticker": "XOM", "rs_vs_spy": -5.0},
            {"ticker": "CVX", "rs_vs_spy": -3.0},
            {"ticker": "JPM", "rs_vs_spy": 2.0},
        ]
        ticker_sectors = {
            "AAPL": "Technology", "MSFT": "Technology",
            "XOM": "Energy", "CVX": "Energy",
            "JPM": "Financials",
        }
        rankings = rank_sectors(rs_data, ticker_sectors)
        assert len(rankings) == 3
        assert rankings[0]["sector"] == "Technology"  # highest avg RS
        assert rankings[-1]["sector"] == "Energy"     # lowest avg RS

    def test_rank_numbers(self):
        rs_data = [
            {"ticker": "A", "rs_vs_spy": 5.0},
            {"ticker": "B", "rs_vs_spy": -5.0},
        ]
        sectors = {"A": "SectorA", "B": "SectorB"}
        rankings = rank_sectors(rs_data, sectors)
        assert rankings[0]["rank"] == 1
        assert rankings[1]["rank"] == 2

    def test_avg_and_median(self):
        rs_data = [
            {"ticker": "A", "rs_vs_spy": 10.0},
            {"ticker": "B", "rs_vs_spy": 20.0},
        ]
        sectors = {"A": "Tech", "B": "Tech"}
        rankings = rank_sectors(rs_data, sectors)
        assert rankings[0]["avg_rs"] == 15.0
        assert rankings[0]["count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# get_hot_sectors / get_cold_sectors
# ═══════════════════════════════════════════════════════════════════════════

class TestHotColdSectors:
    @pytest.fixture
    def rankings(self):
        return [
            {"sector": "Tech", "avg_rs": 10.0},
            {"sector": "Health", "avg_rs": 5.0},
            {"sector": "Fin", "avg_rs": 2.0},
            {"sector": "Energy", "avg_rs": -1.0},
            {"sector": "Utils", "avg_rs": -3.0},
        ]

    def test_hot_sectors(self, rankings):
        hot = get_hot_sectors(rankings, top_n=2)
        assert hot == ["Tech", "Health"]

    def test_cold_sectors(self, rankings):
        cold = get_cold_sectors(rankings, bottom_n=2)
        assert cold == ["Energy", "Utils"]
