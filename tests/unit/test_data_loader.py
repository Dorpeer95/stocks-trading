"""
Unit tests for utils/data_loader.py.

Tests the ticker-list parser, caching, and data-shape validation.
Network calls to yfinance are mocked.
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from utils.data_loader import (
    get_sector_etf,
    SECTOR_ETF_MAP,
    clear_cache,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sector ETF mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestSectorETFMap:
    def test_known_sectors(self):
        known = [
            "Technology", "Health Care", "Financials",
            "Consumer Discretionary", "Industrials", "Energy",
        ]
        for sector in known:
            etf = get_sector_etf(sector)
            assert etf is not None, f"Missing ETF for {sector}"

    def test_unknown_sector_returns_none(self):
        assert get_sector_etf("Nonexistent") is None

    def test_all_etfs_are_strings(self):
        for sector, etf in SECTOR_ETF_MAP.items():
            assert isinstance(etf, str)
            assert len(etf) <= 5  # ticker symbols ≤ 5 chars


# ═══════════════════════════════════════════════════════════════════════════
# fetch_sp500_list (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestFetchSP500List:
    def setup_method(self):
        clear_cache()

    @patch("utils.data_loader.pd.read_html")
    def test_returns_list_of_dicts(self, mock_read_html):
        mock_df = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple Inc.", "Microsoft Corp."],
            "GICS Sector": ["Technology", "Technology"],
            "GICS Sub-Industry": ["Consumer Electronics", "Systems Software"],
        })
        mock_read_html.return_value = [mock_df]

        from utils.data_loader import fetch_sp500_list
        result = fetch_sp500_list()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["sector"] == "Technology"

    @patch("utils.data_loader.pd.read_html")
    def test_dots_replaced_with_dashes(self, mock_read_html):
        """Tickers like BRK.B → BRK-B for yfinance."""
        mock_df = pd.DataFrame({
            "Symbol": ["BRK.B"],
            "Security": ["Berkshire"],
            "GICS Sector": ["Financials"],
            "GICS Sub-Industry": ["Multi-Sector Holdings"],
        })
        mock_read_html.return_value = [mock_df]

        from utils.data_loader import fetch_sp500_list
        result = fetch_sp500_list()
        assert result[0]["ticker"] == "BRK-B"


# ═══════════════════════════════════════════════════════════════════════════
# fetch_price_data (mocked at yf.Ticker level)
# ═══════════════════════════════════════════════════════════════════════════

class TestFetchPriceData:
    def setup_method(self):
        clear_cache()

    @patch("utils.data_loader.yf.Ticker")
    def test_returns_dataframe(self, mock_ticker_cls):
        mock_df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [98.0],
            "Close": [102.0], "Volume": [1000000],
        })
        mock_inst = MagicMock()
        mock_inst.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_inst

        from utils.data_loader import fetch_price_data
        result = fetch_price_data("AAPL", period="1mo")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "Close" in result.columns

    @patch("utils.data_loader.yf.Ticker")
    def test_returns_none_on_empty(self, mock_ticker_cls):
        mock_inst = MagicMock()
        mock_inst.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_inst

        from utils.data_loader import fetch_price_data
        result = fetch_price_data("INVALID", period="1mo")
        assert result is None
