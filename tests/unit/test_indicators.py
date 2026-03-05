"""
Unit tests for utils/indicators.py.

All indicator functions are pure — they take OHLCV DataFrames and return
computed values.  Tests verify correctness, range validity, and graceful
handling of bad/empty input.
"""

import numpy as np
import pandas as pd
import pytest

from utils.indicators import (
    calc_rsi,
    calc_macd,
    calc_atr,
    calc_adx,
    calc_bollinger,
    calc_ema,
    calc_vwap,
    calc_obv,
    calc_relative_volume,
    calc_distance_52w,
    detect_ema_cross,
    macd_signal_direction,
    calc_atr_pct,
    calc_all_indicators,
)


# ═══════════════════════════════════════════════════════════════════════════
# RSI
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcRSI:
    def test_returns_series(self, ohlcv_300):
        result = calc_rsi(ohlcv_300)
        assert isinstance(result, pd.Series)

    def test_range_0_to_100(self, ohlcv_300):
        result = calc_rsi(ohlcv_300).dropna()
        assert result.min() >= 0
        assert result.max() <= 100

    def test_custom_period(self, ohlcv_300):
        result = calc_rsi(ohlcv_300, period=7)
        assert result is not None

    def test_returns_none_on_empty(self, empty_df):
        assert calc_rsi(empty_df) is None

    def test_returns_none_on_too_short(self, ohlcv_10):
        assert calc_rsi(ohlcv_10) is None

    def test_returns_none_on_none(self):
        assert calc_rsi(None) is None

    def test_close_only_works(self, close_only_df):
        result = calc_rsi(close_only_df)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# MACD
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcMACD:
    def test_returns_dataframe(self, ohlcv_300):
        result = calc_macd(ohlcv_300)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, ohlcv_300):
        result = calc_macd(ohlcv_300)
        assert set(result.columns) == {"macd", "signal", "histogram"}

    def test_returns_none_on_short_df(self, ohlcv_10):
        assert calc_macd(ohlcv_10) is None

    def test_returns_none_on_empty(self, empty_df):
        assert calc_macd(empty_df) is None

    def test_custom_parameters(self, ohlcv_300):
        result = calc_macd(ohlcv_300, fast=8, slow=17, signal=9)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# ATR
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcATR:
    def test_returns_series(self, ohlcv_300):
        result = calc_atr(ohlcv_300)
        assert isinstance(result, pd.Series)

    def test_positive_values(self, ohlcv_300):
        result = calc_atr(ohlcv_300).dropna()
        assert (result >= 0).all()

    def test_returns_none_without_hlc(self, close_only_df):
        assert calc_atr(close_only_df) is None

    def test_returns_none_on_empty(self, empty_df):
        assert calc_atr(empty_df) is None


# ═══════════════════════════════════════════════════════════════════════════
# ADX
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcADX:
    def test_returns_series(self, ohlcv_300):
        result = calc_adx(ohlcv_300)
        assert isinstance(result, pd.Series)

    def test_range_0_to_100(self, ohlcv_300):
        result = calc_adx(ohlcv_300).dropna()
        assert result.min() >= 0
        assert result.max() <= 100

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_adx(ohlcv_10) is None

    def test_returns_none_without_hlc(self, close_only_df):
        assert calc_adx(close_only_df) is None


# ═══════════════════════════════════════════════════════════════════════════
# Bollinger Bands
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcBollinger:
    def test_returns_dataframe(self, ohlcv_300):
        result = calc_bollinger(ohlcv_300)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, ohlcv_300):
        result = calc_bollinger(ohlcv_300)
        assert set(result.columns) == {"upper", "middle", "lower", "width"}

    def test_upper_above_lower(self, ohlcv_300):
        result = calc_bollinger(ohlcv_300).dropna()
        assert (result["upper"] >= result["lower"]).all()

    def test_middle_between_bands(self, ohlcv_300):
        result = calc_bollinger(ohlcv_300).dropna()
        assert (result["middle"] >= result["lower"]).all()
        assert (result["middle"] <= result["upper"]).all()

    def test_width_positive(self, ohlcv_300):
        result = calc_bollinger(ohlcv_300).dropna()
        assert (result["width"] >= 0).all()

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_bollinger(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcEMA:
    def test_returns_series(self, ohlcv_300):
        result = calc_ema(ohlcv_300, period=20)
        assert isinstance(result, pd.Series)

    def test_different_periods(self, ohlcv_300):
        for p in [9, 20, 50, 200]:
            result = calc_ema(ohlcv_300, p)
            assert result is not None

    def test_returns_none_on_short(self):
        df = pd.DataFrame({"Close": [100, 101, 102]})
        assert calc_ema(df, period=50) is None


# ═══════════════════════════════════════════════════════════════════════════
# VWAP
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcVWAP:
    def test_returns_series(self, ohlcv_300):
        result = calc_vwap(ohlcv_300)
        assert isinstance(result, pd.Series)

    def test_returns_none_without_volume(self, close_only_df):
        assert calc_vwap(close_only_df) is None

    def test_returns_none_on_empty(self, empty_df):
        assert calc_vwap(empty_df) is None


# ═══════════════════════════════════════════════════════════════════════════
# OBV
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcOBV:
    def test_returns_series(self, ohlcv_300):
        result = calc_obv(ohlcv_300)
        assert isinstance(result, pd.Series)

    def test_returns_none_without_volume(self, close_only_df):
        assert calc_obv(close_only_df) is None


# ═══════════════════════════════════════════════════════════════════════════
# Relative Volume
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcRelativeVolume:
    def test_returns_float(self, ohlcv_300):
        result = calc_relative_volume(ohlcv_300)
        assert isinstance(result, float)

    def test_positive_value(self, ohlcv_300):
        result = calc_relative_volume(ohlcv_300)
        assert result > 0

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_relative_volume(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# Distance 52-week
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcDistance52w:
    def test_returns_dict(self, ohlcv_300):
        result = calc_distance_52w(ohlcv_300)
        assert isinstance(result, dict)
        assert "high_pct" in result
        assert "low_pct" in result

    def test_high_pct_non_positive(self, ohlcv_300):
        result = calc_distance_52w(ohlcv_300)
        assert result["high_pct"] <= 0  # always at or below 52w high

    def test_low_pct_non_negative(self, ohlcv_300):
        result = calc_distance_52w(ohlcv_300)
        assert result["low_pct"] >= 0  # always at or above 52w low

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_distance_52w(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# EMA Cross Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectEmaCross:
    def test_returns_string(self, ohlcv_300):
        result = detect_ema_cross(ohlcv_300)
        assert result in ("golden_cross", "death_cross", "bullish", "bearish")

    def test_returns_none_on_short(self, ohlcv_10):
        assert detect_ema_cross(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# MACD Signal Direction
# ═══════════════════════════════════════════════════════════════════════════

class TestMACDSignalDirection:
    def test_returns_valid_string(self, ohlcv_300):
        result = macd_signal_direction(ohlcv_300)
        assert result in ("bullish", "bearish", "neutral")

    def test_returns_none_on_short(self, ohlcv_10):
        assert macd_signal_direction(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# ATR Percentage
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcATRPct:
    def test_returns_float(self, ohlcv_300):
        result = calc_atr_pct(ohlcv_300)
        assert isinstance(result, float)

    def test_positive_value(self, ohlcv_300):
        result = calc_atr_pct(ohlcv_300)
        assert result > 0

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_atr_pct(ohlcv_10) is None


# ═══════════════════════════════════════════════════════════════════════════
# Composite: calc_all_indicators
# ═══════════════════════════════════════════════════════════════════════════

class TestCalcAllIndicators:
    def test_returns_dict(self, ohlcv_300):
        result = calc_all_indicators(ohlcv_300)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, ohlcv_300):
        result = calc_all_indicators(ohlcv_300)
        expected_keys = {
            "close", "rsi_14", "macd_signal", "adx", "atr_pct", "bb_width",
            "volume_ratio", "distance_52w_high", "distance_52w_low",
            "ema_50", "ema_200", "ema_cross", "obv_latest", "vwap_latest",
        }
        assert expected_keys.issubset(result.keys())

    def test_close_is_positive(self, ohlcv_300):
        result = calc_all_indicators(ohlcv_300)
        assert result["close"] > 0

    def test_rsi_in_range(self, ohlcv_300):
        result = calc_all_indicators(ohlcv_300)
        assert 0 <= result["rsi_14"] <= 100

    def test_returns_none_on_short(self, ohlcv_10):
        assert calc_all_indicators(ohlcv_10) is None

    def test_returns_none_on_empty(self, empty_df):
        assert calc_all_indicators(empty_df) is None
