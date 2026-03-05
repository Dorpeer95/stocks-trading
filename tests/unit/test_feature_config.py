"""
Unit tests for agent/feature_config.py.

Tests feature vector builders, validation, and encoding helpers.
"""

import numpy as np
import pytest

from agent.feature_config import (
    DIRECTION_FEATURES,
    VOLATILITY_FEATURES,
    EARNINGS_FEATURES,
    SECTOR_FEATURES,
    MODEL_FEATURES,
    build_direction_vector,
    build_volatility_vector,
    build_earnings_vector,
    build_sector_vector,
    build_feature_vector,
    validate_features,
    _encode_ema_cross,
    _encode_macd_signal,
    _safe_float,
)


# ═══════════════════════════════════════════════════════════════════════════
# Feature vector shape
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureVectorShape:
    def test_direction_vector_length(self, bullish_stock):
        vec = build_direction_vector(bullish_stock)
        assert vec is not None
        assert len(vec) == len(DIRECTION_FEATURES)

    def test_volatility_vector_length(self, bullish_stock):
        vec = build_volatility_vector(bullish_stock)
        assert vec is not None
        assert len(vec) == len(VOLATILITY_FEATURES)

    def test_earnings_vector_length(self, bullish_stock):
        vec = build_earnings_vector(bullish_stock)
        assert vec is not None
        assert len(vec) == len(EARNINGS_FEATURES)

    def test_sector_vector_length(self):
        sector_data = {"avg_rs": 0.5, "median_rs": 0.3, "momentum_4w": 2.0}
        vec = build_sector_vector(sector_data)
        assert vec is not None
        assert len(vec) == len(SECTOR_FEATURES)


class TestFeatureVectorDtype:
    def test_direction_float32(self, bullish_stock):
        vec = build_direction_vector(bullish_stock)
        assert vec.dtype == np.float32

    def test_volatility_float32(self, bullish_stock):
        vec = build_volatility_vector(bullish_stock)
        assert vec.dtype == np.float32

    def test_earnings_float32(self, bullish_stock):
        vec = build_earnings_vector(bullish_stock)
        assert vec.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════
# No NaN in output
# ═══════════════════════════════════════════════════════════════════════════

class TestNoNaNOutput:
    def test_direction_no_nan(self, bullish_stock):
        vec = build_direction_vector(bullish_stock)
        assert not np.any(np.isnan(vec))

    def test_volatility_no_nan(self, bullish_stock):
        vec = build_volatility_vector(bullish_stock)
        assert not np.any(np.isnan(vec))

    def test_earnings_no_nan(self, bullish_stock):
        vec = build_earnings_vector(bullish_stock)
        assert not np.any(np.isnan(vec))

    def test_empty_stock_no_nan(self):
        """Even an empty stock dict should produce no NaN (uses defaults)."""
        vec = build_direction_vector({})
        assert vec is not None
        assert not np.any(np.isnan(vec))


# ═══════════════════════════════════════════════════════════════════════════
# Encoders
# ═══════════════════════════════════════════════════════════════════════════

class TestEncoders:
    def test_ema_cross_encoder(self):
        assert _encode_ema_cross("golden_cross") == 2.0
        assert _encode_ema_cross("bullish") == 1.0
        assert _encode_ema_cross(None) == 0.0
        assert _encode_ema_cross("bearish") == -1.0
        assert _encode_ema_cross("death_cross") == -2.0
        assert _encode_ema_cross("unknown") == 0.0

    def test_macd_signal_encoder(self):
        assert _encode_macd_signal("bullish") == 1.0
        assert _encode_macd_signal("neutral") == 0.0
        assert _encode_macd_signal("bearish") == -1.0
        assert _encode_macd_signal(None) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# validate_features
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateFeatures:
    def test_valid_direction_vector(self, bullish_stock):
        vec = build_direction_vector(bullish_stock)
        valid, msg = validate_features(vec, "direction")
        assert valid is True
        assert msg == "OK"

    def test_none_vector(self):
        valid, msg = validate_features(None, "direction")
        assert valid is False

    def test_wrong_length(self):
        vec = np.array([1.0, 2.0, 3.0])
        valid, msg = validate_features(vec, "direction")
        assert valid is False
        assert "Expected" in msg

    def test_nan_in_vector(self):
        vec = np.array([np.nan] * len(DIRECTION_FEATURES), dtype=np.float32)
        valid, msg = validate_features(vec, "direction")
        assert valid is False
        assert "NaN" in msg

    def test_unknown_model(self):
        vec = np.array([1.0])
        valid, msg = validate_features(vec, "nonexistent_model")
        assert valid is False


# ═══════════════════════════════════════════════════════════════════════════
# build_feature_vector (generic)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildFeatureVectorGeneric:
    def test_direction(self, bullish_stock):
        vec = build_feature_vector("direction", bullish_stock)
        assert vec is not None
        assert len(vec) == len(DIRECTION_FEATURES)

    def test_volatility(self, bullish_stock):
        vec = build_feature_vector("volatility", bullish_stock)
        assert vec is not None

    def test_unknown_model(self, bullish_stock):
        vec = build_feature_vector("unknown", bullish_stock)
        assert vec is None


# ═══════════════════════════════════════════════════════════════════════════
# _safe_float
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_none_default(self):
        assert _safe_float(None, 99.0) == 99.0

    def test_nan_default(self):
        assert _safe_float(float("nan"), 5.0) == 5.0

    def test_inf_default(self):
        assert _safe_float(float("inf"), 5.0) == 5.0

    def test_string_default(self):
        assert _safe_float("not_a_number", 0.0) == 0.0
