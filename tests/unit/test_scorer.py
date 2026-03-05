"""
Unit tests for agent/scorer.py.

Tests sub-score functions and the composite scoring logic.
Since scorer imports from utils.earnings and utils.insider (which may call
external APIs), we mock those dependencies for isolation.
"""

from unittest.mock import patch

import pytest

from agent.scorer import (
    score_technical,
    score_relative_strength,
    score_fundamental,
    score_macro,
    score_sentiment,
    score_ml,
    score_candidate,
    detect_setup_type,
)


# ═══════════════════════════════════════════════════════════════════════════
# score_technical
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreTechnical:
    def test_range_0_100(self, bullish_stock, bearish_stock):
        assert 0 <= score_technical(bullish_stock) <= 100
        assert 0 <= score_technical(bearish_stock) <= 100

    def test_bullish_higher_than_bearish(self, bullish_stock, bearish_stock):
        assert score_technical(bullish_stock) > score_technical(bearish_stock)

    def test_empty_stock(self):
        """Empty dict should still return a score (baseline 50)."""
        assert score_technical({}) == 50.0

    def test_rsi_neutral_zone(self):
        stock = {"rsi_14": 50}
        assert score_technical(stock) > 50  # neutral zone gets +10

    def test_rsi_overbought_penalty(self):
        stock = {"rsi_14": 80}
        assert score_technical(stock) < 50  # overbought gets -10

    def test_golden_cross_boost(self):
        stock = {"ema_cross": "golden_cross"}
        assert score_technical(stock) == 65  # 50 + 15

    def test_death_cross_penalty(self):
        stock = {"ema_cross": "death_cross"}
        assert score_technical(stock) == 35  # 50 - 15


# ═══════════════════════════════════════════════════════════════════════════
# score_relative_strength
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreRelativeStrength:
    def test_range_0_100(self, bullish_stock, bearish_stock):
        assert 0 <= score_relative_strength(bullish_stock) <= 100
        assert 0 <= score_relative_strength(bearish_stock) <= 100

    def test_high_rs_high_score(self):
        stock = {"rs_percentile": 90, "momentum_4w": 10, "momentum_13w": 15}
        assert score_relative_strength(stock) == 100  # 90 + 5 + 5 = 100

    def test_low_rs_low_score(self):
        stock = {"rs_percentile": 10}
        assert score_relative_strength(stock) == 10

    def test_default_rs(self):
        assert score_relative_strength({}) == 50


# ═══════════════════════════════════════════════════════════════════════════
# score_fundamental
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreFundamental:
    def test_no_fundamentals_returns_50(self):
        assert score_fundamental({}) == 50.0

    def test_strong_fundamentals(self):
        stock = {}
        fundies = {
            "pe_ratio": 15,
            "forward_pe": 12,
            "profit_margin": 0.20,
            "revenue_growth": 0.15,
            "short_pct_float": 0.02,
        }
        result = score_fundamental(stock, fundies)
        # 50 + 10 (fwd < pe) + 5 (reasonable PE) + 10 (margin) + 10 (growth) + 5 (low short) = 90
        assert result == 90.0

    def test_weak_fundamentals(self):
        stock = {}
        fundies = {
            "pe_ratio": 15,
            "forward_pe": 20,  # fwd > pe → no bonus
            "profit_margin": 0.02,  # poor
            "revenue_growth": -0.05,  # negative
            "short_pct_float": 0.25,  # high short
        }
        result = score_fundamental(stock, fundies)
        # 50 + 0 + 5 (reasonable PE) + 0 + 0 - 5 (high short) = 50
        assert result == 50.0


# ═══════════════════════════════════════════════════════════════════════════
# score_macro
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreMacro:
    def test_no_regime_returns_50(self):
        assert score_macro() == 50.0

    def test_bullish_regime(self):
        assert score_macro({"mood": "Bullish"}) == 80.0

    def test_neutral_regime(self):
        assert score_macro({"mood": "Neutral"}) == 50.0

    def test_bearish_regime(self):
        assert score_macro({"mood": "Bearish"}) == 25.0


# ═══════════════════════════════════════════════════════════════════════════
# score_sentiment
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreSentiment:
    def test_no_data_returns_50(self):
        assert score_sentiment() == 50.0

    def test_positive_sentiment(self):
        data = {"score": 10.0}
        assert score_sentiment(data) == 100.0

    def test_negative_sentiment(self):
        data = {"score": -10.0}
        assert score_sentiment(data) == 0.0

    def test_neutral_sentiment(self):
        data = {"score": 0.0}
        assert score_sentiment(data) == 50.0

    def test_risk_flags_reduce_score(self):
        data = {"score": 5.0, "risk_flags": ["lawsuit", "recall"]}
        result = score_sentiment(data)
        # (5 + 10) * 5 = 75 - 10 (2 flags × 5) = 65
        assert result == 65.0


# ═══════════════════════════════════════════════════════════════════════════
# score_ml
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreML:
    def test_no_predictions_returns_50(self):
        assert score_ml() == 50.0

    def test_bullish_direction(self):
        preds = {"direction": {"signal": "bullish", "probability": 0.8}}
        result = score_ml(preds)
        assert result > 50

    def test_bearish_direction(self):
        preds = {"direction": {"signal": "bearish", "probability": 0.2}}
        result = score_ml(preds)
        assert result < 50

    def test_neutral_direction(self):
        preds = {"direction": {"signal": "neutral", "probability": 0.5}}
        result = score_ml(preds)
        assert result == 50.0


# ═══════════════════════════════════════════════════════════════════════════
# detect_setup_type
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectSetupType:
    def test_golden_cross_macd(self):
        stock = {"ema_cross": "golden_cross", "macd_signal": "bullish"}
        assert detect_setup_type(stock) == "Golden Cross + MACD"

    def test_oversold_reversal(self):
        stock = {"rsi_14": 25, "macd_signal": "bullish"}
        assert detect_setup_type(stock) == "Oversold Reversal"

    def test_breakout(self):
        stock = {"distance_52w_high": -2, "adx": 30}
        assert detect_setup_type(stock) == "Breakout (near 52w high)"

    def test_squeeze(self):
        stock = {"bb_width": 0.03, "adx": 15}
        assert detect_setup_type(stock) == "Squeeze (low BB width)"

    def test_mixed_signal_default(self):
        assert detect_setup_type({}) == "Mixed Signal"


# ═══════════════════════════════════════════════════════════════════════════
# score_candidate (composite)
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreCandidate:
    @patch("agent.scorer.score_insider", return_value=50.0)
    @patch("agent.scorer.score_earnings_risk", return_value=60.0)
    def test_returns_enriched_stock(self, mock_earn, mock_ins, bullish_stock):
        result = score_candidate(bullish_stock)
        assert "confidence" in result
        assert "sub_scores" in result
        assert "setup_type" in result

    @patch("agent.scorer.score_insider", return_value=50.0)
    @patch("agent.scorer.score_earnings_risk", return_value=60.0)
    def test_confidence_in_range(self, mock_earn, mock_ins, bullish_stock):
        result = score_candidate(bullish_stock)
        assert 0 <= result["confidence"] <= 100

    def test_skip_api_calls(self, bullish_stock):
        """When skip_api_calls=True, insider/earnings should not be called."""
        result = score_candidate(bullish_stock, skip_api_calls=True)
        assert 0 <= result["confidence"] <= 100
