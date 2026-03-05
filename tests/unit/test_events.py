"""
Unit tests for agent/events.py.

Tests macro event detection thresholds and market regime assessment.
Mocks persistence calls to avoid database dependencies.
"""

from unittest.mock import patch

import pytest

from agent.events import (
    detect_events,
    assess_market_regime,
    THRESHOLDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# detect_events
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectEvents:
    def test_no_events_on_calm_market(self, macro_data_normal):
        events = detect_events(macro_data_normal)
        # Normal data: VIX 14 → vix_calm event only
        types = [e["event_type"] for e in events]
        assert "vix_extreme" not in types
        assert "vix_spike" not in types
        assert "spy_crash" not in types

    def test_vix_calm(self, macro_data_normal):
        events = detect_events(macro_data_normal)
        types = [e["event_type"] for e in events]
        assert "vix_calm" in types

    def test_vix_extreme(self, macro_data_crisis):
        events = detect_events(macro_data_crisis)
        types = [e["event_type"] for e in events]
        assert "vix_extreme" in types

    def test_vix_spike_threshold(self):
        macro = {
            "vix": {"current": 25.0, "prev_close": 20.0, "change_pct": 25.0},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "vix_spike" in types

    def test_oil_surge(self):
        macro = {
            "oil": {"current": 80.0, "prev_close": 72.0, "change_pct": 11.1},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "oil_surge" in types

    def test_oil_crash(self):
        macro = {
            "oil": {"current": 65.0, "prev_close": 72.0, "change_pct": -9.7},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "oil_crash" in types

    def test_no_oil_event_on_small_move(self):
        macro = {
            "oil": {"current": 72.0, "prev_close": 73.0, "change_pct": -1.4},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert all("oil" not in t for t in types)

    def test_spy_crash(self, macro_data_crisis):
        events = detect_events(macro_data_crisis)
        types = [e["event_type"] for e in events]
        assert "spy_crash" in types

    def test_spy_rally(self):
        macro = {
            "spy": {"current": 530.0, "prev_close": 515.0, "change_pct": 2.9},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "spy_rally" in types

    def test_gold_surge(self):
        macro = {
            "gold": {"current": 2150.0, "prev_close": 2050.0, "change_pct": 4.9},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "gold_surge" in types

    def test_dollar_strong(self):
        macro = {
            "dollar": {"current": 107.0, "prev_close": 105.0, "change_pct": 1.9},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "dollar_strong" in types

    def test_treasury_high(self):
        macro = {
            "treasury_10y": {"current": 5.5, "prev_close": 5.3, "change_pct": 3.8},
        }
        events = detect_events(macro)
        types = [e["event_type"] for e in events]
        assert "treasury_high" in types

    def test_event_has_required_keys(self, macro_data_crisis):
        events = detect_events(macro_data_crisis)
        for event in events:
            assert "event_type" in event
            assert "severity" in event
            assert "description" in event
            assert "event_date" in event

    def test_empty_macro_data(self):
        events = detect_events({})
        assert events == []


# ═══════════════════════════════════════════════════════════════════════════
# assess_market_regime
# ═══════════════════════════════════════════════════════════════════════════

class TestAssessMarketRegime:
    def test_crisis_high_vix(self):
        macro = {"vix": {"current": 35.0}, "spy": {"change_pct": -1.0}}
        regime = assess_market_regime(macro)
        assert regime["regime"] == "crisis"
        assert regime["mood"] == "Bearish"
        assert regime["position_size_modifier"] <= 0.25

    def test_volatile_medium_vix(self):
        macro = {"vix": {"current": 22.0}, "spy": {"change_pct": 0.0}}
        regime = assess_market_regime(macro)
        assert regime["regime"] == "volatile"
        assert regime["position_size_modifier"] == 0.5

    def test_normal_conditions(self):
        macro = {"vix": {"current": 17.0}, "spy": {"change_pct": 0.3}}
        regime = assess_market_regime(macro)
        assert regime["regime"] == "normal"
        assert regime["mood"] == "Neutral"
        assert regime["position_size_modifier"] == 1.0

    def test_calm_low_vix(self):
        macro = {"vix": {"current": 12.0}, "spy": {"change_pct": 0.5}}
        regime = assess_market_regime(macro)
        assert regime["regime"] == "calm"
        assert regime["mood"] == "Bullish"

    def test_spy_crash_halves_size(self):
        macro = {"vix": {"current": 17.0}, "spy": {"change_pct": -4.0}}
        regime = assess_market_regime(macro)
        assert regime["position_size_modifier"] == 0.5  # 1.0 × 0.5

    def test_multiple_high_events_trigger_crisis(self):
        macro = {"vix": {"current": 18.0}, "spy": {"change_pct": 0.0}}
        events = [
            {"severity": "high"},
            {"severity": "high"},
        ]
        regime = assess_market_regime(macro, events)
        assert regime["regime"] == "crisis"

    def test_regime_has_required_keys(self, macro_data_normal):
        regime = assess_market_regime(macro_data_normal)
        for key in ("regime", "mood", "risk_level", "position_size_modifier",
                     "vix", "spy_change_pct", "description"):
            assert key in regime
