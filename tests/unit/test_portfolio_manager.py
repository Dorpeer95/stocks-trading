"""Unit tests for the portfolio_manager state machine.

Covers every transition in the NEW -> WATCH_CANDIDATE -> ACTIVE -> WATCH
-> EXITING state machine, plus displacement and absent-from-scan decay.

All persistence calls are patched to an in-memory FakePortfolioDB so the
tests run purely on the Mac and never touch Supabase.
"""

import pytest

from agent import portfolio_manager as pm
from agent.portfolio_manager import (
    CONSECUTIVE_NEEDED,
    DISPLACEMENT_GAP,
    ENTRY_THRESHOLD,
    EXIT_WEAK_WEEKS,
    MAX_SLOTS,
    STAY_THRESHOLD,
    run_portfolio_cycle,
)


# ---------------------------------------------------------------------------
# In-memory persistence replacement
# ---------------------------------------------------------------------------

class FakePortfolioDB:
    """Captures every persistence call so tests can assert on side effects."""

    def __init__(self, initial_holdings=None):
        self.holdings = {
            h["ticker"]: dict(h) for h in (initial_holdings or [])
        }
        self.actions = []

    def get_portfolio_holdings(self, status=None):
        rows = list(self.holdings.values())
        if status:
            rows = [h for h in rows if h.get("status") == status]
        return sorted(
            rows,
            key=lambda h: (h.get("current_confidence") or 0),
            reverse=True,
        )

    def upsert_portfolio_holding(self, holding):
        ticker = holding["ticker"]
        if ticker in self.holdings:
            self.holdings[ticker].update(holding)
        else:
            self.holdings[ticker] = dict(holding)
        return True

    def remove_portfolio_holding(self, ticker):
        self.holdings.pop(ticker, None)
        return True

    def log_portfolio_action(self, ticker, action, **kwargs):
        self.actions.append({"ticker": ticker, "action": action, **kwargs})
        return True

    def action_types(self, ticker=None):
        rows = self.actions
        if ticker is not None:
            rows = [a for a in rows if a["ticker"] == ticker]
        return [a["action"] for a in rows]


@pytest.fixture
def fake_db():
    return FakePortfolioDB()


@pytest.fixture
def patched(fake_db, monkeypatch):
    """Patch every external dependency portfolio_manager imports."""
    monkeypatch.setattr(pm, "get_portfolio_holdings", fake_db.get_portfolio_holdings)
    monkeypatch.setattr(pm, "upsert_portfolio_holding", fake_db.upsert_portfolio_holding)
    monkeypatch.setattr(pm, "remove_portfolio_holding", fake_db.remove_portfolio_holding)
    monkeypatch.setattr(pm, "log_portfolio_action", fake_db.log_portfolio_action)
    monkeypatch.setattr(pm, "get_consecutive_strong_weeks", lambda t, _th: 0)
    monkeypatch.setattr(pm, "get_cached_regime", lambda: {"regime": "neutral"})
    # Tests can also get a holder for get_portfolio_holding
    monkeypatch.setattr(
        pm,
        "get_portfolio_holding",
        lambda t: fake_db.holdings.get(t),
    )
    return fake_db


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _candidate(ticker, confidence, **overrides):
    base = {
        "ticker": ticker,
        "confidence": confidence,
        "sub_scores": {"technical": 50, "rs": 50},
        "setup_type": "vcp",
        "sector": "Tech",
        "entry_price_low": 100.0,
        "close": 100.0,
        "stop_loss": 94.0,
        "target_price": 121.0,
        "reasons": ["clean setup"],
    }
    base.update(overrides)
    return base


def _holding(ticker, status="active", confidence=75, **overrides):
    base = {
        "ticker": ticker,
        "status": status,
        "current_confidence": confidence,
        "entry_confidence": confidence,
        "consecutive_weak_weeks": 0,
        "consecutive_strong_weeks": CONSECUTIVE_NEEDED,
        "weeks_held": 4,
        "sub_scores": {},
        "setup_type": "vcp",
        "sector": "Tech",
        "last_scored_at": "2026-04-13",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Entry tests
# ---------------------------------------------------------------------------

def test_empty_portfolio_no_candidates_is_noop(patched):
    result = run_portfolio_cycle([], scan_date="2026-04-19")
    assert result["new_entries"] == []
    assert result["exits"] == []
    assert result["open_slots"] == MAX_SLOTS
    assert patched.holdings == {}


def test_first_strong_week_does_not_enter(patched, monkeypatch):
    """Candidate scores above ENTRY_THRESHOLD but only 1 consecutive week."""
    monkeypatch.setattr(pm, "get_consecutive_strong_weeks", lambda t, _th: 1)
    candidates = [_candidate("NVDA", ENTRY_THRESHOLD + 5)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert result["new_entries"] == []
    assert "NVDA" not in patched.holdings


def test_second_strong_week_enters(patched, monkeypatch):
    """Second consecutive strong week qualifies for entry."""
    monkeypatch.setattr(
        pm, "get_consecutive_strong_weeks", lambda t, _th: CONSECUTIVE_NEEDED,
    )
    candidates = [_candidate("NVDA", ENTRY_THRESHOLD + 5)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert len(result["new_entries"]) == 1
    assert result["new_entries"][0]["ticker"] == "NVDA"
    assert patched.holdings["NVDA"]["status"] == "active"
    assert "ADDED" in patched.action_types("NVDA")


def test_below_entry_threshold_never_enters(patched, monkeypatch):
    monkeypatch.setattr(
        pm, "get_consecutive_strong_weeks", lambda t, _th: CONSECUTIVE_NEEDED,
    )
    candidates = [_candidate("WEAK", ENTRY_THRESHOLD - 1)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert result["new_entries"] == []
    assert "WEAK" not in patched.holdings


# ---------------------------------------------------------------------------
# 2. Active -> Active (stays strong)
# ---------------------------------------------------------------------------

def test_active_stays_active_when_strong(patched):
    patched.holdings["NVDA"] = _holding("NVDA", status="active", confidence=80)
    candidates = [_candidate("NVDA", 78)]
    run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert patched.holdings["NVDA"]["status"] == "active"
    assert patched.holdings["NVDA"]["consecutive_weak_weeks"] == 0
    assert "HOLD_CONFIRMED" in patched.action_types("NVDA")


# ---------------------------------------------------------------------------
# 3. Active -> Watch (first weak week)
# ---------------------------------------------------------------------------

def test_active_moves_to_watch_on_first_weak_week(patched):
    patched.holdings["AMD"] = _holding("AMD", status="active", confidence=74)
    candidates = [_candidate("AMD", STAY_THRESHOLD - 1)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert patched.holdings["AMD"]["status"] == "watch"
    assert patched.holdings["AMD"]["consecutive_weak_weeks"] == 1
    assert "AMD" in result["watch_flags"]
    assert "WATCH_FLAG" in patched.action_types("AMD")


# ---------------------------------------------------------------------------
# 4. Watch -> Active (recovery)
# ---------------------------------------------------------------------------

def test_watch_recovers_to_active(patched):
    patched.holdings["AMD"] = _holding(
        "AMD", status="watch", confidence=55, consecutive_weak_weeks=1,
    )
    candidates = [_candidate("AMD", STAY_THRESHOLD + 10)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert patched.holdings["AMD"]["status"] == "active"
    assert patched.holdings["AMD"]["consecutive_weak_weeks"] == 0
    assert "AMD" in result["watch_cleared"]


# ---------------------------------------------------------------------------
# 5. Watch -> Exit (second weak week)
# ---------------------------------------------------------------------------

def test_watch_exits_after_second_weak_week(patched):
    patched.holdings["AMD"] = _holding(
        "AMD", status="watch", confidence=55,
        consecutive_weak_weeks=EXIT_WEAK_WEEKS - 1,
    )
    candidates = [_candidate("AMD", STAY_THRESHOLD - 2)]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert "AMD" in result["exits"]
    assert "AMD" not in patched.holdings, "exited holdings are hard-deleted"
    assert "REMOVED" in patched.action_types("AMD")


def test_watch_stays_watch_if_still_weak_but_not_yet_at_exit_count(patched):
    # EXIT_WEAK_WEEKS = 2 by default; start at 0 so we need a case where we're at 1.
    # Simulate a holding that just became 'watch' last week with consecutive_weak_weeks=0
    # (the transition into watch happens and this scan bumps it to 1, still below 2).
    patched.holdings["AMD"] = _holding(
        "AMD", status="watch", confidence=55, consecutive_weak_weeks=0,
    )
    # Need EXIT_WEAK_WEEKS strictly greater than 2 for this sub-test to matter;
    # with default 2, the existing transition logic already covers the case.
    candidates = [_candidate("AMD", STAY_THRESHOLD - 3)]
    run_portfolio_cycle(candidates, scan_date="2026-04-19")
    # With EXIT_WEAK_WEEKS == 2, the weak_weeks increment goes 0 -> 1 here,
    # not yet triggering exit. Holding should remain in 'watch'.
    if EXIT_WEAK_WEEKS > 1:
        assert patched.holdings.get("AMD", {}).get("status") == "watch"


# ---------------------------------------------------------------------------
# 6. GPT veto -> immediate exit regardless of state
# ---------------------------------------------------------------------------

def test_gpt_veto_forces_exit_even_when_strong(patched):
    patched.holdings["NVDA"] = _holding("NVDA", status="active", confidence=80)
    candidates = [
        _candidate(
            "NVDA", 80,
            gpt_vetoed=True,
            gpt_veto_reason="Lawsuit headline Tuesday",
        )
    ]
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert "NVDA" in result["exits"]
    assert "NVDA" not in patched.holdings
    assert "GPT_VETO" in patched.action_types("NVDA")


# ---------------------------------------------------------------------------
# 7. Displacement
# ---------------------------------------------------------------------------

def test_displacement_suggested_when_portfolio_full_and_gap_exceeded(patched, monkeypatch):
    monkeypatch.setattr(
        pm, "get_consecutive_strong_weeks", lambda t, _th: CONSECUTIVE_NEEDED,
    )
    # Fill 8 slots; the weakest is at 65
    for i, t in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        conf = 90 - i * 3  # 90, 87, 84, 81, 78, 75, 72, 69 (actually 7*3=21, 90-21=69 for H)
        patched.holdings[t] = _holding(t, status="active", confidence=conf)
    weakest_conf = min(
        h["current_confidence"] for h in patched.holdings.values()
    )
    # Scored candidate must clear weakest by DISPLACEMENT_GAP to be suggested
    new_conf = weakest_conf + DISPLACEMENT_GAP + 1
    candidates = [
        _candidate(t, patched.holdings[t]["current_confidence"])
        for t in patched.holdings
    ]
    candidates.append(_candidate("NEWHOT", new_conf))
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert len(result["displacements"]) == 1
    d = result["displacements"][0]
    assert d["new_ticker"] == "NEWHOT"
    assert d["gap"] >= DISPLACEMENT_GAP


def test_no_displacement_when_gap_too_small(patched, monkeypatch):
    monkeypatch.setattr(
        pm, "get_consecutive_strong_weeks", lambda t, _th: CONSECUTIVE_NEEDED,
    )
    for i, t in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        patched.holdings[t] = _holding(
            t, status="active", confidence=90 - i * 3,
        )
    weakest_conf = min(
        h["current_confidence"] for h in patched.holdings.values()
    )
    # Only gap - 1; should NOT be suggested
    near_conf = weakest_conf + DISPLACEMENT_GAP - 1
    candidates = [
        _candidate(t, patched.holdings[t]["current_confidence"])
        for t in patched.holdings
    ]
    candidates.append(_candidate("LUKEWARM", near_conf))
    result = run_portfolio_cycle(candidates, scan_date="2026-04-19")
    assert result["displacements"] == []


# ---------------------------------------------------------------------------
# 8. Absent-from-scan decay (Phase 3.2 fix)
#     - 1 scan absence:  prev confidence preserved, no status change
#     - 2+ scan absence: confidence capped below STAY_THRESHOLD to force WATCH
# ---------------------------------------------------------------------------

def test_absent_one_week_preserves_confidence_and_status(patched):
    # Holding last scored last week (1 scan ago relative to scan_date)
    patched.holdings["NVDA"] = _holding(
        "NVDA", status="active", confidence=80, last_scored_at="2026-04-12",
    )
    # NVDA not present in this week's scored candidates
    candidates = [_candidate("MSFT", 75)]
    run_portfolio_cycle(candidates, scan_date="2026-04-19")
    # The fix: previous confidence should NOT be arbitrarily decayed
    # by -8 after a single absence.
    assert patched.holdings["NVDA"]["status"] == "active"
    assert patched.holdings["NVDA"]["current_confidence"] >= STAY_THRESHOLD
    assert patched.holdings["NVDA"]["current_confidence"] == 80


def test_absent_two_or_more_weeks_forces_watch(patched):
    # Holding last scored more than 10 days ago = absent 2+ scans
    patched.holdings["NVDA"] = _holding(
        "NVDA", status="active", confidence=80, last_scored_at="2026-04-05",
    )
    candidates = [_candidate("MSFT", 75)]
    run_portfolio_cycle(candidates, scan_date="2026-04-19")
    # Prolonged absence: we can't see it anymore, degrade to watch
    assert patched.holdings["NVDA"]["status"] == "watch"
    assert (
        patched.holdings["NVDA"]["current_confidence"] < STAY_THRESHOLD
    ), "prolonged absence must cap confidence below stay threshold"


# ---------------------------------------------------------------------------
# 9. Slot accounting
# ---------------------------------------------------------------------------

def test_open_slots_reported_correctly(patched, monkeypatch):
    monkeypatch.setattr(pm, "get_consecutive_strong_weeks", lambda t, _th: 0)
    for t in ("A", "B", "C"):
        patched.holdings[t] = _holding(t, status="active", confidence=80)
    result = run_portfolio_cycle([], scan_date="2026-04-19")
    assert result["open_slots"] == MAX_SLOTS - 3
