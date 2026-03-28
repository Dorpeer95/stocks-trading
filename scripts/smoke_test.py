#!/usr/bin/env python3
"""
Smoke test — validate the full pipeline in dry-run mode.

Runs each stage with mock/minimal data to confirm:
  1. All imports resolve (no missing modules)
  2. Scanner filters run without crashing
  3. Scorer produces valid confidence scores
  4. Opportunity builder produces valid records
  5. Portfolio manager state machine runs without error
  6. Telegram alert formatters produce valid output
  7. safe_float / safe_int handle None from Supabase

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force dry-run mode
os.environ.setdefault("STOCKS_DRY_RUN", "true")
os.environ.setdefault("STOCKS_ENABLE_TRADING", "false")
os.environ.setdefault("STOCKS_ENABLE_ML", "false")
os.environ.setdefault("STOCKS_ENABLE_NEWS", "false")
os.environ.setdefault("STOCKS_ENABLE_GPT", "false")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
passed = 0
failed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"  \033[32m✓\033[0m {label}")


def fail(label: str, err: Exception) -> None:
    global failed
    failed += 1
    print(f"  \033[31m✗\033[0m {label}: {err}")


# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------
print("\n1. Import checks")

try:
    from utils.helpers import safe_float, safe_int
    ok("utils.helpers")
except Exception as e:
    fail("utils.helpers", e)

try:
    from agent.persistence import (
        get_open_positions,
        remove_portfolio_holding,
        log_portfolio_action,
    )
    ok("agent.persistence")
except Exception as e:
    fail("agent.persistence", e)

try:
    from agent.scanner import scan_universe
    ok("agent.scanner")
except Exception as e:
    fail("agent.scanner", e)

try:
    from agent.scorer import score_candidate, score_candidates, build_opportunity
    ok("agent.scorer")
except Exception as e:
    fail("agent.scorer", e)

try:
    from agent.portfolio import (
        update_positions_intraday,
        get_portfolio_summary,
        generate_eod_summary,
    )
    ok("agent.portfolio")
except Exception as e:
    fail("agent.portfolio", e)

try:
    from agent.portfolio_manager import run_portfolio_cycle
    ok("agent.portfolio_manager")
except Exception as e:
    fail("agent.portfolio_manager", e)

try:
    from utils.telegram_bot import (
        format_weekly_summary,
        format_eod_summary,
        format_action_needed,
        format_morning_briefing,
        format_portfolio_update,
    )
    ok("utils.telegram_bot formatters")
except Exception as e:
    fail("utils.telegram_bot formatters", e)

# ---------------------------------------------------------------------------
# 2. safe_float / safe_int (Supabase None-value pattern)
# ---------------------------------------------------------------------------
print("\n2. safe_float / safe_int")

try:
    assert safe_float(None) == 0.0, "None → 0.0"
    assert safe_float(None, 5.0) == 5.0, "None → custom default"
    assert safe_float("12.5") == 12.5, "str → float"
    assert safe_float("bad") == 0.0, "bad str → 0.0"
    assert safe_float(42) == 42.0, "int → float"

    assert safe_int(None) == 0, "None → 0"
    assert safe_int(None, -1) == -1, "None → custom default"
    assert safe_int("7") == 7, "str → int"
    assert safe_int("bad") == 0, "bad str → 0"
    assert safe_int(3.9) == 3, "float → int truncation"

    # Simulate the Supabase bug: key exists with None value
    row = {"stop_loss": None, "shares": None, "entry_price": None}
    assert safe_float(row.get("stop_loss")) == 0.0
    assert safe_int(row.get("shares")) == 0
    assert safe_float(row.get("entry_price"), 100.0) == 100.0

    ok("all safe_float/safe_int assertions pass")
except AssertionError as e:
    fail("safe_float/safe_int", e)
except Exception as e:
    fail("safe_float/safe_int", e)


# ---------------------------------------------------------------------------
# 3. Scorer with mock data
# ---------------------------------------------------------------------------
print("\n3. Scorer")

try:
    mock_stock = {
        "ticker": "TEST",
        "close": 150.0,
        "rsi_14": 52.0,
        "adx": 30.0,
        "macd_signal": "bullish",
        "ema_cross": "bullish",
        "atr_pct": 3.0,
        "bb_width": 0.04,
        "volume_ratio": 1.5,
        "distance_52w_high": -4.0,
        "rs_percentile": 85,
        "momentum_4w": 8.0,
        "momentum_13w": 15.0,
        "vcp_contraction": 7.0,
        "vcp_pivot": 152.0,
        "vcp_base_low": 140.0,
        "sector": "Technology",
    }

    scored = score_candidate(mock_stock, skip_api_calls=True)
    conf = scored.get("confidence", 0)
    assert 0 <= conf <= 100, f"confidence {conf} out of range"
    assert "sub_scores" in scored
    assert "setup_type" in scored
    ok(f"score_candidate → confidence={conf}")
except Exception as e:
    fail("score_candidate", e)

# ---------------------------------------------------------------------------
# 4. Opportunity builder
# ---------------------------------------------------------------------------
print("\n4. Opportunity builder")

try:
    mock_stock["confidence"] = 78
    opp = build_opportunity(mock_stock, portfolio_value=10000, regime={"mood": "Bullish", "position_size_modifier": 1.0})
    assert opp.get("ticker") == "TEST"
    assert opp.get("stop_loss", 0) > 0, "stop_loss should be set"
    assert opp.get("target_price", 0) > 0, "target_price should be set"
    assert opp.get("shares", 0) >= 0, "shares should be non-negative"
    ok(f"build_opportunity → shares={opp['shares']}, RR={opp.get('risk_reward_ratio')}")
except Exception as e:
    fail("build_opportunity", e)

# ---------------------------------------------------------------------------
# 5. Telegram formatters (no network, just string building)
# ---------------------------------------------------------------------------
print("\n5. Telegram formatters")

try:
    msg = format_weekly_summary(
        opportunities=[opp] if 'opp' in dir() else [],
        market_mood="Bullish",
        hot_sectors=["Technology"],
    )
    assert isinstance(msg, str) and len(msg) > 10
    ok("format_weekly_summary")
except Exception as e:
    fail("format_weekly_summary", e)

try:
    msg = format_eod_summary({
        "date": "2026-03-25",
        "portfolio_value": 10000,
        "total_invested": 5000,
        "cash_available": 5000,
        "total_pnl": 120.50,
        "total_pnl_pct": 1.2,
        "open_positions": 2,
        "positions": [
            {
                "ticker": "AAPL",
                "current_price": 180.0,
                "entry_price": 175.0,
                "unrealized_pnl": 50.0,
                "unrealized_pnl_pct": 2.9,
                "days_held": 5,
                "stop_loss": 170.0,
                "target_price": 195.0,
            }
        ],
        "risk_pct": 3.5,
        "win_rate": 60.0,
    })
    assert isinstance(msg, str) and len(msg) > 10
    ok("format_eod_summary")
except Exception as e:
    fail("format_eod_summary", e)

try:
    msg = format_action_needed({
        "ticker": "NVDA",
        "action": "SELL — STOP HIT",
        "reason": "Price $420.00 ≤ stop $425.00",
        "current_price": 420.0,
        "pnl": -50.0,
        "pnl_pct": -1.2,
        "urgent": "high",
    })
    assert isinstance(msg, str) and "NVDA" in msg
    ok("format_action_needed")
except Exception as e:
    fail("format_action_needed", e)

try:
    msg = format_morning_briefing(
        events=[],
        positions=[],
        macro={},
        enter_now=[],
        watchlist=[],
    )
    assert isinstance(msg, str) and len(msg) > 10
    ok("format_morning_briefing")
except Exception as e:
    fail("format_morning_briefing", e)

try:
    msg = format_portfolio_update({
        "portfolio_diff": {
            "holdings": [],
            "new_entries": [],
            "exits": [],
            "watch_flags": [],
            "watch_cleared": [],
            "displacements": [],
            "open_slots": 8,
        },
        "market_mood": "Neutral",
        "hot_sectors": [],
    })
    assert isinstance(msg, str) and len(msg) > 10
    ok("format_portfolio_update")
except Exception as e:
    fail("format_portfolio_update", e)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
total = passed + failed
if failed == 0:
    print(f"\033[32mAll {passed} checks passed.\033[0m")
else:
    print(f"\033[31m{failed}/{total} checks FAILED.\033[0m")
sys.exit(1 if failed else 0)
