# Bot Improvement Plan — Managed Portfolio Mode

> Goal: Transform from "weekly fresh picks" to a proper managed portfolio — consistent holdings,
> high-conviction signals only, clear HOLD/WATCH/BUY/SELL decisions each week.

---

## Core Problem

The bot currently treats every weekly scan as **independent**. It produces 10 "opportunities" that
may have nothing to do with last week's picks. There is no persistent model portfolio — just a list
of pending opportunities. This is noise, not portfolio management.

**What we want instead:**
- A persistent portfolio of N stocks (target: 8 slots)
- Stocks are **held** until a clear exit signal, not swapped every week
- New stocks only enter when (a) a slot is empty, or (b) a much stronger candidate displaces the weakest hold
- Weekly output = "HOLD / WATCH / BUY / SELL" per position — not a fresh list

---

## Phase 1 — DB Changes (Foundation)

### New table: `portfolio_holdings`
The live model portfolio. Always reflects the current intended state.

```sql
CREATE TABLE stocks.portfolio_holdings (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker           VARCHAR(10) NOT NULL UNIQUE,
    status           VARCHAR(20) NOT NULL DEFAULT 'active', -- active | watch | exiting
    entry_confidence NUMERIC(5,1),          -- confidence when first added
    current_confidence NUMERIC(5,1),        -- updated each weekly scan
    consecutive_strong_weeks INTEGER DEFAULT 0, -- weeks above entry threshold before entry
    consecutive_weak_weeks   INTEGER DEFAULT 0, -- weeks below stay threshold (triggers exit)
    added_at         TIMESTAMPTZ DEFAULT NOW(),
    last_scored_at   TIMESTAMPTZ,
    sector           VARCHAR(100),
    entry_price      NUMERIC(10,2),
    stop_loss        NUMERIC(10,2),
    target_price     NUMERIC(10,2),
    sub_scores       JSONB,
    notes            TEXT
);
```

### New table: `signal_history`
Weekly confidence score per ticker — tracks signal continuity over time.

```sql
CREATE TABLE stocks.signal_history (
    id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker       VARCHAR(10) NOT NULL,
    scan_date    DATE NOT NULL,
    confidence   NUMERIC(5,1),
    sub_scores   JSONB,
    in_portfolio BOOLEAN DEFAULT FALSE,
    setup_type   VARCHAR(50),
    UNIQUE(ticker, scan_date)
);
```

### Modify `opportunities` table
Add `scan_date` column (already partially there as `score_date`) and `signal_week` integer
to track which consecutive scan this is.

### New table: `portfolio_log`
Audit trail of every portfolio state change.

```sql
CREATE TABLE stocks.portfolio_log (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker      VARCHAR(10) NOT NULL,
    action      VARCHAR(20) NOT NULL,  -- ADDED | REMOVED | WATCH_FLAG | HOLD_CONFIRMED
    reason      TEXT,
    confidence  NUMERIC(5,1),
    scan_date   DATE DEFAULT CURRENT_DATE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Phase 2 — Portfolio State Machine (New Core Module)

New file: `agent/portfolio_manager.py`

### Signal Thresholds
```
ENTRY_THRESHOLD     = 72    (was 65 — only strong signals get in)
STAY_THRESHOLD      = 58    (stay in portfolio until confidence drops here...)
EXIT_THRESHOLD      = 2     (... for 2 consecutive weeks)
CONSECUTIVE_NEEDED  = 2     (must score ≥ ENTRY_THRESHOLD for 2 straight weeks to enter)
DISPLACEMENT_GAP    = 10    (new stock needs 10+ points above weakest hold to displace it)
MAX_SLOTS           = 8
```

### State Machine per ticker

```
NEW CANDIDATE
    ↓ confidence ≥ 72 for 1st week
WATCH_CANDIDATE (tracked but not yet in portfolio)
    ↓ confidence ≥ 72 for 2nd consecutive week + slot available
ACTIVE (in portfolio)
    ↓ confidence drops 58-72
WATCH (in portfolio, weakening — 1st weak week)
    ↓ confidence < 58 for 2nd consecutive week  OR  hard exit (stop hit / target hit)
EXITING → alert user → remove from portfolio
```

### Displacement rule
If portfolio is full (8 slots) and a new candidate scores ≥ (weakest_hold + 10),
flag as `DISPLACE` alert. User sees: "Consider swapping WEAK_STOCK (conf 62) → NEW_STOCK (conf 77)".
Never auto-displace — always advisory.

### Core methods
- `run_portfolio_cycle(scored_candidates, regime)` — main weekly method
- `_update_existing_holdings(scored_map)` — re-score all current holds
- `_find_entries(candidates, current_holds)` — find stocks ready to enter
- `_find_exits(current_holds)` — flag stocks to exit
- `_find_displacements(candidates, current_holds)` — displacement candidates
- `get_portfolio_diff()` → returns `{holds, watches, new_entries, exits, displacements}`

---

## Phase 3 — Signal Quality Upgrade

### 3a. Scanner tightening (`agent/scanner.py`)
| Filter | Current | New |
|---|---|---|
| Min ADX | 25 | 28 |
| Min RS percentile | 60 | 70 (top 30% only) |
| Max ATR % | 7% | 6% (tighter vol cap) |
| Max VCP contraction | 12% | 10% |
| Max dist from 52w high | -25% | -20% |
| Top candidates returned | 20 | 15 |
| RSI overbought | < 80 | < 75 |

Add new filter: **sector leadership** — stock must be in a top-3 ranked sector.

### 3b. Scorer tightening (`agent/scorer.py`)
| Setting | Current | New |
|---|---|---|
| MIN_CONFIDENCE | 65 | 72 |
| W_TECHNICAL | 29.75% | 30% |
| W_RS | 21.25% | 25% |
| W_ML | 15% | 15% (unchanged) |
| W_FUNDAMENTAL | 12.75% | 12% |
| W_SENTIMENT | 8.5% | 8% |
| W_INSIDER | 8.5% | 8% |
| W_MACRO | 4.25% | 2% |

Rationale: raise RS weight — stocks that consistently outperform the market are the core signal.
Lower macro weight — macro already filters via regime adjustments.

Add to scoring: **signal_age bonus** — if a stock has been consistently scoring ≥65 for 3+ weeks,
add up to +5 to its confidence (rewards persistence over one-off flukes).

### 3c. Fix the risk cap contradiction (`utils/position_sizing.py`)
```
DEFAULT_MAX_TOTAL_RISK = 0.06  →  0.16  (8 positions × 2% = 16%)
```

---

## Phase 4 — Rotation Engine (in `agent/agent.py`)

Modify `weekly_scan()` to call `portfolio_manager.run_portfolio_cycle()` **after** scoring.

New flow:
```
1. Scan universe
2. Score candidates (existing logic)
3. Save signal history for all scored stocks
4. Run portfolio cycle:
   a. Re-score current holdings with fresh data
   b. Flag holdings dropping below stay threshold
   c. Exit holdings at or below exit threshold (2 weak weeks)
   d. Add candidates that meet entry threshold + 2 consecutive weeks
   e. Check for displacement opportunities
5. Send portfolio-diff Telegram alert (new format)
```

---

## Phase 5 — Alert Format Overhaul (`utils/telegram_bot.py`)

Replace `weekly_summary` alert with `portfolio_update` alert.

### New weekly Telegram format
```
📊 PORTFOLIO — Week of {date}
Market: {mood} | VIX: {vix}
━━━━━━━━━━━━━━━━━━━━━━━━━

✅ HOLDING STRONG ({n} positions)
  NVDA  conf 84 ↑ wk4  |  +12.3%  stop $412
  MSFT  conf 79 ↑ wk2  |  +5.1%   stop $380
  ...

⚠️ WATCH — SIGNAL WEAKENING
  AMD   conf 61 ↓ was 74  |  2nd weak week → exit soon

🟢 NEW ENTRY (strong signal 2 weeks)
  META  conf 77  entry $485–492  stop $461  tgt $540

🔴 EXIT SIGNAL
  TSLA  conf 48 — below threshold 2 weeks → SELL

🔄 CONSIDER SWAP (optional)
  WEAK: INTC (conf 63) → STRONG: AVGO (conf 74, +11 pts)

📭 Open slots: 2 / 8
━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Morning briefing — change focus
Instead of "enter now / watchlist", show:
- Portfolio health: each holding's current price vs stop/target
- Any intraday alerts since last briefing
- Today's key holdings to watch (close to stop or close to target)

---

## Phase 6 — Technical Bug Fixes

### Fix 1: Intraday monitoring uses daily data (`agent/portfolio.py` line 71)
```python
# Current (WRONG for intraday):
df = fetch_price_data(ticker, period="5d", interval="1d")

# Fix:
df = fetch_price_data(ticker, period="1d", interval="5m")
current = float(df["Close"].iloc[-1])  # last 5-min close
```

### Fix 2: EOD summary keeps using daily data (fine)
The EOD summary can keep `interval="1d"` since it runs after market close.

---

## Implementation Order

| Phase | Effort | Impact | Do first? |
|---|---|---|---|
| 1 — DB tables | Small | Foundation | ✅ Yes |
| 2 — Portfolio state machine | Large | Core | ✅ Yes |
| 3c — Fix risk cap | Tiny | Bug fix | ✅ Yes |
| 6 — Fix intraday data | Tiny | Bug fix | ✅ Yes |
| 3a/3b — Signal quality | Medium | Quality | After Phase 2 |
| 4 — Rotation engine | Medium | Core | After Phase 2 |
| 5 — Alert format | Medium | UX | After Phase 4 |

---

## What Does NOT Change
- VCP-based entry logic (it's sound)
- XGBoost ML models (keep as-is, add signal_age as a feature in next retrain)
- Alpaca integration (paper trading, low priority)
- Sentiment pipeline (VADER + GPT-4o, keep)
- Fundamental scoring (keep)
- Insider signal (keep)
- Supabase schema for positions/trades/opportunities (keep, new tables are additive)

---

## Success Criteria
- Portfolio contains 6–8 stocks at all times once warmed up
- Average hold duration > 3 weeks (vs currently unknown)
- Weekly alert clearly shows HOLD/WATCH/BUY/SELL — not a fresh list
- New stocks require 2+ weeks of strong signal before entry
- Win rate improves from baseline as noise is filtered out
