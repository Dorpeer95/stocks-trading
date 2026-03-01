# STOCKS-AGENT — COMPLETE DEVELOPMENT PLAN

> **Created:** March 1, 2026  
> **Author:** Architecture Review  
> **Status:** Planning Phase  
> **Version:** 1.0

---

## TABLE OF CONTENTS

1. [Deliverable 1 — Supabase Migration SQL](#deliverable-1--supabase-migration-sql)
2. [Deliverable 2 — Development Phases](#deliverable-2--development-phases)
3. [Deliverable 3 — Feature Priority List](#deliverable-3--feature-priority-list)
4. [Deliverable 4 — Risk Assessment](#deliverable-4--risk-assessment)
5. [Deliverable 5 — API Rate Limit Plan](#deliverable-5--api-rate-limit-plan)
6. [Deliverable 6 — RAM Budget Plan](#deliverable-6--ram-budget-plan)
7. [Deliverable 7 — Dependency Map](#deliverable-7--dependency-map)
8. [Deliverable 8 — Testing Strategy](#deliverable-8--testing-strategy)
9. [Deliverable 9 — Coding Standards](#deliverable-9--coding-standards)
10. [Deliverable 10 — First Prompt](#deliverable-10--first-prompt)
11. [Flagged Issues & Assumptions](#flagged-issues--assumptions)

---

## DELIVERABLE 1 — SUPABASE MIGRATION SQL

### Storage Budget

| Consumer | Estimated Size | Notes |
|----------|---------------|-------|
| Crypto schema (existing) | ~80-120 MB | Already in use |
| Stocks schema (new) | ~150-200 MB | Target budget |
| Supabase overhead (indexes, WAL, system) | ~100 MB | Invisible but real |
| **Safety margin** | **~80-120 MB** | Must preserve |
| **Total** | **≤ 500 MB** | Hard limit |

### Per-Table Storage Estimates

| Table | Row Size (avg) | Rows/Year | Year 1 Size | Year 2 Size | Notes |
|-------|---------------|-----------|-------------|-------------|-------|
| `stocks.universe` | ~200 bytes | 500 (static) | ~100 KB | ~100 KB | Refreshed, not appended |
| `stocks.daily_scores` | ~350 bytes | 126,000 | ~42 MB | ~84 MB | **Largest table** — needs purge policy |
| `stocks.opportunities` | ~500 bytes | 520 | ~260 KB | ~520 KB | ~10/week × 52 weeks |
| `stocks.positions` | ~400 bytes | 200 | ~80 KB | ~160 KB | Advisory positions |
| `stocks.market_events` | ~300 bytes | 1,260 | ~378 KB | ~756 KB | ~5/day × 252 days |
| `stocks.equity_snapshots` | ~200 bytes | 52 | ~10 KB | ~20 KB | Weekly only |
| `stocks.model_versions` | ~5 KB | 50 | ~250 KB | ~500 KB | Includes metadata, not model binary |
| `stocks.trades` | ~500 bytes | 200 | ~100 KB | ~200 KB | Closed positions |
| **Indexes (est. 30% overhead)** | — | — | ~13 MB | ~26 MB | All tables combined |
| **TOTAL** | — | — | **~57 MB** | **~113 MB** | Well within budget |

> **⚠️ CRITICAL:** `daily_scores` dominates storage. Implement a 6-month rolling purge (cron job deletes rows > 180 days old). This caps the table at ~63 MB permanently.

### Complete Migration SQL

```sql
-- =============================================================
-- STOCKS-AGENT SUPABASE MIGRATION
-- Schema: stocks (already created)
-- Run this migration ONCE on Supabase SQL Editor
-- =============================================================

-- Enable necessary extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================
-- TABLE 1: stocks.universe
-- Purpose: S&P 500 stocks the bot scans
-- Write: Weekly refresh (DELETE + INSERT batch)
-- Storage: ~100 KB (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.universe (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL UNIQUE,
    company_name    VARCHAR(200) NOT NULL,
    sector          VARCHAR(50) NOT NULL,
    industry        VARCHAR(100) NOT NULL,
    market_cap_b    NUMERIC(10,2),            -- in billions
    avg_volume_50d  BIGINT,                   -- 50-day avg volume
    in_sp500        BOOLEAN DEFAULT TRUE,
    is_active       BOOLEAN DEFAULT TRUE,     -- soft delete
    added_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_universe_ticker ON stocks.universe(ticker);
CREATE INDEX idx_universe_sector ON stocks.universe(sector);
CREATE INDEX idx_universe_active ON stocks.universe(is_active) WHERE is_active = TRUE;

-- =============================================================
-- TABLE 2: stocks.daily_scores
-- Purpose: Daily composite score per stock
-- Write: Once per trading day, batch of 500 rows
-- Storage: ~42 MB/year (NEEDS ROLLING PURGE)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.daily_scores (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    score_date      DATE NOT NULL,

    -- Composite scores (0-100)
    total_score         SMALLINT NOT NULL CHECK (total_score BETWEEN 0 AND 100),
    technical_score     SMALLINT CHECK (technical_score BETWEEN 0 AND 100),
    rs_score            SMALLINT CHECK (rs_score BETWEEN 0 AND 100),
    fundamental_score   SMALLINT CHECK (fundamental_score BETWEEN 0 AND 100),
    sentiment_score     SMALLINT CHECK (sentiment_score BETWEEN -100 AND 100),

    -- Key indicators (stored for dashboard/debugging)
    rsi_14              NUMERIC(5,2),
    macd_signal         VARCHAR(10),           -- 'bullish', 'bearish', 'neutral'
    adx                 NUMERIC(5,2),
    atr_pct             NUMERIC(5,3),          -- ATR as % of price
    bb_width            NUMERIC(5,3),
    volume_ratio        NUMERIC(5,2),          -- vs 50d avg
    rs_vs_spy           NUMERIC(6,2),
    distance_52w_high   NUMERIC(5,2),          -- % from 52w high

    -- ML model outputs
    direction_prob      NUMERIC(4,3),          -- P(up 5% in 10d)
    volatility_prob     NUMERIC(4,3),          -- P(big move in 5d)

    -- Pattern flags (bitfield-friendly but using boolean for clarity)
    pattern_detected    VARCHAR(50),           -- 'coiling', 'cup_handle', 'bull_flag', etc.

    created_at          TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_daily_scores UNIQUE (ticker, score_date)
);

CREATE INDEX idx_daily_scores_date ON stocks.daily_scores(score_date DESC);
CREATE INDEX idx_daily_scores_ticker_date ON stocks.daily_scores(ticker, score_date DESC);
CREATE INDEX idx_daily_scores_total ON stocks.daily_scores(score_date, total_score DESC);

-- =============================================================
-- TABLE 3: stocks.opportunities
-- Purpose: Weekly buy recommendations (top picks)
-- Write: Weekly (5-10 rows)
-- Storage: ~260 KB/year (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.opportunities (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    scan_date       DATE NOT NULL,
    
    -- Recommendation
    direction       VARCHAR(5) NOT NULL DEFAULT 'long',  -- 'long' or 'avoid'
    confidence      SMALLINT NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    
    -- Entry plan
    entry_price_low     NUMERIC(10,2),
    entry_price_high    NUMERIC(10,2),
    stop_loss           NUMERIC(10,2),
    target_price        NUMERIC(10,2),
    position_size_usd   NUMERIC(10,2),
    risk_usd            NUMERIC(10,2),
    reward_usd          NUMERIC(10,2),
    risk_reward_ratio   NUMERIC(4,2),
    
    -- Reasoning (for Telegram & dashboard)
    reasons             JSONB NOT NULL DEFAULT '[]',  -- array of reason strings
    sector              VARCHAR(50),
    setup_type          VARCHAR(50),                   -- 'coiling', 'breakout', etc.
    
    -- Lifecycle
    status              VARCHAR(20) DEFAULT 'pending', -- pending/entered/expired/skipped
    expires_at          DATE,                          -- opportunity expires
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_opportunities_date ON stocks.opportunities(scan_date DESC);
CREATE INDEX idx_opportunities_status ON stocks.opportunities(status);

-- =============================================================
-- TABLE 4: stocks.positions
-- Purpose: Track advisory positions (human executes)
-- Write: On entry/update signal only
-- Storage: ~80 KB/year (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.positions (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    opportunity_id      UUID REFERENCES stocks.opportunities(id),
    ticker              VARCHAR(10) NOT NULL,
    
    -- Entry
    entry_price         NUMERIC(10,2) NOT NULL,
    entry_date          DATE NOT NULL,
    shares              NUMERIC(10,4) NOT NULL,
    position_size_usd   NUMERIC(10,2) NOT NULL,
    
    -- Risk management
    stop_loss           NUMERIC(10,2) NOT NULL,
    target_price        NUMERIC(10,2) NOT NULL,
    trailing_stop       NUMERIC(10,2),
    
    -- Current state
    current_price       NUMERIC(10,2),
    unrealized_pnl      NUMERIC(10,2),
    unrealized_pnl_pct  NUMERIC(5,2),
    days_held           INTEGER DEFAULT 0,
    
    -- Status
    status              VARCHAR(20) DEFAULT 'open',   -- open/closed/stopped
    exit_price          NUMERIC(10,2),
    exit_date           DATE,
    exit_reason         VARCHAR(50),                   -- 'target', 'stop', 'manual', 'time'
    realized_pnl        NUMERIC(10,2),
    
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_positions_status ON stocks.positions(status);
CREATE INDEX idx_positions_ticker ON stocks.positions(ticker);

-- =============================================================
-- TABLE 5: stocks.market_events
-- Purpose: Detected macro events and correlations
-- Write: Daily (1-5 events)
-- Storage: ~378 KB/year (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.market_events (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    event_date      DATE NOT NULL,
    event_type      VARCHAR(50) NOT NULL,       -- 'oil_spike', 'vix_spike', 'fed_meeting', etc.
    event_detail    VARCHAR(200),               -- human-readable description
    severity        VARCHAR(10) DEFAULT 'low',  -- 'low', 'medium', 'high', 'critical'
    
    -- Correlation actions taken
    actions_suggested   JSONB DEFAULT '[]',     -- [{ticker, action, reason}]
    
    -- Source data
    data_point          NUMERIC(10,4),          -- the value that triggered event
    threshold           NUMERIC(10,4),          -- the threshold it crossed
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_date ON stocks.market_events(event_date DESC);
CREATE INDEX idx_events_type ON stocks.market_events(event_type);

-- =============================================================
-- TABLE 6: stocks.equity_snapshots
-- Purpose: Weekly portfolio value tracking
-- Write: Weekly (1 row)
-- Storage: ~10 KB/year (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.equity_snapshots (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    snapshot_date   DATE NOT NULL UNIQUE,
    
    -- Portfolio state
    total_value         NUMERIC(12,2) NOT NULL,
    cash_balance        NUMERIC(12,2) NOT NULL,
    invested_value      NUMERIC(12,2) NOT NULL,
    open_positions      SMALLINT DEFAULT 0,
    
    -- Performance
    weekly_pnl          NUMERIC(10,2),
    weekly_pnl_pct      NUMERIC(5,2),
    total_pnl           NUMERIC(10,2),
    total_pnl_pct       NUMERIC(5,2),
    
    -- Risk metrics
    total_risk_pct      NUMERIC(5,2),           -- sum of position risks / portfolio
    cash_pct            NUMERIC(5,2),
    max_drawdown_pct    NUMERIC(5,2),
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equity_date ON stocks.equity_snapshots(snapshot_date DESC);

-- =============================================================
-- TABLE 7: stocks.model_versions
-- Purpose: Track trained ML model metadata
-- Write: On training only (weekly/monthly)
-- Storage: ~250 KB/year (negligible)
-- Note: Actual model binaries go to Supabase Storage (separate)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.model_versions (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,       -- 'direction', 'volatility', 'earnings', 'sector_rotation'
    version         VARCHAR(20) NOT NULL,
    
    -- Performance metrics
    accuracy        NUMERIC(5,4),
    precision_score NUMERIC(5,4),
    recall_score    NUMERIC(5,4),
    f1_score        NUMERIC(5,4),
    auc_roc         NUMERIC(5,4),
    
    -- Training metadata
    training_samples    INTEGER,
    feature_count       INTEGER,
    feature_names       JSONB,                  -- ordered list of feature names
    hyperparameters     JSONB,
    
    -- Storage reference
    storage_path        VARCHAR(200),           -- Supabase Storage path to .pkl file
    file_size_kb        INTEGER,
    
    -- Status
    is_active           BOOLEAN DEFAULT FALSE,  -- only one active per model_name
    trained_at          TIMESTAMPTZ NOT NULL,
    deployed_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_name_active ON stocks.model_versions(model_name, is_active) 
    WHERE is_active = TRUE;

-- =============================================================
-- TABLE 8: stocks.trades
-- Purpose: Closed position history (for performance analysis)
-- Write: On position close only
-- Storage: ~100 KB/year (negligible)
-- =============================================================
CREATE TABLE IF NOT EXISTS stocks.trades (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    position_id     UUID REFERENCES stocks.positions(id),
    ticker          VARCHAR(10) NOT NULL,
    
    -- Trade details
    direction       VARCHAR(5) DEFAULT 'long',
    entry_date      DATE NOT NULL,
    exit_date       DATE NOT NULL,
    days_held       INTEGER NOT NULL,
    
    entry_price     NUMERIC(10,2) NOT NULL,
    exit_price      NUMERIC(10,2) NOT NULL,
    shares          NUMERIC(10,4) NOT NULL,
    
    -- P&L
    realized_pnl        NUMERIC(10,2) NOT NULL,
    realized_pnl_pct    NUMERIC(6,2) NOT NULL,
    
    -- Context
    exit_reason         VARCHAR(50) NOT NULL,
    setup_type          VARCHAR(50),
    sector              VARCHAR(50),
    
    -- Scores at entry (for model feedback)
    entry_confidence    SMALLINT,
    entry_total_score   SMALLINT,
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_date ON stocks.trades(exit_date DESC);
CREATE INDEX idx_trades_ticker ON stocks.trades(ticker);

-- =============================================================
-- DATA RETENTION POLICY
-- Run this as a Supabase cron job (pg_cron) or from bot
-- Keeps daily_scores to 180 days max (~21 MB)
-- =============================================================
-- To be called weekly by the bot:
-- DELETE FROM stocks.daily_scores 
-- WHERE score_date < CURRENT_DATE - INTERVAL '180 days';

-- =============================================================
-- ROW LEVEL SECURITY (RLS)
-- Supabase requires RLS. Since this is a single-user bot
-- using the service_role key, we enable RLS but add a
-- permissive policy for the service role.
-- =============================================================
ALTER TABLE stocks.universe ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.daily_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.market_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.equity_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE stocks.trades ENABLE ROW LEVEL SECURITY;

-- Service role bypass (single-user bot)
CREATE POLICY "service_role_all" ON stocks.universe FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.daily_scores FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.opportunities FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.positions FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.market_events FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.equity_snapshots FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.model_versions FOR ALL USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all" ON stocks.trades FOR ALL USING (TRUE) WITH CHECK (TRUE);

-- =============================================================
-- SUPABASE STORAGE BUCKET (for ML models)
-- Create via Supabase Dashboard > Storage > New Bucket
-- Name: stocks-models
-- Public: No
-- Max file size: 10 MB
-- =============================================================
-- Bucket creation must be done in Supabase Dashboard or via API:
-- INSERT INTO storage.buckets (id, name, public) 
-- VALUES ('stocks-models', 'stocks-models', false);
```

---

## DELIVERABLE 2 — DEVELOPMENT PHASES

### Phase 1 — Foundation (Week 1)

**Goal:** Bot starts, connects to all services, fetches data, and sends a basic Telegram message.

| # | File | Key Functions | Description |
|---|------|--------------|-------------|
| 1 | `.env.example` | — | All env var definitions |
| 2 | `requirements.txt` | — | All dependencies with pinned versions |
| 3 | `main.py` | `main()`, `startup()`, `shutdown()` | Entry point, initializes all components |
| 4 | `health.py` | `GET /health`, `GET /status` | FastAPI health check endpoint |
| 5 | `agent/__init__.py` | — | Package init |
| 6 | `agent/persistence.py` | `init_supabase()`, `upsert_universe()`, `insert_daily_scores()`, `get_open_positions()`, `insert_opportunity()`, `update_position()`, `insert_trade()`, `insert_event()`, `insert_equity_snapshot()`, `purge_old_scores()` | All Supabase CRUD operations |
| 7 | `utils/__init__.py` | — | Package init |
| 8 | `utils/data_loader.py` | `fetch_sp500_list()`, `fetch_price_data(ticker, period)`, `fetch_batch_prices(tickers)`, `fetch_macro_data()`, `fetch_fundamentals(ticker)` | yfinance wrapper with caching & rate limiting |
| 9 | `utils/indicators.py` | `calc_rsi()`, `calc_macd()`, `calc_atr()`, `calc_adx()`, `calc_bollinger()`, `calc_ema()`, `calc_vwap()`, `calc_obv()`, `calc_relative_volume()`, `calc_distance_52w()`, `calc_all_indicators(df)` | All technical indicators (pure functions) |
| 10 | `utils/telegram_bot.py` | `init_bot()`, `send_message()`, `send_alert()`, `format_opportunity()`, `format_position_update()`, `format_weekly_summary()`, `format_morning_briefing()` | Telegram integration |
| 11 | `utils/scheduler.py` | `is_market_open()`, `is_trading_day()`, `get_next_run_time()`, `MarketScheduler` class | APScheduler with market hours awareness |
| 12 | `Dockerfile` | — | Multi-stage build, slim image |
| 13 | `.github/workflows/deploy.yml` | — | CI/CD to DigitalOcean |
| 14 | `scripts/setup_digitalocean.sh` | — | Server provisioning script |

**Testable at end of Phase 1:**
- ✅ `python main.py` starts without errors
- ✅ Health endpoint responds at `/health`
- ✅ Fetches S&P 500 list and stores in `stocks.universe`
- ✅ Fetches price data for any ticker via yfinance
- ✅ Calculates all technical indicators for a ticker
- ✅ Sends a test Telegram message
- ✅ Scheduler correctly identifies market hours
- ✅ Supabase read/write works for all tables
- ✅ Docker builds and runs locally

---

### Phase 2 — Intelligence Layers (Week 2)

**Goal:** Bot scans universe, scores stocks, detects events, and sends weekly opportunity alerts.

| # | File | Key Functions | Description |
|---|------|--------------|-------------|
| 1 | `agent/scanner.py` | `scan_universe()`, `filter_by_rs()`, `filter_by_volume()`, `filter_by_setup()`, `get_top_candidates(n)` | Universe filtering pipeline |
| 2 | `agent/scorer.py` | `score_technical(indicators)`, `score_rs(rs_data)`, `score_fundamental(fundamentals)`, `score_sentiment(news)`, `score_total(components)`, `rank_opportunities(scored_list)` | Conviction scoring engine |
| 3 | `agent/events.py` | `check_oil()`, `check_vix()`, `check_dollar()`, `check_treasury()`, `check_gold()`, `check_fed_calendar()`, `check_cpi_calendar()`, `detect_all_events()`, `get_event_correlations(events)` | Macro event detection with correlation map |
| 4 | `agent/portfolio.py` | `calc_position_size(price, stop, portfolio_value)`, `check_portfolio_limits(positions, new_trade)`, `calc_risk_metrics(positions)`, `suggest_exit(position, current_data)`, `rebalance_check()` | Portfolio management & sizing |
| 5 | `utils/sectors.py` | `calc_rs_vs_spy(ticker, periods)`, `calc_rs_vs_sector(ticker, sector_etf)`, `rank_sectors()`, `rank_within_sector(ticker)`, `get_sector_etf(sector_name)` | Relative strength & sector rotation |
| 6 | `utils/position_sizing.py` | `kelly_criterion()`, `fixed_risk_size()`, `max_position_check()`, `cash_reserve_check()` | Position math utilities |
| 7 | `utils/earnings.py` | `get_earnings_dates(ticker)`, `get_earnings_history(ticker)`, `days_to_earnings(ticker)`, `earnings_beat_streak(ticker)` | Earnings calendar & history |
| 8 | `utils/insider.py` | `fetch_insider_filings(ticker)`, `parse_form4()`, `insider_buy_signal(ticker)`, `insider_sell_signal(ticker)` | SEC EDGAR Form 4 parsing |
| 9 | `agent/agent.py` | `weekly_scan()`, `morning_briefing()`, `intraday_monitor()`, `after_market_review()`, `AgentLoop` class | Main orchestration loop |

**Testable at end of Phase 2:**
- ✅ Weekly scan filters 500 → top 50 → top 5 opportunities
- ✅ Scoring produces 0-100 for each stock with breakdown
- ✅ Event detection catches VIX > 25, oil > 3%, etc.
- ✅ Position sizing respects 2% risk, 25% max, 10% cash rules
- ✅ RS scores match manual calculation (spot check 5 tickers)
- ✅ Earnings dates are accurate (spot check 5 tickers)
- ✅ Insider data parses correctly from SEC EDGAR
- ✅ Full weekly Telegram alert sent with top picks
- ✅ Morning briefing sent with macro summary
- ✅ Agent loop runs through full Sunday cycle

---

### Phase 3 — ML Integration (Week 3)

**Goal:** ML models trained locally, uploaded to Supabase, bot downloads and uses them for scoring.

| # | File | Key Functions | Description |
|---|------|--------------|-------------|
| 1 | `agent/feature_config.py` | `DIRECTION_FEATURES`, `VOLATILITY_FEATURES`, `EARNINGS_FEATURES`, `SECTOR_FEATURES`, `build_feature_vector(ticker, data)`, `validate_features(vector)` | Feature definitions & vector building |
| 2 | `agent/ai_model.py` | `load_model(model_name)`, `predict_direction(features)`, `predict_volatility(features)`, `predict_earnings(features)`, `predict_sector_rotation(features)`, `download_model_from_supabase()`, `ModelManager` class | XGBoost inference on server |
| 3 | `scripts/train_model.py` | `prepare_training_data()`, `train_direction_model()`, `train_volatility_model()`, `train_earnings_model()`, `train_sector_model()`, `evaluate_model()`, `upload_to_supabase()` | Local Mac training pipeline |
| 4 | Update `agent/scorer.py` | `score_ml(predictions)` | Integrate ML scores into total score |
| 5 | Update `agent/scanner.py` | `filter_by_ml_score()` | Use ML to pre-filter candidates |
| 6 | Update `agent/agent.py` | `check_model_updates()` | Download new models on Monday morning |

**Testable at end of Phase 3:**
- ✅ Training script runs locally, produces 4 model files
- ✅ Models upload to Supabase Storage successfully
- ✅ Bot downloads models on startup
- ✅ Direction model predicts P(up 5% in 10d) for any ticker
- ✅ Volatility model predicts P(big move in 5d)
- ✅ ML scores integrated into total scoring (weighted)
- ✅ Model version tracked in `stocks.model_versions`
- ✅ Backtesting shows ML adds value vs. technical-only scoring
- ✅ RAM usage stays under 400 MB with models loaded

---

### Phase 4 — GPT-4o & Sentiment Integration (Week 4)

**Goal:** News sentiment scoring, GPT-4o analysis, and polished human-readable alerts.

| # | File | Key Functions | Description |
|---|------|--------------|-------------|
| 1 | `utils/sentiment.py` | `fetch_news(ticker)`, `vader_score(headlines)`, `gpt4o_mini_analyze(text)`, `gpt4o_earnings_analysis(transcript)`, `gpt4o_weekly_briefing(data)`, `aggregate_sentiment(scores)`, `CostTracker` class | VADER + GPT-4o pipeline |
| 2 | Update `agent/scorer.py` | `score_sentiment(news_data)` | Integrate sentiment into total score |
| 3 | Update `agent/agent.py` | `run_gpt_weekly_briefing()`, `run_gpt_exit_helper()` | GPT orchestration in agent loop |
| 4 | Update `utils/telegram_bot.py` | — | Polish alert templates with GPT output |

**Testable at end of Phase 4:**
- ✅ NewsAPI fetches headlines for a ticker (respecting 100/day)
- ✅ VADER produces sentiment score -10 to +10
- ✅ GPT-4o mini writes plain English alerts (< $0.01 each)
- ✅ GPT-4o weekly briefing costs < $0.10
- ✅ CostTracker accurately logs token usage
- ✅ Total GPT cost for a simulated week < $1.25
- ✅ Telegram alerts read naturally in plain English
- ✅ Sentiment score meaningfully affects total score

---

### Phase 5 — Polish & Deploy (Week 5)

**Goal:** Production-ready deployment, monitoring, CI/CD, and paper trading validation.

| # | File | Key Functions | Description |
|---|------|--------------|-------------|
| 1 | `scripts/deploy.sh` | — | Automated deployment to DigitalOcean |
| 2 | `scripts/bot_status.sh` | — | Check systemd service status |
| 3 | `scripts/bot_logs.sh` | — | Tail service logs |
| 4 | `scripts/bot_restart.sh` | — | Restart service safely |
| 5 | `docs/ARCHITECTURE.md` | — | System architecture documentation |
| 6 | `docs/SETUP.md` | — | Setup & deployment guide |
| 7 | `docs/TRADING_LOGIC.md` | — | Trading strategy documentation |
| 8 | Update `Dockerfile` | — | Production optimizations |
| 9 | Update `deploy.yml` | — | Full CI/CD pipeline |
| 10 | `render.yaml` | — | Legacy/backup deploy config |
| 11 | All files | — | Error handling hardening, logging, edge cases |

**Testable at end of Phase 5:**
- ✅ `deploy.sh` deploys to DigitalOcean without manual steps
- ✅ GitHub push → automatic deploy via Actions
- ✅ Bot runs for 7 days without crash
- ✅ Memory stays under 400 MB for 7 days
- ✅ Crypto bot unaffected (RAM, CPU, no port conflicts)
- ✅ Paper trading produces realistic recommendations
- ✅ Telegram alerts arrive on schedule
- ✅ All data persists in Supabase correctly
- ✅ Recovery after reboot (systemd auto-restart)
- ✅ Logs capture all decisions with reasoning

---

## DELIVERABLE 3 — FEATURE PRIORITY LIST (MoSCoW)

### MUST HAVE (Launch blockers)

| # | Feature | Impact | Complexity | Cost/mo | Phase |
|---|---------|--------|-----------|---------|-------|
| 1 | Technical indicators (RSI, MACD, ATR, ADX, BB) | ★★★★★ | Low | $0 | 1 |
| 2 | yfinance data fetching | ★★★★★ | Low | $0 | 1 |
| 3 | Supabase persistence | ★★★★★ | Medium | $0 | 1 |
| 4 | Telegram alerts | ★★★★★ | Low | $0 | 1 |
| 5 | Market hours scheduler | ★★★★★ | Medium | $0 | 1 |
| 6 | RS score vs SPY | ★★★★★ | Medium | $0 | 2 |
| 7 | Position sizing (2% risk) | ★★★★★ | Medium | $0 | 2 |
| 8 | Portfolio limits (25% max, 10% cash) | ★★★★★ | Medium | $0 | 2 |
| 9 | Weekly scan → top 5 picks | ★★★★★ | High | $0 | 2 |
| 10 | Total conviction score (0-100) | ★★★★★ | High | $0 | 2 |
| 11 | Intraday position monitoring | ★★★★☆ | Medium | $0 | 2 |
| 12 | Stop loss / target alerts | ★★★★★ | Low | $0 | 2 |
| 13 | Health check endpoint | ★★★★☆ | Low | $0 | 1 |
| 14 | CI/CD deploy pipeline | ★★★★☆ | Medium | $0 | 1+5 |

### SHOULD HAVE (High value, week 2-3)

| # | Feature | Impact | Complexity | Cost/mo | Phase |
|---|---------|--------|-----------|---------|-------|
| 15 | 50/200 EMA (golden/death cross) | ★★★★☆ | Low | $0 | 1 |
| 16 | VWAP, OBV, relative volume | ★★★★☆ | Low | $0 | 1 |
| 17 | Sector rank (1-11) | ★★★★☆ | Medium | $0 | 2 |
| 18 | Event detection (VIX, oil, dollar) | ★★★★☆ | Medium | $0 | 2 |
| 19 | Event correlation map | ★★★★☆ | Medium | $0 | 2 |
| 20 | Earnings date awareness | ★★★★☆ | Low | $0 | 2 |
| 21 | XGBoost direction model | ★★★★☆ | High | $0 | 3 |
| 22 | XGBoost volatility model | ★★★★☆ | High | $0 | 3 |
| 23 | VADER sentiment scoring | ★★★★☆ | Low | $0 | 4 |
| 24 | GPT-4o mini alert writing | ★★★☆☆ | Medium | ~$2 | 4 |
| 25 | Distance from 52-week high/low | ★★★☆☆ | Low | $0 | 1 |
| 26 | Daily score persistence | ★★★☆☆ | Low | $0 | 2 |

### COULD HAVE (Nice to have, week 4-5)

| # | Feature | Impact | Complexity | Cost/mo | Phase |
|---|---------|--------|-----------|---------|-------|
| 27 | GPT-4o weekly briefing | ★★★☆☆ | Medium | ~$1 | 4 |
| 28 | GPT-4o earnings analysis | ★★★☆☆ | Medium | ~$1 | 4 |
| 29 | SEC EDGAR insider parsing | ★★★☆☆ | High | $0 | 2 |
| 30 | XGBoost earnings beat predictor | ★★★☆☆ | High | $0 | 3 |
| 31 | XGBoost sector rotation model | ★★☆☆☆ | High | $0 | 3 |
| 32 | Pattern detection (cup/handle, flags) | ★★★☆☆ | High | $0 | 3 |
| 33 | Short interest data | ★★☆☆☆ | Medium | $0 | 2 |
| 34 | PE ratio vs sector average | ★★☆☆☆ | Low | $0 | 2 |
| 35 | GPT-4o exit decision helper | ★★☆☆☆ | Medium | ~$1 | 4 |
| 36 | Alpha Vantage backup data | ★★☆☆☆ | Low | $0 | 5 |
| 37 | Prophet forecasting (local Mac) | ★★☆☆☆ | Medium | $0 | 3 |
| 38 | Revenue growth trend | ★★☆☆☆ | Low | $0 | 2 |

### WON'T HAVE (V1 — defer to V2)

| # | Feature | Reason |
|---|---------|--------|
| 39 | FinBERT sentiment (local Mac) | VADER is sufficient for V1, FinBERT adds complexity |
| 40 | Industry rank within sector | Requires too much data for 25 req/day Alpha Vantage |
| 41 | Stock rank within industry | Same as above |
| 42 | Days to cover (short squeeze) | Data quality from free sources is unreliable |
| 43 | Sympathy play detection | Complex, needs more data history to validate |
| 44 | Pocket pivot detection | Low frequency signal, hard to validate |
| 45 | FRED API integration | Nice to have but VIX/treasury via yfinance covers this |

> **⚠️ FLAG:** Items 40-42 (industry rank, stock rank within industry, days to cover) are listed in requirements but free data sources don't provide reliable granularity. Recommend deferring to V2 when a paid data source might be justified.

---

## DELIVERABLE 4 — RISK ASSESSMENT

### Component Risk Matrix

| Component | What Can Go Wrong | Probability | Severity | Graceful Handling | Fallback Behavior |
|-----------|------------------|-------------|----------|-------------------|--------------------|
| **yfinance** | Rate limited, data delayed, API changes, weekend gaps | Medium | High | Exponential backoff with 3 retries, cache last-good data | Use Alpha Vantage backup. If both fail, skip scan, send "⚠️ Data unavailable" to Telegram, retry next cycle |
| **Supabase** | Connection timeout, 500MB limit hit, free tier outage | Low | Critical | Connection pool with retry, monitor storage via query, alert at 400MB | Queue writes to local SQLite file, replay when Supabase recovers. Emergency purge of `daily_scores` > 90 days |
| **Telegram** | Bot blocked, rate limited (30 msg/sec), message too long | Low | Medium | Retry 3x with backoff, split long messages into chunks | Log alerts to file. On next successful send, include missed alert summary |
| **NewsAPI** | 100/day limit exhausted, API down, irrelevant results | Medium | Low | Track calls in memory counter, stop fetching when at 90 calls | Skip sentiment for remaining stocks. Set sentiment_score = 0 (neutral). Log warning |
| **Alpha Vantage** | 25/day limit (very tight), slow response | High | Low | Use only as fallback for yfinance failures, never as primary | Return None, scorer treats missing AV data as neutral |
| **OpenAI API** | Cost overrun, rate limit, API down, hallucination | Medium | Medium | CostTracker class enforces hard $5/month cap, kill switch at $4 | If budget exceeded: use template strings instead of GPT for alerts. If API down: use pre-formatted templates |
| **XGBoost models** | Model file corrupted, poor predictions, stale model | Low | Medium | Validate model on load (predict on 5 known samples, check range). Track model age, alert if > 30 days old | If model fails to load: disable ML scoring layer, use technical + RS only. Score weighted accordingly |
| **SEC EDGAR** | Parsing failures (HTML changes), slow response, missing data | Medium | Low | Try/except per filing, validate parsed fields | Skip insider data for that ticker, set insider_signal = 'unknown' |
| **DigitalOcean** | RAM exceeded (OOM kill), CPU spike, disk full | Medium | Critical | Monitor RSS memory every 60s, alert at 380MB. Graceful shutdown if approaching 400MB. Log rotate to prevent disk fill | If OOM: systemd auto-restart with 10s delay. Reduce batch sizes (scan 100 stocks instead of 500). Emergency mode: positions only |
| **Scheduler** | Timezone bugs, DST transitions, missed schedules | Medium | High | Use `pytz` with explicit `America/New_York`. Log every scheduled run. Detect and alert on missed runs | If schedule missed: run on next check cycle. Sunday scan can run Monday 6AM as backup |
| **Market data quality** | Stock splits, delistings, ticker changes, halted stocks | Medium | Medium | Validate price data (no >50% day moves unless confirmed). Check for splits in yfinance metadata | Flag suspicious data, skip stock for that cycle, alert to Telegram for manual review |
| **Concurrent bots** | Crypto bot + stocks bot compete for RAM/CPU | Medium | High | Monitor combined memory at startup. If crypto bot using >400MB, reduce stocks batch size | Emergency mode: scan top 100 only, skip ML models, reduce monitoring frequency to 30 min |

### Catastrophic Failure Recovery

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Bot crash loop | systemd `Restart=on-failure` with `RestartSec=30`, `StartLimitBurst=5` | After 5 crashes in 5 min: stop and send Telegram "🔴 Bot crashed, needs manual restart" |
| Supabase data corruption | Checksum on equity_snapshots total. If weekly P&L > ±50%, flag | Alert to Telegram, freeze all recommendations, require manual review |
| Model producing bad signals | Track model prediction accuracy rolling 30 days. Alert if accuracy < 40% | Auto-disable ML layer, revert to technical+RS scoring only |
| Cost overrun (OpenAI) | CostTracker checks after every API call | Hard stop at $5/month. Switch to template-based alerts for rest of month |

---

## DELIVERABLE 5 — API RATE LIMIT PLAN

### yfinance (Unofficial Yahoo Finance)

| Metric | Value |
|--------|-------|
| **Official limit** | None (unofficial API, no contract) |
| **Practical limit** | ~2,000 requests/hour before throttling |
| **Bot daily need** | ~600-800 calls (500 universe + macro + position updates) |
| **Strategy** | Batch download via `yf.download(tickers, period)` — 1 call for all 500 tickers |
| **Caching** | Cache price data in memory for 15 minutes during market hours, 24h outside |
| **Fallback** | Alpha Vantage for individual tickers if yfinance fails |

**Detailed call breakdown:**

| Operation | When | Calls | Notes |
|-----------|------|-------|-------|
| S&P 500 batch download (1Y data) | Sunday scan | 1 | `yf.download(all_500, period='1y')` |
| Macro data (VIX, oil, gold, dollar, 10yr) | Daily 8:30 AM | 5 | Individual `yf.Ticker().history()` |
| Position updates (3-5 open positions) | Every 15 min × 26 slots | ~130 | `yf.download(positions, period='1d')` batched |
| Fundamental data (top 50 candidates) | Sunday scan | 50 | `yf.Ticker().info` |
| Earnings dates (top 50) | Sunday scan | 50 | `yf.Ticker().calendar` |
| **Daily total (weekday)** | — | **~185** | Well within practical limits |
| **Sunday total** | — | **~105** | Batch download is key |

### NewsAPI (Free Tier)

| Metric | Value |
|--------|-------|
| **Hard limit** | 100 requests/day |
| **Bot daily need** | 15-25 calls |
| **Strategy** | Fetch news for top 10 scored stocks + 5 open positions only |
| **Caching** | Cache results for 6 hours (news doesn't change that fast) |
| **Fallback** | Skip sentiment, set score to 0 (neutral) |

**Detailed call breakdown:**

| Operation | When | Calls | Notes |
|-----------|------|-------|-------|
| Top 10 stocks headlines | Morning 8:30 AM | 10 | `everything?q={ticker}&sortBy=relevancy` |
| Open positions (3-5) | Morning 8:30 AM | 5 | Same endpoint |
| Market-wide news | Morning 8:30 AM | 1 | `top-headlines?category=business` |
| Midday update (top 5) | 12:30 PM | 5 | Only on high-volatility days |
| **Daily total** | — | **~21** | 79 calls buffer remaining |

### Alpha Vantage (Free Tier)

| Metric | Value |
|--------|-------|
| **Hard limit** | 25 requests/day, 5/minute |
| **Bot daily need** | 0 (backup only) |
| **Strategy** | Reserved entirely as fallback for yfinance failures |
| **Caching** | Cache for 24 hours |
| **Fallback** | If AV also fails, use last cached data |

**Detailed call breakdown:**

| Operation | When | Calls | Notes |
|-----------|------|-------|-------|
| yfinance failure backup | On failure only | 0-10 | `TIME_SERIES_DAILY` endpoint |
| Supplemental data | Never in V1 | 0 | Reserved for V2 features |
| **Daily total** | — | **0-10** | 15-25 calls buffer remaining |

### OpenAI API

| Metric | Value |
|--------|-------|
| **Rate limit** | 500 RPM (GPT-4o), 500 RPM (GPT-4o-mini) |
| **Monthly budget** | $5.00 hard cap |
| **Strategy** | GPT-4o-mini for daily tasks, GPT-4o for weekly only |

**Detailed cost breakdown:**

| Operation | Frequency | Model | Tokens/call | Cost/call | Monthly cost |
|-----------|-----------|-------|-------------|-----------|-------------|
| Format Telegram alert | 5/week | GPT-4o-mini | ~500 | $0.0001 | $0.002 |
| Morning briefing write | 5/week (daily) | GPT-4o-mini | ~800 | $0.0001 | $0.002 |
| News analysis (flagged) | 10/week | GPT-4o-mini | ~1,000 | $0.0002 | $0.003 |
| Exit decision helper | 5/week | GPT-4o-mini | ~600 | $0.0001 | $0.002 |
| Weekly market briefing | 1/week | GPT-4o | ~3,000 | $0.0075 | $0.03 |
| Earnings analysis | 2/month | GPT-4o | ~5,000 | $0.0125 | $0.025 |
| **Monthly total** | — | — | — | — | **~$0.06** |

> **⚠️ NOTE:** Estimated cost is well under $5/month. Even at 10x the estimates, it would be ~$0.60/month. The $5 budget provides massive headroom. Consider using GPT-4o more liberally if results warrant it.

### SEC EDGAR

| Metric | Value |
|--------|-------|
| **Rate limit** | 10 requests/second (with User-Agent header) |
| **Bot daily need** | 5-15 requests (after-market check) |
| **Strategy** | Only check top candidates and open positions |
| **Caching** | Cache for 24 hours |
| **Fallback** | Skip insider data, treat as neutral |

### Summary Rate Limit Dashboard

| Source | Daily Limit | Daily Usage | Buffer | Risk Level |
|--------|------------|-------------|--------|------------|
| yfinance | ~2,000 | ~185 | 91% free | 🟢 Low |
| NewsAPI | 100 | ~21 | 79% free | 🟢 Low |
| Alpha Vantage | 25 | 0-10 | 60-100% free | 🟢 Low |
| OpenAI | 500 RPM | ~5/day | 99% free | 🟢 Low |
| SEC EDGAR | 10/sec | ~10/day | 99% free | 🟢 Low |

---

## DELIVERABLE 6 — RAM BUDGET PLAN

### Target: ≤ 400 MB RSS

| Component | Estimated RAM | Notes |
|-----------|--------------|-------|
| **Python 3.11 interpreter** | 30 MB | Base interpreter + builtins |
| **Imported libraries (idle)** | | |
| — pandas | 35 MB | DataFrame engine |
| — numpy | 20 MB | Numerical arrays |
| — xgboost | 25 MB | Model runtime (C++ core) |
| — scikit-learn (metrics only) | 15 MB | Only import specific modules, not all |
| — FastAPI + uvicorn | 15 MB | Health endpoint |
| — APScheduler | 5 MB | Scheduler |
| — python-telegram-bot | 8 MB | Telegram SDK |
| — supabase-py | 5 MB | DB client |
| — openai SDK | 5 MB | API client |
| — vaderSentiment | 3 MB | Lexicon + model |
| — yfinance | 8 MB | Yahoo Finance wrapper |
| **Data in memory** | | |
| — S&P 500 × 1Y daily OHLCV | 50 MB | ~500 tickers × 252 rows × ~400 bytes. Loaded during scan, freed after |
| — Top 50 candidate DataFrames | 10 MB | After filtering, only keep 50 |
| — Macro data (5 tickers × 1Y) | 1 MB | Small dataset |
| — Open positions cache | 1 MB | 3-5 positions |
| **ML Models in memory** | | |
| — Direction model (.pkl) | 5 MB | XGBoost model loaded |
| — Volatility model (.pkl) | 3 MB | Smaller model |
| — Earnings model (.pkl) | 2 MB | Smallest model |
| — Sector rotation model (.pkl) | 2 MB | Smallest model |
| **Working memory (temp)** | | |
| — Indicator calculations | 20 MB | Temp arrays during calc, freed after |
| — News/sentiment data | 5 MB | Headlines in memory |
| — GPT request/response buffers | 2 MB | Short-lived |
| **OS/process overhead** | 15 MB | Thread stacks, file descriptors, etc. |

| **Category** | **Subtotal** |
|-------------|-------------|
| Python + libraries | 174 MB |
| Data in memory | 62 MB |
| ML models | 12 MB |
| Working memory (temporary) | 27 MB |
| OS overhead | 15 MB |
| **TOTAL ESTIMATE** | **290 MB** |
| **Safety margin** | **110 MB** (28%) |
| **Budget** | **400 MB** |

### RAM Optimization Strategies

1. **Lazy loading:** Don't load all 500 stocks at once. Use `yf.download()` which returns a single DataFrame, then process in chunks of 50
2. **Garbage collection:** Explicitly `del` large DataFrames after processing and call `gc.collect()`
3. **Selective imports:** `from sklearn.metrics import accuracy_score` not `import sklearn`
4. **Model unloading:** Load ML models only during scoring, unload after (saves ~12 MB during idle monitoring)
5. **DataFrame dtypes:** Use `float32` instead of `float64` for price data (halves memory)
6. **Memory monitoring:** Log RSS memory every 60 seconds, alert at 380 MB

### Peak Memory Timeline

| Time | Activity | Estimated RSS | Notes |
|------|----------|--------------|-------|
| Idle (night/weekend) | Sleeping | 180 MB | Just interpreter + libraries |
| Sunday 8 PM (weekly scan) | Full 500-stock scan | **310 MB** ← Peak | Load all data, score, then free |
| Weekday 8:30 AM | Morning briefing | 220 MB | News + macro data |
| Weekday 9:30-4 PM | Position monitoring | 195 MB | Only 3-5 tickers |
| Weekday 4:30 PM | After-market review | 230 MB | Score updates + insider check |

> **✅ VERDICT:** Peak 310 MB is well under 400 MB budget. Combined with crypto bot's ~350 MB, total ~660 MB fits safely under 1 GB droplet.

> **⚠️ FLAG:** The 1 GB droplet has ~950 MB usable after OS overhead. Combined 660 MB peak leaves 290 MB for OS, which is comfortable. However, if both bots peak simultaneously (unlikely — different market hours), it could reach ~750 MB. Monitor closely in Week 5.

---

## DELIVERABLE 7 — DEPENDENCY MAP

### Build Order (Critical Path)

```
Level 0 (No dependencies — build first, in parallel):
├── .env.example
├── requirements.txt
├── Dockerfile
├── .github/workflows/deploy.yml
├── scripts/setup_digitalocean.sh
├── utils/__init__.py
└── agent/__init__.py

Level 1 (Foundation utilities — build in parallel):
├── utils/indicators.py        ← Pure functions, no deps
├── utils/position_sizing.py   ← Pure functions, no deps
├── utils/scheduler.py         ← Only depends on pytz, apscheduler
└── utils/telegram_bot.py      ← Only depends on telegram SDK + env vars

Level 2 (Data layer — build in parallel):
├── utils/data_loader.py       ← Depends on: yfinance
├── utils/earnings.py          ← Depends on: yfinance
├── utils/insider.py           ← Depends on: requests (SEC EDGAR)
├── utils/sectors.py           ← Depends on: data_loader, indicators
└── agent/persistence.py       ← Depends on: supabase-py + env vars

Level 3 (Intelligence — sequential, depends on Level 1+2):
├── agent/events.py            ← Depends on: data_loader
├── agent/scanner.py           ← Depends on: data_loader, indicators, sectors, persistence
├── agent/scorer.py            ← Depends on: indicators, sectors, events
└── agent/portfolio.py         ← Depends on: position_sizing, persistence

Level 4 (ML — depends on Level 2+3):
├── agent/feature_config.py    ← Depends on: indicators (for feature names)
├── agent/ai_model.py          ← Depends on: feature_config, persistence (download models)
└── scripts/train_model.py     ← Depends on: feature_config, data_loader (local Mac only)

Level 5 (AI — depends on Level 3):
└── utils/sentiment.py         ← Depends on: openai, vaderSentiment, data_loader

Level 6 (Orchestration — depends on everything):
├── agent/agent.py             ← Depends on: ALL agent/ and utils/ modules
└── main.py                    ← Depends on: agent.py, health.py, scheduler
    └── health.py              ← Depends on: FastAPI, persistence (status check)

Level 7 (Polish — depends on Level 6):
├── scripts/deploy.sh
├── scripts/bot_status.sh
├── scripts/bot_logs.sh
├── scripts/bot_restart.sh
├── docs/ARCHITECTURE.md
├── docs/SETUP.md
└── docs/TRADING_LOGIC.md
```

### Visual Dependency Graph

```
main.py
├── health.py ──→ persistence.py ──→ supabase
├── agent/agent.py
│   ├── scanner.py
│   │   ├── data_loader.py ──→ yfinance
│   │   ├── indicators.py (pure)
│   │   ├── sectors.py ──→ data_loader
│   │   └── persistence.py
│   ├── scorer.py
│   │   ├── indicators.py (pure)
│   │   ├── sectors.py
│   │   ├── events.py ──→ data_loader
│   │   ├── ai_model.py ──→ feature_config, persistence
│   │   └── sentiment.py ──→ openai, vader, data_loader
│   ├── portfolio.py
│   │   ├── position_sizing.py (pure)
│   │   └── persistence.py
│   ├── events.py ──→ data_loader
│   └── persistence.py
├── utils/telegram_bot.py ──→ telegram SDK
└── utils/scheduler.py ──→ apscheduler
```

### Parallel Build Opportunities

| Phase | Can Build in Parallel | Sequential Requirement |
|-------|----------------------|----------------------|
| Week 1 | `indicators.py`, `position_sizing.py`, `telegram_bot.py`, `scheduler.py` all in parallel | `data_loader.py` before `sectors.py`. `persistence.py` before `main.py` |
| Week 2 | `events.py`, `earnings.py`, `insider.py` in parallel | `scanner.py` → `scorer.py` → `portfolio.py` → `agent.py` (sequential) |
| Week 3 | `feature_config.py` and `train_model.py` in parallel | `feature_config.py` before `ai_model.py` |
| Week 4 | `sentiment.py` standalone | Integration into `scorer.py` and `agent.py` sequential |
| Week 5 | All scripts and docs in parallel | Final integration testing sequential |

---

## DELIVERABLE 8 — TESTING STRATEGY

### Test Structure

```
tests/
├── conftest.py                 — Shared fixtures (mock data, mock Supabase)
├── unit/
│   ├── test_indicators.py      — Pure function tests (highest coverage)
│   ├── test_position_sizing.py — Math validation
│   ├── test_scorer.py          — Scoring logic
│   ├── test_portfolio.py       — Portfolio rules
│   ├── test_events.py          — Event detection thresholds
│   ├── test_sectors.py         — RS calculations
│   ├── test_feature_config.py  — Feature vector shape
│   ├── test_scheduler.py       — Market hours logic
│   └── test_data_loader.py     — Data validation
├── integration/
│   ├── test_supabase.py        — CRUD operations (uses test schema)
│   ├── test_telegram.py        — Message sending (uses test chat)
│   ├── test_yfinance.py        — Data fetching (live API)
│   ├── test_newsapi.py         — News fetching (1 call only)
│   ├── test_agent_loop.py      — Full scan cycle
│   └── test_ml_pipeline.py     — Model load + predict
└── validation/
    ├── paper_trading_log.py    — Record recommendations vs actual
    └── backtest_scorer.py      — Score historical data, measure hit rate
```

### Unit Tests (Run on every commit)

| Module | Tests | Key Assertions |
|--------|-------|----------------|
| `indicators.py` | 30+ tests | RSI(14) for known AAPL data = expected value ± 0.01. MACD crossover detected correctly. ATR matches manual calculation. All functions handle NaN/empty input gracefully |
| `position_sizing.py` | 15 tests | 2% risk rule: $10,000 portfolio, $5 stop distance → $200 risk → 40 shares. Max 25% position enforced. 10% cash reserve enforced. Edge cases: $0 stop, negative price |
| `scorer.py` | 20 tests | RSI < 30 → high technical score. RS > 80 → high RS score. Total score = weighted sum. Score always 0-100 |
| `portfolio.py` | 15 tests | Can't exceed 6% total risk. Can't exceed 25% single position. Cash reserve maintained. Position sizing matches formula |
| `events.py` | 15 tests | VIX > 25 triggers event. Oil change > 3% triggers event. Fed meeting date detection. Correlation map returns correct tickers |
| `scheduler.py` | 20 tests | 9:30 AM ET Monday = market open. 9:30 AM ET Saturday = market closed. DST transition handled. Pre/post market times correct |
| `feature_config.py` | 10 tests | Feature vector has correct length. No NaN in output. Feature names match model expectation |

### Integration Tests (Run on deploy)

| Test | What It Validates | Max Duration |
|------|------------------|-------------|
| Supabase CRUD | Insert, read, update, delete for all 8 tables | 30s |
| yfinance fetch | Fetch AAPL 1Y data, validate OHLCV columns | 15s |
| Telegram send | Send test message to test chat ID | 5s |
| NewsAPI fetch | Fetch 1 headline for AAPL | 5s |
| Full scan (mock) | Scanner → Scorer → Portfolio → Telegram (with mocked APIs) | 60s |
| ML model load | Load all 4 models, predict on test vector | 10s |

### Paper Trading Validation (Week 5)

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Signal accuracy | > 55% of "buy" signals profitable within 3 weeks | Track entry vs. price 15 trading days later |
| Risk/reward ratio | Average winning trade > 1.5x average losing trade | Compare realized P&L |
| Score correlation | Higher confidence (80+) should have higher win rate than lower (60-70) | Bucket trades by confidence, compare win rates |
| False positive rate | < 30% of buy signals hit stop loss within 5 days | Track stop hits |
| Timing accuracy | > 50% of signals enter within 2% of suggested entry range | Compare suggested vs. actual entry prices |

### Performance Benchmarks

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Sunday scan duration | < 10 minutes | Time the full `weekly_scan()` |
| Morning briefing duration | < 2 minutes | Time `morning_briefing()` |
| Intraday monitor cycle | < 30 seconds | Time single `intraday_monitor()` |
| Memory peak (Sunday scan) | < 350 MB | `psutil.Process().memory_info().rss` |
| Memory idle | < 200 MB | Same measurement during monitoring |
| Model inference time | < 100ms per stock | Time `predict_direction()` |

---

## DELIVERABLE 9 — CODING STANDARDS

### File Structure Conventions

```python
"""
Module docstring: What this file does, one line.

Detailed description if needed.
"""

# Standard library imports
import os
import json
from datetime import datetime, timezone

# Third-party imports
import pandas as pd
import numpy as np
from supabase import create_client

# Local imports
from utils.data_loader import fetch_price_data
from agent.persistence import insert_daily_scores

# Constants (UPPER_SNAKE_CASE)
MAX_POSITION_PCT = 0.25
RISK_PER_TRADE = 0.02
CASH_RESERVE_PCT = 0.10

# Module-level logger
import logging
logger = logging.getLogger(__name__)
```

### Error Handling Pattern

```python
# Standard try/except with logging and fallback
async def fetch_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch price data for a ticker. Returns None on failure."""
    try:
        df = await data_loader.fetch_price_data(ticker, period="1y")
        if df is None or df.empty:
            logger.warning(f"Empty data for {ticker}")
            return None
        return df
    except ConnectionError as e:
        logger.error(f"Connection error fetching {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {ticker}: {e}", exc_info=True)
        return None
```

**Rules:**
- Never use bare `except:` — always catch specific exceptions
- Always log the error with context (ticker, function, relevant params)
- Always have a fallback return value (`None`, `0`, empty dict)
- Use `exc_info=True` for unexpected exceptions (includes traceback)
- Never let an exception crash the main agent loop

### Logging Format

```python
# Configure in main.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "logs/stocks-agent.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=3
        )
    ]
)

# Usage examples:
logger.info(f"Weekly scan started | universe={500}")
logger.info(f"Top candidate | ticker=NVDA score=87 rs=92 tech=85")
logger.warning(f"NewsAPI limit approaching | used=85 limit=100")
logger.error(f"yfinance failed | ticker=AAPL retry=3/3")
```

**Log format:** `2026-03-01 08:30:00 | INFO    | agent.scanner             | Weekly scan started | universe=500`

### Supabase Query Patterns

```python
# INSERT (single)
result = supabase.table("daily_scores").insert({
    "ticker": ticker,
    "score_date": date.isoformat(),
    "total_score": score,
}).execute()

# INSERT (batch — preferred for daily_scores)
rows = [{"ticker": t, "score_date": d, "total_score": s} for t, s in scores.items()]
result = supabase.table("daily_scores").insert(rows).execute()

# UPSERT (for universe refresh)
result = supabase.table("universe").upsert(
    rows, on_conflict="ticker"
).execute()

# SELECT with filters
result = supabase.table("positions") \
    .select("*") \
    .eq("status", "open") \
    .order("created_at", desc=True) \
    .execute()

# UPDATE
result = supabase.table("positions") \
    .update({"current_price": price, "updated_at": "now()"}) \
    .eq("id", position_id) \
    .execute()

# DELETE (data purge)
result = supabase.table("daily_scores") \
    .delete() \
    .lt("score_date", cutoff_date.isoformat()) \
    .execute()
```

**Rules:**
- Always use the `stocks` schema (configure in client or prefix)
- Always `.execute()` at the end
- Always check `result.data` for success
- Batch inserts for `daily_scores` (500 rows at once, not 500 individual calls)
- Use `upsert` with `on_conflict` for idempotent operations

### Telegram Alert Format

```python
# Template constants in telegram_bot.py
WEEKLY_HEADER = "📊 WEEKLY STOCK OPPORTUNITIES\n\n"
DAILY_HEADER = "☀️ MORNING BRIEFING — {date}\n\n"
ACTION_HEADER = "⚠️ ACTION NEEDED — {ticker}\n\n"
EOD_HEADER = "🌙 END OF DAY SUMMARY — {date}\n\n"

# Emoji conventions:
# 🟢 Bullish / Positive
# 🔴 Bearish / Negative  
# 🟡 Neutral / Caution
# ⚠️ Action required
# 📊 Data / Analysis
# 💰 Money / P&L
# 🎯 Target
# 🛑 Stop loss
# ☀️ Morning
# 🌙 Evening

# Message length: max 4096 chars per Telegram message
# If longer: split into multiple messages with "1/3", "2/3", "3/3"
```

### Environment Variable Naming

```bash
# Prefix: STOCKS_ for all stocks-agent variables
# This avoids collision with crypto bot variables

# API Keys
STOCKS_SUPABASE_URL=
STOCKS_SUPABASE_KEY=        # service_role key
STOCKS_TELEGRAM_TOKEN=       # same bot, different chat commands
STOCKS_TELEGRAM_CHAT_ID=
STOCKS_OPENAI_API_KEY=
STOCKS_NEWSAPI_KEY=
STOCKS_ALPHA_VANTAGE_KEY=

# Configuration
STOCKS_PORTFOLIO_VALUE=10000
STOCKS_MAX_RISK_PCT=0.02
STOCKS_MAX_POSITION_PCT=0.25
STOCKS_CASH_RESERVE_PCT=0.10

# Feature flags
STOCKS_ENABLE_ML=true
STOCKS_ENABLE_GPT=true
STOCKS_ENABLE_NEWS=true
STOCKS_DRY_RUN=false         # true = log but don't send alerts

# Environment
STOCKS_ENV=production        # production / development / test
STOCKS_LOG_LEVEL=INFO
```

### Type Hints

```python
# Always use type hints for function signatures
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime

def score_technical(
    indicators: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> int:
    """Calculate technical score 0-100 from indicator values."""
    ...

def calc_position_size(
    price: float,
    stop_loss: float,
    portfolio_value: float,
    risk_pct: float = 0.02
) -> Tuple[int, float]:
    """Returns (shares, position_value_usd)."""
    ...
```

### Data Classes for Core Types

```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import date

@dataclass
class Opportunity:
    ticker: str
    confidence: int          # 0-100
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    position_size_usd: float
    risk_usd: float
    reasons: List[str]
    setup_type: str
    sector: str

@dataclass
class Position:
    id: str
    ticker: str
    entry_price: float
    shares: float
    stop_loss: float
    target: float
    status: str              # 'open', 'closed', 'stopped'

@dataclass
class MarketEvent:
    event_type: str
    severity: str            # 'low', 'medium', 'high', 'critical'
    detail: str
    affected_tickers: List[str]
    suggested_action: str
```

---

## DELIVERABLE 10 — FIRST PROMPT

> **Copy this prompt exactly into GitHub Copilot to begin Phase 1 building.**

---

```
You are building Phase 1 of a stocks trading advisory bot called stocks-agent.
This is a NEW standalone project in the /Users/dorp/stocks-trading directory.

CONTEXT:
- See DEVELOPMENT_PLAN.md in the workspace for the complete architecture
- This is Phase 1: Foundation (Week 1)
- The bot advises on US stock swing trades (1-6 weeks hold)
- Advisory only — suggests trades, human executes
- Must run on DigitalOcean droplet (400MB RAM budget)
- Uses Supabase (stocks schema), Telegram, yfinance

BUILD THE FOLLOWING FILES IN ORDER:

1. requirements.txt
   Pin all versions. Include:
   fastapi==0.109.0, uvicorn==0.27.0, python-telegram-bot==20.7,
   supabase==2.3.4, yfinance==0.2.36, pandas==2.2.0, numpy==1.26.3,
   ta==0.11.0, xgboost==2.0.3, scikit-learn==1.4.0,
   apscheduler==3.10.4, openai==1.12.0, vaderSentiment==3.3.2,
   python-dotenv==1.0.1, httpx==0.27.0, psutil==5.9.8,
   newsapi-python==0.2.7

2. .env.example
   All STOCKS_ prefixed env vars as defined in DEVELOPMENT_PLAN.md
   Deliverable 9 Coding Standards section.

3. utils/__init__.py (empty)
4. agent/__init__.py (empty)

5. utils/indicators.py
   Implement ALL these as pure functions taking a pandas DataFrame (OHLCV):
   - calc_rsi(df, period=14) -> Series
   - calc_macd(df) -> DataFrame with macd, signal, histogram
   - calc_atr(df, period=14) -> Series
   - calc_adx(df, period=14) -> Series
   - calc_bollinger(df, period=20, std=2) -> DataFrame with upper, middle, lower, width
   - calc_ema(df, period) -> Series
   - calc_vwap(df) -> Series
   - calc_obv(df) -> Series
   - calc_relative_volume(df, period=50) -> float (latest vol / avg vol)
   - calc_distance_52w(df) -> dict with {high_pct, low_pct}
   - calc_all_indicators(df) -> dict with ALL indicator values
   Use the ta library internally. No API calls. No side effects.
   Every function must handle empty/NaN DataFrames gracefully (return None).
   Include docstrings and type hints on every function.

6. utils/data_loader.py
   - fetch_sp500_list() -> List[dict] with ticker, name, sector, industry
     (scrape Wikipedia S&P 500 page with pandas.read_html)
   - fetch_price_data(ticker, period="1y", interval="1d") -> DataFrame
     (yfinance wrapper with retry logic)
   - fetch_batch_prices(tickers, period="1y") -> Dict[str, DataFrame]
     (use yf.download for efficiency, split result by ticker)
   - fetch_macro_data() -> dict
     (fetch VIX, oil, gold, dollar, 10yr via yfinance)
   - fetch_fundamentals(ticker) -> dict
     (market cap, PE, sector, earnings dates from yf.Ticker.info)
   Add retry decorator (3 retries, exponential backoff).
   Add 15-minute in-memory cache for price data.
   All functions must handle errors and return None on failure.

7. agent/persistence.py
   - init_supabase() -> Client
   - upsert_universe(stocks: List[dict]) -> bool
   - insert_daily_scores(scores: List[dict]) -> bool
   - get_open_positions() -> List[dict]
   - insert_opportunity(opp: dict) -> bool
   - update_position(id, updates: dict) -> bool
   - insert_trade(trade: dict) -> bool
   - insert_event(event: dict) -> bool
   - insert_equity_snapshot(snapshot: dict) -> bool
   - purge_old_scores(days=180) -> int (rows deleted)
   Use supabase-py client with STOCKS_SUPABASE_URL and STOCKS_SUPABASE_KEY.
   All operations on the 'stocks' schema.
   Batch inserts where possible.
   Every function: try/except, log errors, return success boolean.

8. utils/telegram_bot.py
   - init_bot() -> Bot instance
   - send_message(text: str) -> bool
   - send_alert(alert_type: str, data: dict) -> bool
   - format_opportunity(opp: dict) -> str
   - format_position_update(pos: dict) -> str
   - format_weekly_summary(opps: List[dict], market_mood: str) -> str
   - format_morning_briefing(events: List[dict], positions: List[dict]) -> str
   Use python-telegram-bot library.
   Use emoji conventions from DEVELOPMENT_PLAN.md.
   Split messages > 4096 chars.
   Handle send failures gracefully (retry 2x, then log).

9. utils/scheduler.py
   - is_market_open() -> bool (checks current time in ET)
   - is_trading_day() -> bool (Mon-Fri, not US holidays)
   - get_next_run_time(schedule_type) -> datetime
   - MarketScheduler class:
     - schedule_weekly_scan(callback) — Sunday 8 PM ET
     - schedule_morning_briefing(callback) — Mon-Fri 8:30 AM ET
     - schedule_intraday_monitor(callback) — Every 15 min during market hours
     - schedule_after_market(callback) — Mon-Fri 4:30 PM ET
     - start() / stop()
   Use APScheduler with timezone='America/New_York'.
   Handle DST transitions correctly.
   Log every scheduled execution.

10. health.py
    FastAPI app with:
    - GET /health -> {"status": "ok", "uptime": seconds, "memory_mb": current_rss}
    - GET /status -> {"positions": count, "last_scan": timestamp, "next_run": timestamp}
    Run on port 8001 (crypto bot uses 8000).

11. main.py
    - Load .env
    - Configure logging (format from DEVELOPMENT_PLAN.md)
    - Initialize Supabase, Telegram, Scheduler
    - Start health server in background thread
    - Start scheduler
    - Log startup message
    - Handle SIGTERM/SIGINT gracefully
    - On startup: send Telegram "🟢 Stocks bot started"
    - On shutdown: send Telegram "🔴 Stocks bot stopped"

12. Dockerfile
    - FROM python:3.11-slim
    - Multi-stage build
    - Non-root user
    - Copy requirements first (cache layer)
    - Expose 8001
    - CMD: python main.py

13. .github/workflows/deploy.yml
    - Trigger: push to main branch
    - Steps: checkout, SSH to DigitalOcean, git pull, pip install, restart service
    - Use GitHub secrets for SSH key and host

14. scripts/setup_digitalocean.sh
    - Create /home/algotrading/stocks-agent directory
    - Create Python venv
    - Create systemd service file (algotrading-stocks.service)
    - Enable and start service
    - Configure log rotation

CODING STANDARDS:
- Follow ALL conventions from DEVELOPMENT_PLAN.md Deliverable 9
- Type hints on every function
- Docstrings on every function
- Logger per module: logger = logging.getLogger(__name__)
- Error handling: specific exceptions, log + fallback, never crash
- Environment variables: STOCKS_ prefix
- No hardcoded credentials anywhere

After creating all files, verify:
- No import errors (python -c "import main")
- Health endpoint works
- Telegram test message sends
- All indicators compute for sample AAPL data
```

---

## FLAGGED ISSUES & ASSUMPTIONS

### ⚠️ Issues Found in Requirements

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 1 | **Industry rank within sector** requires granular industry classification data. Free sources (yfinance) have inconsistent industry labels across the S&P 500. | Medium | Defer to V2. Use sector rank only in V1 |
| 2 | **Days to cover (short squeeze)** requires short interest data. yfinance's `shortPercentOfFloat` is often stale or missing. | Medium | Defer to V2. Include as "unknown" when unavailable |
| 3 | **Alpha Vantage 25 req/day** is extremely tight. Cannot be used for routine data — only as emergency backup. The requirement lists it as a data source, but it's effectively unusable for scanning. | Low | Use exclusively as yfinance fallback. Never for routine scanning |
| 4 | **Supabase `stocks` schema "already created"** — need to verify it's truly empty and the `stocks` schema exists. If not, the migration SQL needs a `CREATE SCHEMA IF NOT EXISTS stocks;` prepended. | Low | Add `CREATE SCHEMA IF NOT EXISTS stocks;` to migration |
| 5 | **Same Telegram bot for crypto + stocks** — if using the same bot token, need to differentiate commands/channels. Recommend using same bot but different chat groups or thread topics. | Low | Use same bot, send stocks alerts to a different Telegram group/topic. Use `STOCKS_TELEGRAM_CHAT_ID` separate from crypto's chat ID |
| 6 | **`render.yaml` listed as "legacy"** in file structure but project targets DigitalOcean deployment. | Low | Create as empty placeholder with comment. Don't invest time |
| 7 | **FinBERT listed in schedule** (Sunday local Mac) but listed as "WON'T HAVE" in V1. | Low | Remove from V1 schedule. Add back in V2 |
| 8 | **Prophet forecasting** listed in ML models but not in the 4 main models. It's mentioned in the Sunday local Mac schedule. | Low | Include as optional Phase 3 stretch goal. Not a blocker |
| 9 | **Pocket pivot detection** is listed in Layer 3 but moved to WON'T HAVE. | Low | Consistent — it's acknowledged as low-frequency, hard to validate |
| 10 | **OpenAI cost estimate ($5/month)** is extremely conservative. Actual usage as planned will be ~$0.06-0.60/month. The budget has 8-80x headroom. | Info | Good problem to have. Consider using GPT-4o more liberally for better alert quality |

### Assumptions Made

| # | Assumption | Impact if Wrong |
|---|-----------|----------------|
| 1 | Crypto bot uses port 8000 for health checks | If different, update stocks bot port accordingly |
| 2 | Supabase free tier 500MB is for database only (storage bucket is separate) | If shared, ML model storage could be an issue — models are ~5-10 MB each |
| 3 | DigitalOcean droplet has Python 3.11+ available | If not, Dockerfile handles this. For systemd, may need to install Python first |
| 4 | Crypto bot and stocks bot have separate Telegram chat IDs | If same chat, messages will interleave. Add "[STOCKS]" prefix to all messages |
| 5 | yfinance will continue working without authentication | If Yahoo changes API, entire data pipeline breaks. Alpha Vantage is thin backup |
| 6 | User's local Mac has Python 3.11+, pip, and sufficient disk for ML training | Training requires ~2GB disk for datasets + models |
| 7 | GitHub Actions has access to DigitalOcean via SSH | User must add SSH private key as GitHub secret `DO_SSH_KEY` |
| 8 | `stocks` schema in Supabase is completely empty | If tables exist, migration will fail. Add `IF NOT EXISTS` to all `CREATE TABLE` |
| 9 | The crypto bot's systemd service is named `algotrading-crypto.service` | Need to verify actual service name to avoid conflicts |
| 10 | US market holidays follow standard NYSE schedule | Using `exchange_calendars` or hardcoded list for 2026 |

---

## APPENDIX A — COMPLETE REQUIREMENTS.TXT

```
# Core
fastapi==0.109.0
uvicorn==0.27.0
python-dotenv==1.0.1

# Data
yfinance==0.2.36
pandas==2.2.0
numpy==1.26.3

# Technical Analysis
ta==0.11.0

# Machine Learning
xgboost==2.0.3
scikit-learn==1.4.0

# Database
supabase==2.3.4

# Telegram
python-telegram-bot==20.7

# AI/NLP
openai==1.12.0
vaderSentiment==3.3.2

# News
newsapi-python==0.2.7

# Scheduling
apscheduler==3.10.4
pytz==2024.1

# HTTP
httpx==0.27.0

# Monitoring
psutil==5.9.8

# Utilities
beautifulsoup4==4.12.3
lxml==5.1.0
```

> **Note:** Pin versions to what's current as of March 2026. The versions above are estimates — verify and update at build time.

---

## APPENDIX B — SYSTEMD SERVICE FILE

```ini
[Unit]
Description=Stocks Trading Advisory Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=algotrading
Group=algotrading
WorkingDirectory=/home/algotrading/stocks-agent
Environment=PATH=/home/algotrading/stocks-agent/venv/bin:/usr/bin
EnvironmentFile=/home/algotrading/stocks-agent/.env
ExecStart=/home/algotrading/stocks-agent/venv/bin/python main.py
Restart=on-failure
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=300

# Memory limit (hard cap)
MemoryMax=450M
MemoryHigh=400M

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=stocks-agent

[Install]
WantedBy=multi-user.target
```

---

## APPENDIX C — WEEKLY EXECUTION TIMELINE

```
SUNDAY
  20:00 ET ─── Weekly Scan Start
  20:01      ├── Fetch S&P 500 list (yfinance)
  20:03      ├── Download 1Y price data for 500 stocks
  20:08      ├── Calculate all indicators (500 stocks)
  20:12      ├── Calculate RS scores (500 stocks)
  20:14      ├── Run sector rotation model
  20:15      ├── Filter to top 50 candidates
  20:16      ├── Detect patterns (ML)
  20:18      ├── Score all 50 candidates
  20:19      ├── Select top 5 opportunities
  20:20      ├── Calculate position sizes
  20:22      ├── GPT-4o weekly briefing
  20:24      ├── Send Telegram weekly summary
  20:25      └── Save to Supabase
  20:25 ─── Weekly Scan Complete (~25 min)

MONDAY-FRIDAY
  06:00 ET ─── Check for new ML models in Supabase Storage
  08:30 ET ─── Morning Briefing
  08:30      ├── Fetch overnight news (NewsAPI)
  08:31      ├── VADER sentiment scoring
  08:32      ├── Check macro events
  08:33      ├── Check earnings calendar
  08:34      ├── Send morning Telegram
  08:35 ─── Morning Briefing Complete (~5 min)

  09:30 ET ─── Market Opens
  09:30-16:00  Every 15 minutes:
               ├── Fetch position prices
               ├── Check stop proximity
               ├── Check target proximity
               ├── Alert if action needed
               └── (~30 sec per cycle)

  16:00 ET ─── Market Closes
  16:30 ET ─── After-Market Review
  16:30      ├── Update daily scores
  16:32      ├── Check insider filings
  16:35      ├── GPT-4o mini exit analysis
  16:37      ├── Send EOD summary
  16:38 ─── After-Market Complete (~8 min)

  16:38-08:29  Sleep (no activity)
```

---

*End of Development Plan — Version 1.0*
*Ready to proceed with Phase 1 building.*
