-- ============================================================
-- Migration 002: Portfolio Manager Tables
-- Run in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── portfolio_holdings ────────────────────────────────────────
-- The live model portfolio. Always reflects the current
-- intended state — at most MAX_SLOTS rows in 'active' status.
CREATE TABLE IF NOT EXISTS stocks.portfolio_holdings (
    id                       UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker                   VARCHAR(10) NOT NULL UNIQUE,
    status                   VARCHAR(20) NOT NULL DEFAULT 'active',
        -- active   : in portfolio, signal strong
        -- watch    : in portfolio, signal weakening (1 weak week)
        -- exiting  : in portfolio, flagged for removal (2 weak weeks)
    entry_confidence         NUMERIC(5,1),   -- confidence when first added
    current_confidence       NUMERIC(5,1),   -- updated each weekly scan
    prev_confidence          NUMERIC(5,1),   -- last week's confidence
    consecutive_strong_weeks INTEGER DEFAULT 0,
        -- weeks ≥ ENTRY_THRESHOLD before entering portfolio
    consecutive_weak_weeks   INTEGER DEFAULT 0,
        -- weeks < STAY_THRESHOLD — triggers exit at 2
    weeks_held               INTEGER DEFAULT 0,
        -- total weeks in portfolio
    added_at                 TIMESTAMPTZ DEFAULT NOW(),
    last_scored_at           TIMESTAMPTZ,
    sector                   VARCHAR(100),
    entry_price              NUMERIC(10,2),
    stop_loss                NUMERIC(10,2),
    target_price             NUMERIC(10,2),
    sub_scores               JSONB,
    setup_type               VARCHAR(50),
    gpt_risk_flag            BOOLEAN DEFAULT FALSE,
    notes                    TEXT,
    updated_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_status
    ON stocks.portfolio_holdings(status);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_ticker
    ON stocks.portfolio_holdings(ticker);


-- ── signal_history ────────────────────────────────────────────
-- Weekly confidence score per ticker across all scans.
-- Used to check signal continuity (2-week confirmation rule).
CREATE TABLE IF NOT EXISTS stocks.signal_history (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker        VARCHAR(10) NOT NULL,
    scan_date     DATE NOT NULL,
    confidence    NUMERIC(5,1),
    sub_scores    JSONB,
    in_portfolio  BOOLEAN DEFAULT FALSE,
    setup_type    VARCHAR(50),
    passed_scan   BOOLEAN DEFAULT TRUE,  -- passed hard filters?
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, scan_date)
);

CREATE INDEX IF NOT EXISTS idx_signal_history_ticker_date
    ON stocks.signal_history(ticker, scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_history_scan_date
    ON stocks.signal_history(scan_date DESC);


-- ── portfolio_log ─────────────────────────────────────────────
-- Audit trail of every portfolio state change.
CREATE TABLE IF NOT EXISTS stocks.portfolio_log (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker      VARCHAR(10) NOT NULL,
    action      VARCHAR(30) NOT NULL,
        -- ADDED | REMOVED | WATCH_FLAG | WATCH_CLEARED
        -- HOLD_CONFIRMED | DISPLACEMENT_CANDIDATE | GPT_VETO
    reason      TEXT,
    confidence  NUMERIC(5,1),
    prev_status VARCHAR(20),
    new_status  VARCHAR(20),
    scan_date   DATE DEFAULT CURRENT_DATE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_log_ticker
    ON stocks.portfolio_log(ticker, scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_log_date
    ON stocks.portfolio_log(scan_date DESC);


-- ── Permissions ───────────────────────────────────────────────
GRANT ALL ON stocks.portfolio_holdings TO service_role;
GRANT ALL ON stocks.signal_history TO service_role;
GRANT ALL ON stocks.portfolio_log TO service_role;
