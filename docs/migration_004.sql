-- migration_004: columns needed by Layer 3/4 features
-- Apply via Supabase dashboard → SQL Editor, or psql

-- 1. positions: add sector (needed for exposure summary)
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS sector TEXT;

CREATE INDEX IF NOT EXISTS idx_positions_sector ON positions (sector);

-- 2. portfolio_holdings: add regime_at_entry (needed for postmortem REGIME_MISMATCH)
ALTER TABLE portfolio_holdings
    ADD COLUMN IF NOT EXISTS regime_at_entry TEXT DEFAULT 'unknown';

-- 3. opportunities: prevent duplicate scan entries per ticker per scan date
--    (scanner can upsert without creating ghost rows)
ALTER TABLE opportunities
    ADD COLUMN IF NOT EXISTS scan_date DATE;

CREATE UNIQUE INDEX IF NOT EXISTS uq_opportunities_ticker_scan
    ON opportunities (ticker, scan_date)
    WHERE scan_date IS NOT NULL;

-- 4. trade_postmortems: ensure table exists with all required columns
CREATE TABLE IF NOT EXISTS trade_postmortems (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    outcome         TEXT        NOT NULL,  -- TRUE_POSITIVE | FALSE_POSITIVE | SCRATCH | REGIME_MISMATCH
    entry_date      DATE,
    exit_date       DATE,
    entry_confidence NUMERIC,
    sub_scores_at_entry JSONB,
    regime_at_entry TEXT,
    regime_at_exit  TEXT,
    pnl             NUMERIC,
    pnl_pct         NUMERIC,
    mae_pct         NUMERIC,
    mfe_pct         NUMERIC,
    hold_days       INTEGER,
    setup_type      TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_postmortems_ticker  ON trade_postmortems (ticker);
CREATE INDEX IF NOT EXISTS idx_postmortems_outcome ON trade_postmortems (outcome);
CREATE INDEX IF NOT EXISTS idx_postmortems_created ON trade_postmortems (created_at DESC);

-- 5. calibrated_weights: used by weight_calibrator.py
CREATE TABLE IF NOT EXISTS calibrated_weights (
    id          BIGSERIAL PRIMARY KEY,
    weights     JSONB       NOT NULL,
    sample_size INTEGER     NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active      BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_calibrated_weights_active ON calibrated_weights (active, computed_at DESC);

-- 6. regime_snapshots: weekly regime state log
CREATE TABLE IF NOT EXISTS regime_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    scan_date       DATE        NOT NULL UNIQUE,
    regime          TEXT        NOT NULL,
    bullish_score   NUMERIC,
    position_size_modifier NUMERIC,
    components      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
