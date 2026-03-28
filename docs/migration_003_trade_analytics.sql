-- ============================================================
-- Migration 003: Trade Analytics + Missing Columns
-- Run in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── 1. POSITIONS — columns the code writes but schema lacks ──────────
ALTER TABLE stocks.positions
    ADD COLUMN IF NOT EXISTS trailing_stop    NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS atr_value        NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS exit_price       NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS exit_date        DATE,
    ADD COLUMN IF NOT EXISTS high_water_mark  NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS low_water_mark   NUMERIC(10,2);

-- ── 2. TRADES — analytics columns ───────────────────────────────────
ALTER TABLE stocks.trades
    ADD COLUMN IF NOT EXISTS hold_days          INTEGER,
    ADD COLUMN IF NOT EXISTS realized_pnl_pct   NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS setup_type         VARCHAR(50),
    ADD COLUMN IF NOT EXISTS entry_confidence   NUMERIC(5,1),
    ADD COLUMN IF NOT EXISTS mae_pct            NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS mfe_pct            NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS sub_scores_at_entry JSONB;

-- ── 3. EQUITY_SNAPSHOTS — columns the code writes ───────────────────
ALTER TABLE stocks.equity_snapshots
    ADD COLUMN IF NOT EXISTS portfolio_value  NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS open_positions   INTEGER,
    ADD COLUMN IF NOT EXISTS daily_pnl        NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS total_pnl        NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS win_rate         NUMERIC(5,1);

-- ── 4. MARKET_EVENTS — columns the code writes ─────────────────────
ALTER TABLE stocks.market_events
    ADD COLUMN IF NOT EXISTS description      TEXT,
    ADD COLUMN IF NOT EXISTS vix_level        NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS spy_change_pct   NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS regime           VARCHAR(20);

-- ── 5. UNIVERSE — code writes market_cap, schema has market_cap_b ───
ALTER TABLE stocks.universe
    ADD COLUMN IF NOT EXISTS market_cap       NUMERIC(12,2);

-- ── 6. GPT_BRIEFINGS — may not exist yet ────────────────────────────
CREATE TABLE IF NOT EXISTS stocks.gpt_briefings (
    id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content      TEXT,
    market_mood  VARCHAR(20),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ── 7. TRADE_POSTMORTEMS — closed-loop learning (Layer 4) ──────────
CREATE TABLE IF NOT EXISTS stocks.trade_postmortems (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker              VARCHAR(10) NOT NULL,
    trade_id            UUID,
    outcome             VARCHAR(30) NOT NULL,
        -- TRUE_POSITIVE | FALSE_POSITIVE | REGIME_MISMATCH
    entry_date          DATE,
    exit_date           DATE,
    entry_confidence    NUMERIC(5,1),
    exit_confidence     NUMERIC(5,1),
    sub_scores_at_entry JSONB,
    sub_scores_at_exit  JSONB,
    regime_at_entry     VARCHAR(30),
    regime_at_exit      VARCHAR(30),
    pnl                 NUMERIC(10,2),
    pnl_pct             NUMERIC(6,2),
    mae_pct             NUMERIC(6,2),
    mfe_pct             NUMERIC(6,2),
    hold_days           INTEGER,
    setup_type          VARCHAR(50),
    signal_source       VARCHAR(50),
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_postmortems_ticker
    ON stocks.trade_postmortems(ticker);
CREATE INDEX IF NOT EXISTS idx_trade_postmortems_outcome
    ON stocks.trade_postmortems(outcome);

-- ── 8. REGIME_SNAPSHOTS — regime history (Layer 3) ──────────────────
CREATE TABLE IF NOT EXISTS stocks.regime_snapshots (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    snapshot_date       DATE NOT NULL,
    regime              VARCHAR(30) NOT NULL,
    components          JSONB,
        -- {rsp_spy, yield_curve, credit, size_factor, equity_bond, sector_rotation}
    position_size_mod   NUMERIC(4,2),
    breadth_score       NUMERIC(5,1),
    top_risk_score      NUMERIC(5,1),
    vix_level           NUMERIC(6,2),
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_snapshots_date
    ON stocks.regime_snapshots(snapshot_date DESC);

-- ── 9. CALIBRATED_WEIGHTS — auto-tuned scorer weights (Layer 4) ─────
CREATE TABLE IF NOT EXISTS stocks.calibrated_weights (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    weights         JSONB NOT NULL,
        -- {technical, rs, fundamental, sentiment, insider, macro, ml, canslim}
    sample_size     INTEGER NOT NULL,
    win_rate        NUMERIC(5,1),
    avg_pnl         NUMERIC(10,2),
    calibration_date DATE NOT NULL,
    notes           TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── 10. Permissions ─────────────────────────────────────────────────
GRANT ALL ON stocks.gpt_briefings TO service_role;
GRANT ALL ON stocks.trade_postmortems TO service_role;
GRANT ALL ON stocks.regime_snapshots TO service_role;
GRANT ALL ON stocks.calibrated_weights TO service_role;

-- ── Done. Verify ────────────────────────────────────────────────────
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'stocks'
ORDER BY table_name, ordinal_position;
