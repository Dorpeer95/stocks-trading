-- ============================================================
-- Migration 001: Fix opportunities table (v2 — safe, no assumptions)
-- Run in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── Step 1: Add ALL columns the code expects ─────────────────────────────
-- IF NOT EXISTS means this is safe to re-run at any time.

ALTER TABLE stocks.opportunities
    ADD COLUMN IF NOT EXISTS acted_on          BOOLEAN       DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS scan_date         DATE,
    ADD COLUMN IF NOT EXISTS entry_price       NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS notes             TEXT,
    ADD COLUMN IF NOT EXISTS atr               NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS risk_reward       NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS risk_reward_ratio NUMERIC(5,2);

-- ── Step 2: Clean stale data ──────────────────────────────────────────────
-- Delete every opportunity older than 30 days — they are expired setups.
DELETE FROM stocks.opportunities
WHERE created_at < NOW() - INTERVAL '30 days';

-- ── Step 3: Mark remaining old rows as acted_on so they don't
--    appear in your watchlist. Fresh rows from the next scan will
--    be acted_on = FALSE (the default).
UPDATE stocks.opportunities
    SET acted_on = TRUE
    WHERE acted_on IS NULL OR acted_on = FALSE;

-- ── Step 4: Backfill scan_date from score_date where available
UPDATE stocks.opportunities
    SET scan_date = score_date
    WHERE scan_date IS NULL AND score_date IS NOT NULL;

-- ── Step 5: Backfill entry_price as average of low/high
UPDATE stocks.opportunities
    SET entry_price = ROUND((entry_price_low + entry_price_high) / 2.0, 2)
    WHERE entry_price IS NULL
      AND entry_price_low IS NOT NULL
      AND entry_price_high IS NOT NULL;

-- ── Done. Check the result ────────────────────────────────────────────────
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'stocks'
  AND table_name   = 'opportunities'
ORDER BY ordinal_position;
