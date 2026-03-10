-- ============================================================
-- stocks-agent Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- Create dedicated schema
CREATE SCHEMA IF NOT EXISTS stocks;

-- Universe (S&P 500 constituents)
CREATE TABLE IF NOT EXISTS stocks.universe (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap_b NUMERIC(10,2),
    avg_volume_50d BIGINT,
    in_sp500 BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);



-- Opportunities (weekly top picks)
CREATE TABLE IF NOT EXISTS stocks.opportunities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    score_date DATE,
    confidence NUMERIC(5,1),
    setup_type VARCHAR(50),
    entry_price_low NUMERIC(10,2),
    entry_price_high NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    target_price NUMERIC(10,2),
    position_size_usd NUMERIC(12,2),
    shares INTEGER,
    risk_usd NUMERIC(10,2),
    reward_usd NUMERIC(10,2),
    risk_reward_ratio NUMERIC(5,2),
    reasons TEXT[],
    sub_scores JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    expired_at TIMESTAMPTZ
);

-- Positions (open advisory trades)
CREATE TABLE IF NOT EXISTS stocks.positions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    entry_date DATE,
    closed_at TIMESTAMPTZ,
    entry_price NUMERIC(10,2),
    current_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    target_price NUMERIC(10,2),
    shares INTEGER,
    position_size_usd NUMERIC(12,2),
    unrealized_pnl NUMERIC(10,2),
    unrealized_pnl_pct NUMERIC(6,2),
    realized_pnl NUMERIC(10,2),
    status VARCHAR(20) DEFAULT 'open',
    days_held INTEGER DEFAULT 0,
    exit_reason VARCHAR(50),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades (closed trade history)
CREATE TABLE IF NOT EXISTS stocks.trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    entry_date DATE,
    exit_date DATE,
    entry_price NUMERIC(10,2),
    exit_price NUMERIC(10,2),
    shares INTEGER,
    realized_pnl NUMERIC(10,2),
    pnl_pct NUMERIC(6,2),
    exit_reason VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market Events
CREATE TABLE IF NOT EXISTS stocks.market_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_detail TEXT,
    event_date DATE DEFAULT CURRENT_DATE,
    severity VARCHAR(10) DEFAULT 'low',
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB
);

-- Equity Snapshots
CREATE TABLE IF NOT EXISTS stocks.equity_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    total_value NUMERIC(12,2),
    cash NUMERIC(12,2),
    invested NUMERIC(12,2),
    unrealized_pnl NUMERIC(10,2),
    realized_pnl NUMERIC(10,2),
    position_count INTEGER
);

-- Model Versions
CREATE TABLE IF NOT EXISTS stocks.model_versions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    file_path TEXT,
    metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE
);

-- Bot Status
CREATE TABLE IF NOT EXISTS stocks.bot_status (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Permissions
GRANT USAGE ON SCHEMA stocks TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA stocks TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA stocks TO service_role;
