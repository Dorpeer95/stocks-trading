# Setup & Deployment Guide

> Last updated: 2026-03-01

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Environment Variables](#environment-variables)
4. [Supabase Setup](#supabase-setup)
5. [Telegram Bot Setup](#telegram-bot-setup)
6. [API Keys](#api-keys)
7. [DigitalOcean Deployment](#digitalocean-deployment)
8. [First Run Checklist](#first-run-checklist)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement     | Version   | Notes                                 |
|-----------------|-----------|---------------------------------------|
| Python          | 3.11+     | 3.11 recommended for production       |
| Git             | 2.x       |                                       |
| pip             | 23+       |                                       |

Optional (for deployment):
- DigitalOcean droplet (Ubuntu 22.04+, 1 GB RAM)
- Docker (for containerised deployment)

---

## Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Dorpeer95/stocks-trading.git
cd stocks-trading

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (see section below)

# 5. Create logs directory
mkdir -p logs

# 6. Run in dry-run mode (no real trades/alerts)
STOCKS_DRY_RUN=true python main.py
```

### Running Components Individually

```bash
# Health server only
python -c "from health import run_health_server; run_health_server()"

# Test a single scan (requires all API keys)
python -c "
from agent.agent import AgentLoop
loop = AgentLoop()
loop.weekly_scan()
"

# Train ML models (requires historical data)
python scripts/train_model.py --model direction --output models/
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in each value:

### Required

| Variable                  | Description                          | Where to get it                  |
|---------------------------|--------------------------------------|----------------------------------|
| `STOCKS_SUPABASE_URL`     | Supabase project URL                 | Supabase dashboard → Settings    |
| `STOCKS_SUPABASE_KEY`     | Supabase service role key            | Supabase dashboard → API         |
| `STOCKS_TELEGRAM_TOKEN`   | Telegram bot API token               | @BotFather on Telegram           |
| `STOCKS_TELEGRAM_CHAT_ID` | Telegram chat/group ID               | See Telegram setup below         |

### Optional (recommended)

| Variable                  | Default  | Description                     |
|---------------------------|----------|---------------------------------|
| `STOCKS_OPENAI_API_KEY`   | —        | OpenAI API key for GPT features |
| `STOCKS_NEWSAPI_KEY`      | —        | NewsAPI key for news sentiment  |
| `STOCKS_ALPHA_VANTAGE_KEY`| —        | Alpha Vantage backup data       |

### Configuration

| Variable                   | Default      | Description                       |
|----------------------------|--------------|-----------------------------------|
| `STOCKS_PORTFOLIO_VALUE`   | `10000`      | Paper portfolio size (USD)        |
| `STOCKS_MAX_RISK_PCT`      | `0.02`       | Max risk per trade (2%)           |
| `STOCKS_MAX_POSITION_PCT`  | `0.25`       | Max single position (25%)         |
| `STOCKS_CASH_RESERVE_PCT`  | `0.10`       | Minimum cash reserve (10%)        |

### Feature Flags

| Variable              | Default | Description                              |
|-----------------------|---------|------------------------------------------|
| `STOCKS_ENABLE_ML`    | `true`  | Enable XGBoost ML scoring layer          |
| `STOCKS_ENABLE_GPT`   | `true`  | Enable GPT-4o sentiment & briefings      |
| `STOCKS_ENABLE_NEWS`  | `true`  | Enable NewsAPI news fetching             |
| `STOCKS_DRY_RUN`      | `false` | Disable persistence & real alerts        |
| `STOCKS_LOG_LEVEL`    | `INFO`  | Logging level (DEBUG, INFO, WARNING)     |
| `STOCKS_HEALTH_PORT`  | `8001`  | Health check server port                 |
| `STOCKS_ENV`          | `development` | Environment name                    |

---

## Supabase Setup

### 1. Create a Project

1. Go to [supabase.com](https://supabase.com) and create a free project
2. Note the project URL and service role key from Settings → API

### 2. Create the Schema

Run the following SQL in the Supabase SQL editor:

```sql
-- Create dedicated schema
CREATE SCHEMA IF NOT EXISTS stocks;

-- Opportunities
CREATE TABLE stocks.opportunities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    confidence NUMERIC(5,1),
    setup_type VARCHAR(50),
    entry_price_low NUMERIC(10,2),
    entry_price_high NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    target_price NUMERIC(10,2),
    position_size_usd NUMERIC(12,2),
    risk_usd NUMERIC(10,2),
    reward_usd NUMERIC(10,2),
    risk_reward_ratio NUMERIC(5,2),
    reasons TEXT[],
    sub_scores JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    expired_at TIMESTAMPTZ
);

-- Positions
CREATE TABLE stocks.positions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    opened_at TIMESTAMPTZ DEFAULT NOW(),
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
    exit_reason VARCHAR(50)
);

-- Daily Scores
CREATE TABLE stocks.daily_scores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    score_date DATE NOT NULL,
    total_score NUMERIC(5,1),
    technical_score NUMERIC(5,1),
    rs_score NUMERIC(5,1),
    fundamental_score NUMERIC(5,1),
    sentiment_score NUMERIC(5,1),
    rsi_14 NUMERIC(5,1),
    adx NUMERIC(5,1),
    macd_signal VARCHAR(10),
    atr_pct NUMERIC(5,2),
    rs_percentile NUMERIC(5,1),
    volume_ratio NUMERIC(5,2),
    UNIQUE(ticker, score_date)
);

-- Events
CREATE TABLE stocks.events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_detail TEXT,
    severity VARCHAR(10) DEFAULT 'low',
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB
);

-- Equity Snapshots
CREATE TABLE stocks.equity_snapshots (
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
CREATE TABLE stocks.model_versions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    file_path TEXT,
    metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE
);

-- Bot Status
CREATE TABLE stocks.bot_status (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Watchlist
CREATE TABLE stocks.watchlist (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- Create Storage bucket for ML models
-- (Do this via Supabase dashboard → Storage → New bucket → "models")
```

### 3. Create Storage Bucket

1. Go to Supabase dashboard → Storage
2. Create a new bucket named `models`
3. Set it to **private** (service role key has full access)

---

## Telegram Bot Setup

### 1. Create the Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Follow the prompts to name your bot (e.g., "Stocks Agent")
4. Copy the API token → `STOCKS_TELEGRAM_TOKEN`

### 2. Get Your Chat ID

1. Create a group or use a direct chat with your bot
2. Send a message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Find `"chat":{"id": 123456789}` in the response
5. Copy the chat ID → `STOCKS_TELEGRAM_CHAT_ID`

### 3. Test

```bash
python -c "
from utils.telegram_bot import init_bot, send_message
init_bot()
send_message('🟢 stocks-agent test message')
print('Check Telegram!')
"
```

---

## API Keys

### OpenAI (optional but recommended)

1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an API key
3. Set a monthly spending limit of $5 (the bot also enforces this internally)
4. Add to `.env` as `STOCKS_OPENAI_API_KEY`

### NewsAPI (optional but recommended)

1. Go to [newsapi.org](https://newsapi.org)
2. Register for a free plan (100 requests/day)
3. Copy the API key → `STOCKS_NEWSAPI_KEY`

### Alpha Vantage (optional)

1. Go to [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Get a free key (25 requests/day)
3. Copy → `STOCKS_ALPHA_VANTAGE_KEY`

---

## DigitalOcean Deployment

### Prerequisites

- DigitalOcean droplet with Ubuntu 22.04+
- `algotrading` user exists (shared with crypto bot)
- Python 3.11+ installed
- Git installed

### Automated Setup

```bash
# SSH into your droplet
ssh root@your-droplet-ip

# Run the setup script
curl -sL https://raw.githubusercontent.com/Dorpeer95/stocks-trading/main/scripts/setup_digitalocean.sh | bash
```

Or manually:

```bash
# Clone on the server
sudo -u algotrading git clone https://github.com/Dorpeer95/stocks-trading.git /home/algotrading/stocks-agent
cd /home/algotrading/stocks-agent

# Run setup
sudo bash scripts/setup_digitalocean.sh
```

### Post-Setup

1. **Edit environment**: `sudo -u algotrading nano /home/algotrading/stocks-agent/.env`
2. **Restart service**: `sudo systemctl restart algotrading-stocks`
3. **Check health**: `curl http://localhost:8001/health`

### CI/CD (GitHub Actions)

Pushes to `main` automatically deploy. Set these repository secrets:

| Secret          | Value                        |
|-----------------|------------------------------|
| `DO_HOST`       | Your droplet IP              |
| `DO_USERNAME`   | `algotrading`                |
| `DO_SSH_KEY`    | Private SSH key for droplet  |

### Useful Commands on Server

```bash
# Service management
sudo systemctl status algotrading-stocks
sudo systemctl restart algotrading-stocks
sudo journalctl -u algotrading-stocks -f

# Health check
curl http://localhost:8001/health | python3 -m json.tool
curl http://localhost:8001/status | python3 -m json.tool

# Memory check (shared with crypto bot)
ps aux | grep -E "stocks|crypto" | grep -v grep

# Using helper scripts
bash scripts/bot_status.sh
bash scripts/bot_logs.sh 100
bash scripts/bot_restart.sh
```

---

## First Run Checklist

- [ ] `.env` file created with all required keys
- [ ] Supabase schema created (all 8 tables)
- [ ] Supabase Storage bucket `models` created
- [ ] Telegram bot created and chat ID obtained
- [ ] Test Telegram message received
- [ ] `python main.py` starts without errors
- [ ] Health endpoint responds: `curl http://localhost:8001/health`
- [ ] First weekly scan runs (check logs or trigger manually)
- [ ] Telegram alerts arrive with proper formatting
- [ ] Dry-run mode tested: `STOCKS_DRY_RUN=true python main.py`

---

## Troubleshooting

### Bot won't start

```bash
# Check for missing env vars
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('STOCKS_SUPABASE_URL'))"

# Check for import errors
python -c "from agent.agent import AgentLoop; print('OK')"
```

### Telegram messages not arriving

- Verify `STOCKS_TELEGRAM_TOKEN` and `STOCKS_TELEGRAM_CHAT_ID`
- Make sure you've sent a message to the bot first
- Check if bot is blocked or removed from group

### Supabase connection fails

- Verify `STOCKS_SUPABASE_URL` (should be `https://xxxxx.supabase.co`)
- Use service role key, not anon key
- Check if project is paused (free tier pauses after inactivity)

### High memory usage

- Check with `ps aux | grep stocks`
- Reduce batch size: scan fewer stocks per cycle
- Disable ML models: `STOCKS_ENABLE_ML=false`
- Clear data cache between scans (automatic in weekly scan)

### yfinance rate limiting

- The bot uses exponential backoff with 3 retries
- If persistent, reduce scan frequency or universe size
- Check if yfinance API has changed (update `yfinance` package)

### OpenAI costs unexpected

- Check CostTracker: the bot logs every API call cost
- Hard cap at $5/month is enforced in code
- Disable GPT: `STOCKS_ENABLE_GPT=false`
- Disable news: `STOCKS_ENABLE_NEWS=false`
