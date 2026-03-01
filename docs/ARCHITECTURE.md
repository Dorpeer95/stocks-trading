# Architecture — stocks-agent

> Last updated: 2026-03-01

## Overview

**stocks-agent** is a Python-based stock market advisory bot that scans the
S&P 500 universe weekly, scores candidates using technical indicators,
relative strength, fundamentals, ML models, and news sentiment, then delivers
actionable trade ideas via Telegram.

It is **advisory only** — the bot suggests trades; the user executes them.

```
┌─────────────────────────────────────────────────────────────────┐
│                        stocks-agent                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │  main.py  │──▶│ Scheduler│──▶│ AgentLoop│──▶│ Telegram Bot│ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘ │
│       │                              │                          │
│       ▼                              ▼                          │
│  ┌──────────┐        ┌───────────────────────────┐             │
│  │health.py │        │        Agent Layer         │             │
│  │ :8001    │        │  scanner → scorer → opps   │             │
│  └──────────┘        │  events → portfolio        │             │
│                      │  ai_model → feature_config │             │
│                      └───────────────────────────┘             │
│                              │                                  │
│                              ▼                                  │
│                 ┌─────────────────────────┐                     │
│                 │       Utils Layer        │                     │
│                 │  data_loader  indicators │                     │
│                 │  sentiment   sectors     │                     │
│                 │  earnings    insider     │                     │
│                 │  position_sizing         │                     │
│                 └─────────────────────────┘                     │
│                              │                                  │
│              ┌───────────────┼──────────────┐                   │
│              ▼               ▼              ▼                   │
│        ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│        │ yfinance │   │ Supabase │   │  OpenAI  │             │
│        │ NewsAPI  │   │ (Postgres)│  │ VADER    │             │
│        └──────────┘   └──────────┘   └──────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
stocks-trading/
├── main.py                        # Entry point, signal handlers, startup/shutdown
├── health.py                      # FastAPI health server on :8001
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Multi-stage production image
├── .env.example                   # Environment variable template
├── .gitignore
│
├── agent/                         # Core intelligence layer
│   ├── __init__.py
│   ├── agent.py                   # AgentLoop — orchestrates all scheduled tasks
│   ├── scanner.py                 # Universe scan, filters (ADX, RS, volume, etc.)
│   ├── scorer.py                  # Composite scoring (technical, RS, ML, sentiment)
│   ├── events.py                  # Macro event detection, market regime assessment
│   ├── portfolio.py               # Position tracking, EOD summaries, equity snapshots
│   ├── persistence.py             # Supabase CRUD for all 8 tables
│   ├── ai_model.py                # XGBoost ModelManager (load, predict, batch)
│   └── feature_config.py          # ML feature definitions & vector builders
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py             # yfinance wrapper, caching, batch fetching
│   ├── indicators.py              # 15+ technical indicators + calc_all_indicators()
│   ├── telegram_bot.py            # Telegram bot, formatters, send_alert()
│   ├── scheduler.py               # MarketScheduler (APScheduler + market hours)
│   ├── sentiment.py               # VADER + NewsAPI + GPT-4o pipeline, CostTracker
│   ├── position_sizing.py         # Fixed-risk sizing, Kelly criterion, constraints
│   ├── sectors.py                 # Multi-timeframe relative strength vs SPY
│   ├── earnings.py                # Earnings dates, beat streaks, risk flags
│   └── insider.py                 # SEC EDGAR insider transaction parsing
│
├── scripts/                       # Ops scripts
│   ├── setup_digitalocean.sh      # Server provisioning + systemd setup
│   ├── deploy.sh                  # Manual deploy via SSH
│   ├── bot_status.sh              # Check service status + memory
│   ├── bot_logs.sh                # Tail journalctl logs
│   ├── bot_restart.sh             # Safe stop → start → health check
│   └── train_model.py             # XGBoost training pipeline (run on Mac)
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # This file
│   ├── SETUP.md                   # Setup & deployment guide
│   └── TRADING_LOGIC.md           # Trading strategy documentation
│
├── models/                        # Local ML model cache (gitignored)
│
└── .github/workflows/
    └── deploy.yml                 # CI/CD: push to main → deploy
```

---

## Component Details

### 1. Entry Point — `main.py`

Responsibilities:
- Load `.env` via `python-dotenv`
- Configure logging (stdout + rotating file, 10 MB × 3 backups)
- Initialize Supabase, Telegram, health server, AgentLoop, scheduler
- Register SIGTERM/SIGINT handlers for graceful shutdown
- Main thread blocks on `shutdown_event` until signal received

### 2. Health Server — `health.py`

FastAPI server on port 8001 (crypto bot uses 8000).

| Endpoint   | Returns |
|------------|---------|
| `/health`  | `{status, uptime_seconds, memory_mb, memory_warning}` |
| `/status`  | Health data + `{open_positions, last_scan, next_weekly_scan, ...}` |

Memory warning triggers at 380 MB RSS.

### 3. Scheduler — `utils/scheduler.py`

APScheduler with `America/New_York` timezone, market-hours-aware.

| Schedule              | Time (ET)                    | Method                    |
|-----------------------|------------------------------|---------------------------|
| Weekly scan           | Sunday 8:00 PM               | `agent.weekly_scan()`     |
| Morning briefing      | Mon–Fri 8:30 AM              | `agent.morning_briefing()`|
| Intraday monitor      | Every 15 min, market hours   | `agent.intraday_monitor()`|
| After-market review   | Mon–Fri 4:30 PM              | `agent.after_market_review()` |
| Model check           | Mon–Fri 6:00 AM              | `agent.model_check()`     |

Skips US market holidays. DST-safe via `pytz`.

### 4. Agent Loop — `agent/agent.py`

Orchestrates all scheduled callbacks. Key method: `weekly_scan()`.

**Weekly Scan Pipeline (12 steps):**

```
1. Clear data cache
2. Scan universe (download prices, compute indicators, RS)
3. ML pre-filter (reject confidently bearish)
4. Sentiment analysis (NewsAPI + VADER + GPT-4o-mini)
5. Score candidates (composite: technical 30% + RS 20% + fundamental 12.75%
                     + sentiment 8.5% + insider 8.5% + macro 5% + ML 15%)
6. Build opportunities (entry/stop/target + position sizing)
7. Persist opportunities to Supabase
8. Persist daily scores
9. GPT-4o weekly briefing (natural language market summary)
10. Send Telegram alert (with optional AI Analyst Briefing)
11. Take equity snapshot
12. Purge old scores (monthly, >180 days)
```

### 5. Scanner — `agent/scanner.py`

Scans the S&P 500 universe and applies hard filters:

| Filter             | Threshold          | Rationale                      |
|--------------------|--------------------|--------------------------------|
| ADX                | > 20               | Trending stocks only           |
| RS percentile      | > 50th             | Above-average relative strength|
| ATR %              | < 8%               | Not excessively volatile       |
| Volume ratio       | > 0.5× average     | Sufficient liquidity           |
| Death cross        | Not present         | SMA 50 above SMA 200           |
| RSI                | < 80               | Not extremely overbought       |

### 6. Scorer — `agent/scorer.py`

Composite confidence score (0–100) with configurable weights:

| Component           | Weight | Source                  |
|---------------------|--------|-------------------------|
| Technical           | 30%    | RSI, ADX, MACD, EMA, BB|
| Relative Strength   | 20%    | RS vs SPY, multi-TF    |
| ML Models           | 15%    | XGBoost direction + vol |
| Fundamental         | 12.75% | PE, growth, margins    |
| Sentiment           | 8.5%   | VADER + GPT-4o-mini    |
| Insider             | 8.5%   | SEC EDGAR transactions  |
| Macro               | 5%     | VIX, regime assessment  |

Minimum confidence: 60/100 to generate an opportunity.

### 7. ML Models — `agent/ai_model.py` + `agent/feature_config.py`

Four XGBoost models (~12 MB total RAM):

| Model           | Features | Output                        |
|-----------------|----------|-------------------------------|
| Direction       | 15       | bullish / bearish / neutral   |
| Volatility      | 9        | high_vol / low_vol / neutral  |
| Earnings        | 11       | beat / miss / inline          |
| Sector Rotation | 8        | rotate_in / rotate_out / hold |

Models stored in Supabase Storage bucket `models`. Loaded daily at 6 AM ET.
Fallback to local `models/` directory if Supabase unavailable.

### 8. Sentiment Pipeline — `utils/sentiment.py`

```
NewsAPI (headlines) → VADER (-10 to +10) → GPT-4o-mini (enhanced analysis)
                                         → Aggregate (30% VADER + 70% GPT)
```

Cost controls:
- **CostTracker**: $5/month hard cap, kill switch at $4, monthly auto-reset
- **NewsAPI**: 90-call daily limit (100 API max), 6-hour cache per ticker
- **GPT-4o-mini**: ~$0.0001/call for daily sentiment
- **GPT-4o**: ~$0.0075/call for weekly briefing only

### 9. Persistence — `agent/persistence.py`

Supabase (PostgreSQL) with `stocks` schema, 8 tables:

| Table               | Purpose                          |
|---------------------|----------------------------------|
| `opportunities`     | Generated trade ideas            |
| `positions`         | Open / closed positions          |
| `daily_scores`      | Historical scoring data          |
| `events`            | Detected macro events            |
| `equity_snapshots`  | Daily portfolio value snapshots  |
| `model_versions`    | ML model metadata                |
| `bot_status`        | Runtime health & status KVs      |
| `watchlist`         | User-defined tickers to watch    |

### 10. Telegram — `utils/telegram_bot.py`

Alert types with formatted templates:

| Alert Type         | When                              |
|--------------------|-----------------------------------|
| `weekly_summary`   | Sunday scan complete              |
| `morning_briefing` | Mon–Fri 8:30 AM ET               |
| `eod_summary`      | Mon–Fri 4:30 PM ET               |
| `opportunity`      | New trade idea generated          |
| `action_needed`    | Stop/target hit, system alerts    |
| `position_update`  | Position P&L changes              |
| `bot_start`        | Service started                   |
| `bot_stop`         | Service stopped                   |

Messages >4096 chars auto-split at newline boundaries. Retry 3× with
exponential backoff. Weekly summary includes optional GPT-4o AI Analyst
Briefing section.

---

## Data Flow

```
                    ┌────────────────┐
                    │    yfinance    │
                    │ (S&P 500 data) │
                    └───────┬────────┘
                            │ prices, fundamentals
                            ▼
┌──────────┐    ┌───────────────────────┐    ┌──────────┐
│ NewsAPI  │───▶│      AgentLoop        │◀──▶│ Supabase │
│ (headlines)│  │                       │    │ (persist) │
└──────────┘   │  scan → filter → score │    └──────────┘
               │  → opportunities       │
┌──────────┐   │                        │    ┌──────────┐
│  OpenAI  │◀──│  sentiment analysis    │──▶│ Telegram  │
│ (GPT-4o) │   │  GPT briefing         │   │ (alerts)  │
└──────────┘   └───────────────────────┘   └──────────┘
```

---

## Infrastructure

| Service        | Tier          | Budget     | Purpose              |
|----------------|---------------|------------|----------------------|
| DigitalOcean   | $6/mo (1GB)   | $6/mo      | Compute (shared w/ crypto) |
| Supabase       | Free (500MB)  | $0         | PostgreSQL + Storage |
| Telegram       | Free          | $0         | Alert delivery       |
| yfinance       | Free          | $0         | Market data          |
| NewsAPI        | Free (100/day)| $0         | News headlines       |
| OpenAI         | Pay-per-use   | ~$0.06–0.60| Sentiment + briefings|
| GitHub Actions | Free (2000m)  | $0         | CI/CD                |

**Total: ~$6/mo**

### Memory Budget (400 MB target)

| Component        | Estimate   |
|------------------|------------|
| Python runtime   | 30 MB      |
| FastAPI + uvicorn| 25 MB      |
| pandas/numpy     | 40 MB      |
| yfinance data    | 50–80 MB   |
| XGBoost models   | 12 MB      |
| Application code | 20 MB      |
| Headroom         | ~170 MB    |
| **Total**        | **~230 MB**|

systemd enforces `MemoryMax=450M` and `MemoryHigh=400M`.

---

## Error Handling Strategy

| Layer            | Strategy                                     |
|------------------|----------------------------------------------|
| Data fetching    | 3 retries with exponential backoff, cache fallback |
| Supabase         | Connection retry, local file queue on failure |
| Telegram         | 3 retries, rate limit handling, message splitting |
| OpenAI           | CostTracker budget enforcement, template fallback |
| ML models        | Graceful disable, technical-only scoring      |
| Scheduler        | Missed run detection, backup schedule          |
| Memory           | Monitor RSS every 60s, emergency mode at 380 MB |

Every `except` block logs the full traceback. Critical failures send a
Telegram alert before stopping.

---

## Security

- Non-root process (`botuser` in Docker, `algotrading` on droplet)
- Secrets via environment variables only (never in code/git)
- `.env` is gitignored
- Supabase service role key (not anon key)
- systemd `MemoryMax` prevents OOM affecting other services
- Health endpoint on localhost only (no sensitive data exposed)
