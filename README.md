# stocks-agent 📊

**Stock market advisory bot for US swing trading (S&P 500 focus)**

Advisory only — the bot suggests trades, you execute them.

## What It Does

- **Weekly scan** of 500 S&P stocks → top picks with entry, stop, target
- **Composite scoring** (technical 30% + RS 20% + ML 15% + fundamental 13% + sentiment 9% + insider 9% + macro 5%)
- **XGBoost ML models** for direction prediction and pre-filtering
- **VADER + GPT-4o** news sentiment with $5/month cost cap
- **Telegram alerts** — weekly picks, morning briefing, intraday stops, EOD summary
- **AI Analyst Briefing** — GPT-4o writes a weekly market summary in plain English

## Quick Start

```bash
# Clone
git clone https://github.com/Dorpeer95/stocks-trading.git
cd stocks-trading

# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (see docs/SETUP.md)

# Run (dry-run mode — no real alerts)
STOCKS_DRY_RUN=true python main.py
```

See [docs/SETUP.md](docs/SETUP.md) for complete setup including Supabase,
Telegram, and DigitalOcean deployment.

## Architecture

```
stocks-trading/
├── main.py                        # Entry point, signal handling, graceful shutdown
├── health.py                      # FastAPI health server (:8001) + memory watchdog
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Multi-stage production image
├── .env.example                   # Environment variable template
│
├── agent/                         # Intelligence layer
│   ├── agent.py                   # AgentLoop — 12-step weekly scan pipeline
│   ├── scanner.py                 # Universe scan + hard filters
│   ├── scorer.py                  # Composite confidence scoring (0-100)
│   ├── events.py                  # Macro event detection, market regime
│   ├── portfolio.py               # Position tracking, EOD summaries
│   ├── persistence.py             # Supabase CRUD (8 tables)
│   ├── ai_model.py                # XGBoost model manager (4 models)
│   └── feature_config.py          # ML feature definitions
│
├── utils/                         # Shared utilities
│   ├── data_loader.py             # yfinance wrapper with cache + retry
│   ├── indicators.py              # 15+ technical indicators
│   ├── telegram_bot.py            # Alert formatting & delivery
│   ├── scheduler.py               # Market-hours-aware scheduler
│   ├── sentiment.py               # VADER + NewsAPI + GPT-4o pipeline
│   ├── position_sizing.py         # Fixed-risk sizing, Kelly criterion
│   ├── sectors.py                 # Multi-timeframe RS vs SPY
│   ├── earnings.py                # Earnings dates, beat streaks
│   └── insider.py                 # SEC EDGAR insider parsing
│
├── scripts/                       # Operations
│   ├── setup_digitalocean.sh      # Server provisioning + systemd
│   ├── deploy.sh                  # Manual deploy via SSH
│   ├── bot_status.sh              # Service status + memory
│   ├── bot_logs.sh                # Tail journalctl logs
│   ├── bot_restart.sh             # Safe restart with health check
│   └── train_model.py             # XGBoost training pipeline
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # System design & data flow
│   ├── SETUP.md                   # Setup & deployment guide
│   └── TRADING_LOGIC.md           # Scoring, indicators, entry/exit rules
│
└── .github/workflows/
    └── deploy.yml                 # CI/CD: validate → deploy → rollback
```

## Schedule

| Event             | Time (ET)          | Frequency     |
|-------------------|--------------------|---------------|
| Weekly scan       | Sunday 8:00 PM     | Weekly        |
| Morning briefing  | Mon–Fri 8:30 AM    | Daily         |
| Intraday monitor  | Every 15 min       | Market hours  |
| EOD summary       | Mon–Fri 4:30 PM    | Daily         |
| Model check       | Mon–Fri 6:00 AM    | Daily         |

## Status

- **Phase 1 ✅** — Foundation (data, indicators, persistence, alerts, scheduler)
- **Phase 2 ✅** — Intelligence (scanner, scorer, events, portfolio, position sizing)
- **Phase 3 ✅** — ML Integration (XGBoost models, feature engineering, pre-filtering)
- **Phase 4 ✅** — GPT-4o & Sentiment (VADER, NewsAPI, GPT analysis, AI briefing)
- **Phase 5 ✅** — Polish & Deploy (docs, CI/CD, Dockerfile, error hardening)

## Infrastructure

| Service        | Tier              | Monthly Cost |
|----------------|-------------------|-------------|
| DigitalOcean   | $6/mo (1GB/1CPU)  | $6          |
| Supabase       | Free (500MB)      | $0          |
| Telegram       | Free              | $0          |
| yfinance       | Free              | $0          |
| NewsAPI        | Free (100 req/day)| $0          |
| OpenAI         | Pay-per-use       | ~$0.06–0.60 |
| **Total**      |                   | **~$6/mo**  |

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design, data flow, component details
- [Setup Guide](docs/SETUP.md) — local dev, deployment, API keys, troubleshooting
- [Trading Logic](docs/TRADING_LOGIC.md) — scoring weights, indicators, position sizing
- [Development Plan](DEVELOPMENT_PLAN.md) — full architecture spec (10 deliverables)