# stocks-agent 📊

**Rules-based swing-trade screener for US stocks (S&P 500 focus)**

Advisory only — the bot ranks candidates and sends alerts, you execute trades manually.

> **Honest status:** This is a **rules-based screener** with an XGBoost pre-filter
> and a state-machine portfolio tracker. GPT-4o is currently used only to **write
> narrative briefings** after decisions are made — it has **no influence** on
> entries, exits, sizing, or risk. Converting GPT into a real decision-maker
> (entry veto + exit analyst) is Phase 2 of the roadmap; before that, Phase 1
> must prove the strategy has positive expectancy on 3+ years of historical
> data via the pipeline backtester (`scripts/backtest_pipeline.py`, Mac-only).
> Until Phase 1 clears its acceptance gate, treat every alert as a **screener
> output, not a trade recommendation**.

## What It Does

- **Weekly scan** of 500 S&P stocks → ranked picks with entry, stop, target (rules + ML)
- **Composite scoring** (technical 30% + RS 20% + ML 15% + fundamental 13% + sentiment 9% + insider 9% + macro 5%)
- **XGBoost ML models** pre-filter candidates (direction, volatility, earnings, sector rotation)
- **Portfolio state machine** tracks holdings through NEW → WATCH_CANDIDATE → ACTIVE → WATCH → EXITING
- **VADER + GPT-4o** news sentiment scoring with $5/month cost cap
- **Realtime stop/target checks** via Alpaca last-trade (<2s lag); yfinance 5m fallback
- **Telegram alerts** — weekly picks, morning briefing, intraday stops, EOD summary
- **Telegram commands** — `/status` (live health & holdings), `/pause`, `/resume`
- **Weekly narrative briefing** — GPT-4o summarizes the week in plain English (read-only; no decision influence)

## What It Does NOT Do (yet)

- ❌ **No auto-execution.** Alerts only — the user places every order.
- ❌ **No LLM decision-making.** GPT writes briefings but does not approve, veto, or size any trade. (Planned for Phase 2.)
- ❌ **No validated edge.** Full pipeline has not been backtested end-to-end. Ship-blocker for live money use. (Phase 1.)

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

## Build Status

- **Phase 1 ✅** — Foundation (data, indicators, persistence, alerts, scheduler)
- **Phase 2 ✅** — Intelligence (scanner, scorer, events, portfolio, position sizing)
- **Phase 3 ✅** — ML Integration (XGBoost models, feature engineering, pre-filtering)
- **Phase 4 ✅** — GPT-4o & Sentiment (VADER, NewsAPI, GPT narrative briefing)
- **Phase 5 ✅** — Polish & Deploy (docs, CI/CD, Dockerfile, error hardening)

## Trust Roadmap (what's still missing before this is a real "AI portfolio advisor")

- **⏳ Edge proof** — `scripts/backtest_pipeline.py` runs the real scan → score → portfolio cycle on 3+ years of history. Acceptance gate: profit factor ≥ 1.5, max drawdown ≤ 20%, ≥ 80 trades, beats SPY buy-and-hold. **Until this passes, do not put real money behind the alerts.**
- **⏳ LLM decision layer** — `agent/llm_agent.py` with tool registry. Claude Opus / GPT-4o becomes the entry veto gate, exit analyst, and weekly portfolio reviewer. Every decision logged to an `llm_decisions` audit table. Only unlocked after edge proof passes.
- **⏳ A/B gate check** — Re-run the backtest with `--llm-gate=on` to prove the LLM layer measurably improves drawdown before enabling it live.

See the full plan in `/root/.claude/plans/goal-improve-the-exsist-async-lark.md` (dev machine only).

## Infrastructure

| Service        | Tier              | Monthly Cost |
|----------------|-------------------|-------------|
| DigitalOcean   | $6/mo (1GB/1CPU)  | $6          |
| Supabase       | Free (500MB)      | $0          |
| Telegram       | Free              | $0          |
| yfinance       | Free              | $0          |
| NewsAPI        | Free (100 req/day)| $0          |
| OpenAI         | Pay-per-use       | ~$0.06–0.60 |
| **Total (current, narrator-only)** | | **~$6/mo** |
| *Projected after Phase 2 LLM agent (with prompt caching + batched decisions)* | | *~$8–11/mo* |

> **Dev / test memory rule:** The DO server is 1GB RAM with a 430MB critical
> threshold. `scripts/backtest_pipeline.py`, `pytest`, and `scripts/train_model.py`
> must run on a local Mac (or GitHub Actions) only — never on the server.
> Model weights are trained locally and uploaded to Supabase Storage; the
> server downloads them via the daily model check.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design, data flow, component details
- [Setup Guide](docs/SETUP.md) — local dev, deployment, API keys, troubleshooting
- [Trading Logic](docs/TRADING_LOGIC.md) — scoring weights, indicators, position sizing
- [Development Plan](DEVELOPMENT_PLAN.md) — full architecture spec (10 deliverables)