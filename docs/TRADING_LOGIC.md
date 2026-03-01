# Trading Logic

> Last updated: 2026-03-01

## Overview

stocks-agent is a **swing trading advisory bot** focused on the S&P 500.
It identifies stocks with strong technicals, relative strength, and
favorable sentiment, then delivers buy/sell ideas with precise entry,
stop, and target levels.

**Holding period:** 1–6 weeks
**Style:** Trend-following with momentum confirmation
**Universe:** S&P 500 (~500 stocks)

---

## 1. Scoring System

Every candidate receives a composite confidence score from 0–100.
Only candidates scoring ≥ 60 become opportunities.

### Weight Allocation

| Component           | Weight | What it measures                     |
|---------------------|--------|--------------------------------------|
| Technical           | 30.0%  | Trend, momentum, volatility signals  |
| Relative Strength   | 20.0%  | Performance vs SPY across timeframes |
| ML Models           | 15.0%  | XGBoost directional + volatility     |
| Fundamental         | 12.75% | Valuation, growth, profitability     |
| Sentiment           | 8.5%   | News tone (VADER + GPT-4o-mini)      |
| Insider Activity    | 8.5%   | SEC EDGAR insider transactions       |
| Macro               | 5.0%   | Market regime, VIX, event risk       |

> Weights redistribute automatically when ML is disabled
> (`STOCKS_ENABLE_ML=false` → ML weight goes to technical & RS).

### Sub-Score Ranges

Each sub-score maps to 0–100:

- **0–20**: Very bearish / very poor
- **20–40**: Bearish / below average
- **40–60**: Neutral / average
- **60–80**: Bullish / above average
- **80–100**: Very bullish / excellent

---

## 2. Technical Indicators

### Indicators Computed

| Indicator             | Function              | Parameters            |
|-----------------------|-----------------------|-----------------------|
| RSI                   | Relative Strength Idx | 14-period             |
| MACD                  | Moving Avg Convergence| 12, 26, 9             |
| ADX                   | Avg Directional Index | 14-period             |
| ATR (%)               | Avg True Range / price| 14-period             |
| Bollinger Bands       | Price channels         | 20-period, 2σ         |
| SMA 20 / 50 / 200     | Moving averages       |                       |
| EMA 12 / 26 / 50 / 200| Exp. moving averages  |                       |
| VWAP                  | Volume-weighted price  | Session               |
| OBV                   | On-balance volume      |                       |
| Volume ratio          | Current / 20-day avg   |                       |
| Distance 52W high/low | % from annual extremes |                       |
| Momentum (4/13/26w)   | % price change         |                       |

### Technical Sub-Score Calculation

```
Base score starts at 50 (neutral)

RSI:
  < 30  → +15 (oversold bounce)
  30-45 → +10 (pulling back in uptrend)
  45-55 → +5  (neutral)
  55-70 → +10 (strong momentum)
  > 80  → -10 (overbought risk)

MACD:
  bullish crossover → +10
  bearish crossover → -10

ADX:
  > 25  → +10 (strong trend)
  < 15  → -5  (no trend, chop)

EMA alignment:
  EMA 12 > 26 > 50 > 200 → +15 (perfect alignment)
  Golden cross (50 > 200)  → +10
  Death cross (50 < 200)   → -15

Volume:
  ratio > 1.5 → +5 (above-average interest)
  ratio < 0.5 → -5 (low interest)
```

---

## 3. Relative Strength

Multi-timeframe relative strength vs SPY:

| Timeframe | Weight | Calculation                          |
|-----------|--------|--------------------------------------|
| 4 weeks   | 25%    | (stock_return − SPY_return) × 100    |
| 13 weeks  | 35%    | (stock_return − SPY_return) × 100    |
| 26 weeks  | 40%    | (stock_return − SPY_return) × 100    |

RS percentile ranks each stock against the full S&P 500 universe.
Only stocks in the **top 50th percentile** pass the scanner filter.

### RS Sub-Score

Direct mapping: RS percentile → sub-score (e.g., 75th percentile → 75).

---

## 4. Scanner Filters

Hard filters applied before scoring (pass/fail):

| Filter             | Threshold        | Why                               |
|--------------------|------------------|-----------------------------------|
| ADX                | > 20             | Must be in a trend                |
| RS percentile      | > 50th           | Better than average vs SPY        |
| ATR %              | < 8%             | Not too volatile for swing trades |
| Volume ratio       | > 0.5×           | Minimum liquidity                 |
| Death cross        | Absent           | SMA 50 must be above SMA 200      |
| RSI                | < 80             | Not extremely overbought          |

### ML Pre-Filter (Phase 3)

After hard filters, XGBoost direction model pre-filters candidates:
- **Confidently bearish** (signal=bearish AND confidence > 60%) → **rejected**
- All others pass through to scoring

---

## 5. ML Models

Four XGBoost binary classifiers:

### Direction Model (15 features)

Predicts whether a stock will go up in the next 1–4 weeks.

Features: RSI, ADX, ATR%, BB width, MACD signal, EMA cross, volume
ratio, distance from 52W high, close/SMA50 ratio, SMA50/SMA200 ratio,
RS percentile, 4W/13W/26W momentum, RS vs SPY.

### Volatility Model (9 features)

Predicts whether volatility will expand or contract.

Features: ATR%, BB width, volume ratio, ADX, RSI, 4W/13W momentum,
distance from 52W high, close/SMA50 ratio.

### Earnings Model (11 features)

Predicts earnings beat probability.

Features: RSI, ADX, volume ratio, 4W/13W momentum, PE ratio, forward
PE, revenue growth, profit margin, RS percentile, historical beat
streak, days to earnings.

### Sector Rotation Model (8 features)

Predicts whether capital is rotating into or out of a sector.

Features: sector avg RS, median RS, sector 4W momentum, SPY 4W
momentum, sector vs SPY spread, momentum delta, relative momentum
strength, RS trend.

### ML Sub-Score Mapping

```
Direction signal:
  bullish  → 50 + (probability × 50)  → 50-100
  bearish  → 50 - (probability × 50)  → 0-50
  neutral  → 50

Volatility adjustment:
  low volatility + high confidence → +5 bonus (calm = good for entries)
  high volatility + high confidence → -5 penalty

Final ML score: clamped to 0-100
```

---

## 6. Sentiment Pipeline

### Data Flow

```
NewsAPI headlines → VADER scoring → GPT-4o-mini analysis → Aggregate
```

### VADER Scoring

VADER (Valence Aware Dictionary for sEntiment Reasoning) processes
each headline and returns a compound score (-1 to +1), scaled to
-10 to +10 across all headlines.

### GPT-4o-mini Enhancement

When enabled, GPT-4o-mini analyzes the combined headlines with context:
- Overall sentiment (-10 to +10)
- Key themes and catalysts
- Risk flags (SEC investigations, lawsuits, etc.)
- Plain English summary

### Aggregation

```
If GPT available:  final = VADER × 0.3 + GPT × 0.7
If VADER only:     final = VADER score
If no news:        final = 0 (neutral)
```

### Sentiment Sub-Score

```
Raw score (-10 to +10) → mapped to 0-100:
  -10 → 0
    0 → 50
  +10 → 100

Formula: sub_score = (raw + 10) × 5

Risk flag penalty: -5 per flag (max -15)
```

---

## 7. Position Sizing

### Fixed-Risk Model

Every trade risks exactly **2% of portfolio value**.

```
risk_amount = portfolio_value × 0.02
entry_price = current close
stop_loss   = entry - (ATR × 2.0)
stop_distance = entry - stop_loss
shares = floor(risk_amount / stop_distance)
position_size = shares × entry_price
```

### Constraints

| Constraint              | Limit    | Enforcement              |
|-------------------------|----------|--------------------------|
| Max position size       | 25%      | Cap at portfolio × 0.25  |
| Cash reserve            | 10%      | Must keep 10% uninvested |
| Max open positions      | 8        | Won't open 9th           |
| Min position size       | $500     | Skip if too small        |

### Kelly Criterion (informational)

Half-Kelly is computed for each opportunity but used as a reference,
not as the primary sizing method:

```
kelly = (win_probability × avg_win - loss_probability × avg_loss) / avg_win
half_kelly = kelly / 2
```

### Regime Adjustment

Position sizes are adjusted based on market regime:

| Regime     | Adjustment | Rationale                   |
|------------|------------|-----------------------------|
| Crisis     | × 0.5      | Halve positions in crashes  |
| Volatile   | × 0.75     | Reduce in high-VIX regimes  |
| Normal     | × 1.0      | Standard sizing             |
| Calm       | × 1.0      | Standard sizing             |

---

## 8. Entry & Exit Logic

### Entry Zone

```
entry_price_low  = close × 0.99  (1% below current)
entry_price_high = close × 1.01  (1% above current)
```

The bot provides a price zone, not a single price. The user sets a
limit order within this range.

### Stop Loss

```
stop_loss = entry - (ATR_14 × 2.0)
```

ATR-based stops adapt to each stock's volatility. A stock with 5%
ATR gets a wider stop than one with 1.5% ATR.

### Target Price

```
target = entry + (ATR_14 × 4.0)
```

This gives a 2:1 reward-to-risk ratio by default.

### Risk/Reward

```
risk_usd   = shares × (entry - stop_loss)
reward_usd = shares × (target - entry)
risk_reward_ratio = reward_usd / risk_usd  (target: ≥ 2.0)
```

### Exit Signals (Intraday Monitor)

The bot checks open positions every 15 minutes during market hours:

| Signal            | Action       | Urgency |
|-------------------|-------------|---------|
| Price ≤ stop loss  | Sell (loss)  | High    |
| Price ≥ target     | Sell (gain)  | High    |
| Days held > 30     | Review       | Medium  |
| RSI > 80           | Tighten stop | Medium  |
| Death cross forms  | Sell         | High    |
| Volume spike down  | Review       | Medium  |

---

## 9. Market Regime

The bot assesses overall market conditions using macro data:

| Indicator   | Source     | Thresholds                         |
|-------------|------------|------------------------------------|
| VIX         | yfinance   | < 15 calm, 15-25 normal, 25-35 volatile, > 35 crisis |
| SPY trend   | yfinance   | 50-day vs 200-day SMA              |
| Treasury    | yfinance   | 10Y yield level and direction      |
| Dollar      | yfinance   | DXY trend                          |

### Regime Classification

| Regime    | VIX     | SPY Trend | Effect on Bot                    |
|-----------|---------|-----------|----------------------------------|
| Crisis    | > 35    | Down      | Halve sizes, tighten stops       |
| Volatile  | 25–35   | Any       | Reduce sizes 25%, wider stops    |
| Normal    | 15–25   | Up        | Standard operation               |
| Calm      | < 15    | Up        | Standard operation               |

---

## 10. Insider Activity

Parsed from SEC EDGAR insider transaction filings.

### Signal Interpretation

| Pattern                          | Signal    | Score Effect |
|----------------------------------|-----------|-------------|
| Multiple insiders buying         | Bullish   | +15         |
| Large cluster buy (>3 insiders)  | Very bullish | +20      |
| CEO/CFO buying                   | Bullish   | +10         |
| Multiple insiders selling        | Bearish   | -10         |
| Routine option exercises         | Neutral   | 0           |
| No recent activity               | Neutral   | 0           |

---

## 11. Fundamental Scoring

When available from yfinance:

| Metric           | Favorable           | Score Effect |
|------------------|---------------------|-------------|
| P/E ratio        | < sector average     | +10         |
| Forward P/E      | < trailing P/E       | +5          |
| Revenue growth   | > 10% YoY            | +10         |
| Profit margin    | > 10%                | +5          |
| Debt/equity      | < 1.0                | +5          |

Fundamentals are fetched per-ticker (slow), so they are only
retrieved during full scans with `fetch_extras=True`.

---

## 12. Alert Schedule

| Alert              | Time (ET)     | Frequency      | Content                     |
|--------------------|---------------|----------------|-----------------------------|
| Weekly scan        | Sun 8 PM      | Weekly         | Top picks + AI briefing     |
| Morning briefing   | Mon–Fri 8:30  | Daily          | Macro + positions + events  |
| Intraday alerts    | Market hours  | As needed      | Stop/target hits            |
| EOD summary        | Mon–Fri 4:30  | Daily          | P&L + recommendations      |

---

## 13. Cost Budget

| Service       | Per Call     | Daily Usage    | Monthly Cost    |
|---------------|-------------|----------------|-----------------|
| GPT-4o-mini   | ~$0.0001    | ~15 calls      | ~$0.045         |
| GPT-4o        | ~$0.0075    | ~1 call/week   | ~$0.030         |
| NewsAPI       | Free        | ~21 calls      | $0              |
| yfinance      | Free        | ~500+ calls    | $0              |
| **Total AI**  |             |                | **~$0.06–0.60** |

Hard cap: $5/month enforced by `CostTracker` class.
Kill switch at $4 — stops all GPT calls, falls back to templates.
