"""
Multi-factor scorer — assigns a 0-100 confidence score to each candidate.

Combines technical, relative-strength, fundamental, sentiment,
insider, macro, and ML signals into a single score.  Produces
``opportunities`` records ready for persistence.
"""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

from utils.earnings import earnings_risk_flag, fetch_earnings_history, calc_beat_streak
from utils.insider import get_insider_signal
from agent.macro_events import get_macro_bias, apply_macro_bias

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight configuration (env-overridable)
# ---------------------------------------------------------------------------
# When ML is active the weights are re-distributed to include W_ML.
# Traditional weights are scaled down proportionally so the total = 1.0.
ENABLE_ML = os.getenv("STOCKS_ENABLE_ML", "true").lower() == "true"
W_ML = float(os.getenv("STOCKS_W_ML", "0.15"))  # 15% when active

# Base weights (sum = 1.0 when ML is off)
_W_TECHNICAL_BASE = float(os.getenv("STOCKS_W_TECHNICAL", "0.35"))
_W_RS_BASE = float(os.getenv("STOCKS_W_RS", "0.25"))
_W_FUNDAMENTAL_BASE = float(os.getenv("STOCKS_W_FUNDAMENTAL", "0.15"))
_W_SENTIMENT_BASE = float(os.getenv("STOCKS_W_SENTIMENT", "0.10"))
_W_INSIDER_BASE = float(os.getenv("STOCKS_W_INSIDER", "0.10"))
_W_MACRO_BASE = float(os.getenv("STOCKS_W_MACRO", "0.05"))

if ENABLE_ML:
    # Scale traditional weights down so total = 1.0
    _scale = 1.0 - W_ML
    W_TECHNICAL = _W_TECHNICAL_BASE * _scale
    W_RS = _W_RS_BASE * _scale
    W_FUNDAMENTAL = _W_FUNDAMENTAL_BASE * _scale
    W_SENTIMENT = _W_SENTIMENT_BASE * _scale
    W_INSIDER = _W_INSIDER_BASE * _scale
    W_MACRO = _W_MACRO_BASE * _scale
else:
    W_TECHNICAL = _W_TECHNICAL_BASE
    W_RS = _W_RS_BASE
    W_FUNDAMENTAL = _W_FUNDAMENTAL_BASE
    W_SENTIMENT = _W_SENTIMENT_BASE
    W_INSIDER = _W_INSIDER_BASE
    W_MACRO = _W_MACRO_BASE
    W_ML = 0.0

# Minimum confidence to become an opportunity
MIN_CONFIDENCE = int(os.getenv("STOCKS_MIN_CONFIDENCE", "60"))


# ---------------------------------------------------------------------------
# Sub-scores (each returns 0-100)
# ---------------------------------------------------------------------------

def score_technical(stock: Dict[str, Any]) -> float:
    """Score based on technical indicators (0-100)."""
    score = 50.0  # start neutral

    rsi = stock.get("rsi_14")
    adx = stock.get("adx")
    macd = stock.get("macd_signal")
    ema_cross = stock.get("ema_cross")
    atr_pct = stock.get("atr_pct")
    bb_width = stock.get("bb_width")
    vol_ratio = stock.get("volume_ratio")
    dist_high = stock.get("distance_52w_high")

    # RSI zone scoring
    if rsi is not None:
        if 40 <= rsi <= 60:
            score += 10   # neutral zone = room to run
        elif 30 <= rsi < 40:
            score += 15   # oversold bounce potential
        elif rsi < 30:
            score += 5    # deeply oversold — risky reversal
        elif 60 < rsi <= 70:
            score += 5    # strong but not overextended
        elif rsi > 70:
            score -= 10   # overbought risk

    # ADX (trend strength)
    if adx is not None:
        if adx > 40:
            score += 15
        elif adx > 25:
            score += 10
        elif adx > 20:
            score += 5

    # MACD signal
    if macd == "bullish":
        score += 10
    elif macd == "bearish":
        score -= 10

    # EMA cross
    if ema_cross == "golden_cross":
        score += 15
    elif ema_cross == "bullish":
        score += 5
    elif ema_cross == "death_cross":
        score -= 15
    elif ema_cross == "bearish":
        score -= 5

    # Volume confirmation
    if vol_ratio is not None:
        if vol_ratio > 2.0:
            score += 10
        elif vol_ratio > 1.3:
            score += 5
        elif vol_ratio < 0.5:
            score -= 5

    # Near 52-week high (momentum play)
    if dist_high is not None:
        if -5 <= dist_high <= 0:
            score += 5    # near high = strength
        elif dist_high < -20:
            score -= 5    # far from high

    return max(0, min(100, score))


def score_relative_strength(stock: Dict[str, Any]) -> float:
    """Score based on RS percentile and momentum (0-100)."""
    rs_pct = stock.get("rs_percentile", 50)
    m4w = stock.get("momentum_4w")
    m13w = stock.get("momentum_13w")
    rs_sector = stock.get("rs_vs_sector")

    score = rs_pct  # RS percentile is already 0-100

    # Bonus for positive momentum
    if m4w is not None and m4w > 5:
        score += 5
    if m13w is not None and m13w > 10:
        score += 5

    # Bonus for outperforming sector
    if rs_sector is not None and rs_sector > 5:
        score += 5

    return max(0, min(100, score))


def score_fundamental(stock: Dict[str, Any], fundamentals: Optional[Dict[str, Any]] = None) -> float:
    """Score based on fundamental data (0-100)."""
    score = 50.0

    if fundamentals is None:
        return score

    pe = fundamentals.get("pe_ratio")
    fwd_pe = fundamentals.get("forward_pe")
    profit_margin = fundamentals.get("profit_margin")
    revenue_growth = fundamentals.get("revenue_growth")
    short_pct = fundamentals.get("short_pct_float")

    # PE ratio
    if pe is not None and fwd_pe is not None:
        if fwd_pe < pe:
            score += 10  # earnings expected to grow
        if 5 < pe < 25:
            score += 5   # reasonable valuation

    # Profitability
    if profit_margin is not None and profit_margin > 0.15:
        score += 10
    elif profit_margin is not None and profit_margin > 0.05:
        score += 5

    # Revenue growth
    if revenue_growth is not None and revenue_growth > 0.1:
        score += 10
    elif revenue_growth is not None and revenue_growth > 0:
        score += 5

    # Short interest (contrarian: high short = squeeze potential but risky)
    if short_pct is not None:
        if short_pct > 0.20:
            score -= 5   # too much short interest = risk
        elif short_pct > 0.10:
            score += 0   # neutral
        else:
            score += 5   # low short = healthy

    return max(0, min(100, score))


def score_insider(ticker: str) -> float:
    """Score based on insider activity (0-100)."""
    try:
        signal = get_insider_signal(ticker)
        base = 50.0
        adj = signal.get("score_adjustment", 0)

        # Scale adjustment: ±10 from insider maps to ±20 in score
        return max(0, min(100, base + adj * 2))
    except Exception as e:
        logger.debug(f"Insider scoring skipped for {ticker}: {e}")
        return 50.0  # neutral if unavailable


def score_earnings_risk(ticker: str) -> float:
    """Score based on earnings proximity (0-100, higher = safer)."""
    try:
        risk = earnings_risk_flag(ticker)
        if not risk.get("has_risk"):
            return 80.0  # no imminent earnings = good

        days = risk.get("days_to_earnings", 30)
        if days is None:
            return 60.0
        if days <= 3:
            return 10.0
        elif days <= 7:
            return 30.0
        elif days <= 14:
            return 50.0
        else:
            return 70.0
    except Exception as e:
        logger.debug(f"Earnings scoring skipped for {ticker}: {e}")
        return 60.0


def score_macro(regime: Optional[Dict[str, Any]] = None) -> float:
    """Score based on macro regime (0-100)."""
    if regime is None:
        return 50.0

    mood = regime.get("mood", "Neutral")
    if mood == "Bullish":
        return 80.0
    elif mood == "Neutral":
        return 50.0
    else:
        return 25.0


def score_sentiment(sentiment_data: Optional[Dict[str, Any]] = None) -> float:
    """Score based on news sentiment (0-100).

    Parameters
    ----------
    sentiment_data : Output from ``sentiment.aggregate_sentiment()``.
        Key field: ``score`` in range -10 to +10.

    Returns
    -------
    0-100 score.  50 = neutral / no data.
    """
    if sentiment_data is None:
        return 50.0

    raw = sentiment_data.get("score", 0.0)

    # raw is -10 to +10 → map to 0-100
    # -10 → 0,  0 → 50,  +10 → 100
    score = (raw + 10) * 5  # linear mapping

    # Bonus / penalty for risk flags
    risk_flags = sentiment_data.get("risk_flags", [])
    if risk_flags:
        score -= min(15, len(risk_flags) * 5)

    return max(0, min(100, round(score, 1)))


def score_ml(ml_predictions: Optional[Dict[str, Any]] = None) -> float:
    """Score based on ML model predictions (0-100).

    Parameters
    ----------
    ml_predictions : Dict with keys from ``ModelManager.predict_batch()``:
        - ``direction`` : dict with ``signal`` (bullish/bearish/neutral),
                          ``probability`` (0-1), ``confidence`` (0-100)
        - ``volatility`` : dict with ``signal``, ``probability``, ``confidence``

    Returns
    -------
    0-100 score. 50 = neutral / no predictions available.
    """
    if ml_predictions is None:
        return 50.0

    score = 50.0

    # Direction model (dominant signal)
    direction = ml_predictions.get("direction")
    if direction and direction.get("signal") is not None:
        prob = direction.get("probability", 0.5)
        signal = direction["signal"]
        if signal == "bullish":
            # Scale probability [0.55, 1.0] → [+5, +35]
            boost = min(35, max(5, (prob - 0.5) * 70))
            score += boost
        elif signal == "bearish":
            # Scale probability [0.0, 0.45] → [-5, -35]
            penalty = min(35, max(5, (0.5 - prob) * 70))
            score -= penalty
        # neutral → no change

    # Volatility model (risk modifier)
    volatility = ml_predictions.get("volatility")
    if volatility and volatility.get("signal") is not None:
        vol_signal = volatility["signal"]
        if vol_signal == "bullish":
            # High volatility expected — slight negative for swing trades
            score -= 5
        elif vol_signal == "bearish":
            # Low volatility expected — good for controlled trades
            score += 5

    return max(0, min(100, round(score, 1)))


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_candidate(
    stock: Dict[str, Any],
    fundamentals: Optional[Dict[str, Any]] = None,
    regime: Optional[Dict[str, Any]] = None,
    skip_api_calls: bool = False,
    ml_predictions: Optional[Dict[str, Any]] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    macro_bias: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the composite confidence score for a candidate.

    Parameters
    ----------
    stock : Candidate dict from scanner (with indicators + RS).
    fundamentals : Output from ``data_loader.fetch_fundamentals()``.
    regime : Output from ``events.assess_market_regime()``.
    skip_api_calls : If True, skip insider/earnings API calls
                     (for batch speed; use cached or default scores).
    ml_predictions : Output from ``ModelManager.predict_batch()`` for
                     this ticker.  When provided and ENABLE_ML is True,
                     the ML score is included in the composite.
    sentiment_data : Output from ``sentiment.aggregate_sentiment()``
                     for this ticker.  If provided, real news sentiment
                     is used; otherwise falls back to earnings-based proxy.

    Returns
    -------
    The stock dict enriched with ``confidence``, ``sub_scores``,
    and ``setup_type``.
    """
    ticker = stock.get("ticker", "???")

    # Sub-scores
    tech = score_technical(stock)
    rs = score_relative_strength(stock)
    fund = score_fundamental(stock, fundamentals)

    if skip_api_calls:
        insider = 50.0
        earnings = 60.0
    else:
        insider = score_insider(ticker)
        earnings = score_earnings_risk(ticker)

    macro = score_macro(regime)
    
    # Apply GPT macro bias to specific sectors
    sector = stock.get("sector")
    if not sector and fundamentals:
        sector = fundamentals.get("sector")
    
    if sector and macro_bias:
        macro = apply_macro_bias(macro, sector, macro_bias)

    ml = score_ml(ml_predictions) if ENABLE_ML else 50.0

    # Sentiment: use real news sentiment if available, else earnings-based proxy
    if sentiment_data is not None:
        sent = score_sentiment(sentiment_data)
    else:
        # Fallback: derive sentiment-like score from earnings safety
        sent = (earnings * 0.7 + 30)  # 0-100 range

    # Composite (weighted sum)
    confidence = (
        tech * W_TECHNICAL
        + rs * W_RS
        + fund * W_FUNDAMENTAL
        + sent * W_SENTIMENT
        + insider * W_INSIDER
        + macro * W_MACRO
        + ml * W_ML
    )

    confidence = round(max(0, min(100, confidence)), 1)

    # Detect setup type
    setup = detect_setup_type(stock)

    stock["confidence"] = confidence
    stock["sub_scores"] = {
        "technical": round(tech, 1),
        "relative_strength": round(rs, 1),
        "fundamental": round(fund, 1),
        "sentiment": round(sent, 1),
        "insider": round(insider, 1),
        "earnings_safety": round(earnings, 1),
        "macro": round(macro, 1),
        "ml": round(ml, 1),
    }
    stock["setup_type"] = setup

    # Attach sentiment metadata if available
    if sentiment_data:
        stock["sentiment_summary"] = sentiment_data.get("gpt_summary")
        stock["sentiment_source"] = sentiment_data.get("source", "none")

    return stock


def detect_setup_type(stock: Dict[str, Any]) -> str:
    """Classify the type of setup based on indicator patterns."""
    ema_cross = stock.get("ema_cross")
    rsi = stock.get("rsi_14")
    adx = stock.get("adx")
    macd = stock.get("macd_signal")
    dist_high = stock.get("distance_52w_high")
    bb_width = stock.get("bb_width")

    if ema_cross == "golden_cross" and macd == "bullish":
        return "Golden Cross + MACD"

    if rsi is not None and rsi < 35 and macd == "bullish":
        return "Oversold Reversal"

    if dist_high is not None and dist_high > -3 and adx and adx > 25:
        return "Breakout (near 52w high)"

    if bb_width is not None and bb_width < 0.05 and adx and adx < 20:
        return "Squeeze (low BB width)"

    if ema_cross == "bullish" and adx and adx > 25:
        return "Trend Continuation"

    if rsi is not None and 45 <= rsi <= 55 and macd == "bullish":
        return "Pullback to Mean"

    return "Mixed Signal"


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: List[Dict[str, Any]],
    regime: Optional[Dict[str, Any]] = None,
    fetch_extras: bool = True,
    ml_predictions_map: Optional[Dict[str, Dict[str, Any]]] = None,
    sentiment_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Score a list of candidates and return sorted by confidence.

    Parameters
    ----------
    candidates : Output from ``scanner.apply_filters()``.
    regime : Market regime dict from ``events.assess_market_regime()``.
    fetch_extras : If True, fetch fundamentals/insider per ticker
                   (slower, but more accurate). False for fast scan.
    ml_predictions_map : Dict of ``{ticker: predictions_dict}`` from
                         ``ModelManager``.  If provided and ``ENABLE_ML``
                         is True, ML scores are included in the composite.
    sentiment_map : Dict of ``{ticker: sentiment_dict}`` from
                    ``sentiment.batch_sentiment()``.  If provided,
                    real news sentiment is used for scoring.

    Returns
    -------
    List of scored candidates sorted by ``confidence`` descending.
    Only candidates above ``MIN_CONFIDENCE`` are included.
    """
    scored: List[Dict[str, Any]] = []
    
    macro_bias = get_macro_bias() if fetch_extras else None

    for i, c in enumerate(candidates):
        ticker = c.get("ticker", "???")
        logger.debug(f"Scoring {ticker} ({i + 1}/{len(candidates)})")

        fundamentals = None
        if fetch_extras:
            try:
                from utils.data_loader import fetch_fundamentals as ff
                fundamentals = ff(ticker)
            except Exception as e:
                logger.debug(f"Fundamentals skipped for {ticker}: {e}")

        # Per-ticker ML predictions
        ml_preds = None
        if ml_predictions_map and ticker in ml_predictions_map:
            ml_preds = ml_predictions_map[ticker]

        # Per-ticker sentiment
        sent_data = None
        if sentiment_map and ticker in sentiment_map:
            sent_data = sentiment_map[ticker]

        scored_candidate = score_candidate(
            c,
            fundamentals=fundamentals,
            regime=regime,
            skip_api_calls=not fetch_extras,
            ml_predictions=ml_preds,
            sentiment_data=sent_data,
            macro_bias=macro_bias,
        )
        scored.append(scored_candidate)

    # Sort by confidence
    scored.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # Filter below minimum
    above = [s for s in scored if s.get("confidence", 0) >= MIN_CONFIDENCE]

    logger.info(
        f"Scored {len(candidates)} candidates: "
        f"{len(above)} above {MIN_CONFIDENCE} confidence"
    )

    return above


# ---------------------------------------------------------------------------
# Opportunity builder
# ---------------------------------------------------------------------------

def build_opportunity(
    stock: Dict[str, Any],
    portfolio_value: float,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an opportunity record ready for persistence.

    Uses ATR for stop-loss and target calculation.
    """
    ticker = stock.get("ticker", "???")
    close = stock.get("close", 0)
    atr_pct = stock.get("atr_pct", 2.0) or 2.0
    confidence = stock.get("confidence", 0)

    # Stop loss: 1.5 × ATR below entry
    atr_value = close * (atr_pct / 100)
    stop_loss = round(close - (1.5 * atr_value), 2)

    # Target: 3 × ATR above entry (2:1 risk/reward)
    target = round(close + (3.0 * atr_value), 2)

    # Entry zone: close ± 0.5% for limit order range
    entry_low = round(close * 0.995, 2)
    entry_high = round(close * 1.005, 2)

    # Position sizing (basic — portfolio.py does the real calculation)
    from utils.position_sizing import full_position_plan

    size_mod = 1.0
    if regime:
        size_mod = regime.get("position_size_modifier", 1.0)

    risk_pct = 0.02 * size_mod
    plan = full_position_plan(portfolio_value, close, stop_loss, target, risk_pct)

    position_value = plan.position_value if plan and plan.is_valid else 0
    risk_usd = plan.risk_amount if plan and plan.is_valid else 0
    reward_usd = plan.reward_amount if plan and plan.is_valid else 0
    shares = plan.shares if plan and plan.is_valid else 0
    rr_ratio = plan.risk_reward_ratio if plan and plan.is_valid else 0

    reasons = stock.get("filter_reasons", [])
    setup = stock.get("setup_type", "Mixed Signal")

    return {
        "ticker": ticker,
        "score_date": date.today().isoformat(),
        "confidence": confidence,
        "setup_type": setup,
        "entry_price_low": entry_low,
        "entry_price_high": entry_high,
        "stop_loss": stop_loss,
        "target_price": target,
        "position_size_usd": round(position_value, 2),
        "shares": shares,
        "risk_usd": round(risk_usd, 2),
        "reward_usd": round(reward_usd, 2),
        "risk_reward_ratio": rr_ratio,
        "reasons": reasons[:5],  # top 5 reasons
        "sub_scores": stock.get("sub_scores", {}),
        "status": "pending",
    }
