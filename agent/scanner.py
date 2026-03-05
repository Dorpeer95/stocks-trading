"""
Universe scanner — filters S&P 500 to find top swing trade candidates.

Runs during the weekly scan and produces a shortlist of stocks that
pass all technical / fundamental / RS filters, with optional ML
pre-filtering via the direction model.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.data_loader import (
    fetch_batch_prices,
    fetch_fundamentals,
    fetch_price_data,
    fetch_sp500_list,
    get_sector_etf,
    SECTOR_ETF_MAP,
)
from utils.indicators import calc_all_indicators
from utils.sectors import compute_universe_rs, rank_sectors, get_hot_sectors
from agent.persistence import upsert_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter thresholds (can be overridden via env)
# ---------------------------------------------------------------------------
MIN_ADX = float(os.getenv("STOCKS_MIN_ADX", "20"))
MIN_RS_PERCENTILE = float(os.getenv("STOCKS_MIN_RS_PERCENTILE", "50"))
MAX_ATR_PCT = float(os.getenv("STOCKS_MAX_ATR_PCT", "8"))
MIN_VOLUME_RATIO = float(os.getenv("STOCKS_MIN_VOLUME_RATIO", "0.5"))
TOP_CANDIDATES_LIMIT = int(os.getenv("STOCKS_TOP_CANDIDATES", "30"))

# ML filter settings
ENABLE_ML_FILTER = os.getenv("STOCKS_ENABLE_ML_FILTER", "true").lower() == "true"
ML_DIRECTION_THRESHOLD = float(os.getenv("STOCKS_ML_DIRECTION_THRESHOLD", "0.45"))

# Batch download chunk size (stay under yfinance rate limits)
BATCH_SIZE = 20
BATCH_DELAY = 5.0  # seconds between batches


# ---------------------------------------------------------------------------
# Full universe scan
# ---------------------------------------------------------------------------

def scan_universe() -> Dict[str, Any]:
    """Run the full weekly universe scan.

    Steps
    -----
    1. Fetch S&P 500 list from Wikipedia.
    2. Download 1-year price data in batches.
    3. Compute technical indicators for each stock.
    4. Compute relative strength vs SPY and sector ETFs.
    5. Apply filters to find candidates.
    6. Rank and return the top N.

    Returns
    -------
    Dict with keys:
    - ``candidates``: list of scored candidate dicts
    - ``sector_rankings``: sector rank list
    - ``hot_sectors``: top 3 sectors
    - ``universe_size``: total stocks scanned
    - ``passed_filter``: stocks that passed all filters
    - ``scan_time_s``: scan duration in seconds
    """
    start = time.time()
    logger.info("=" * 50)
    logger.info("  WEEKLY UNIVERSE SCAN STARTING")
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # 1. Get S&P 500 constituents
    # ------------------------------------------------------------------
    sp500 = fetch_sp500_list()
    if not sp500:
        logger.error("Failed to fetch S&P 500 list")
        return _empty_scan_result(time.time() - start)

    tickers = [s["ticker"] for s in sp500]
    ticker_sectors = {s["ticker"]: s["sector"] for s in sp500}
    logger.info(f"Universe: {len(tickers)} S&P 500 stocks")

    # Persist universe
    try:
        upsert_universe(sp500)
    except Exception as e:
        logger.warning(f"Failed to upsert universe: {e}")

    # ------------------------------------------------------------------
    # 2. Download price data in batches
    # ------------------------------------------------------------------
    all_price_data: Dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        logger.info(
            f"Downloading batch {i // BATCH_SIZE + 1}/"
            f"{(len(tickers) - 1) // BATCH_SIZE + 1} "
            f"({len(batch)} tickers)"
        )

        batch_data = fetch_batch_prices(batch, period="1y")
        if batch_data:
            all_price_data.update(batch_data)

        # Rate-limit delay between batches
        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    logger.info(
        f"Downloaded price data for {len(all_price_data)}/{len(tickers)} tickers"
    )

    if not all_price_data:
        logger.error("No price data downloaded")
        return _empty_scan_result(time.time() - start)

    # ------------------------------------------------------------------
    # 3. Fetch SPY and sector ETF data for RS
    # ------------------------------------------------------------------
    spy_df = fetch_price_data("SPY", period="1y")
    if spy_df is None:
        logger.error("Failed to fetch SPY data for RS")
        return _empty_scan_result(time.time() - start)

    # Get unique sector ETFs
    sector_etf_tickers = list(set(SECTOR_ETF_MAP.values()))
    sector_etf_data = fetch_batch_prices(sector_etf_tickers, period="1y")
    if sector_etf_data is None:
        sector_etf_data = {}

    # ------------------------------------------------------------------
    # 4. Compute indicators + RS for each stock
    # ------------------------------------------------------------------
    stock_data: List[Dict[str, Any]] = []

    for ticker, df in all_price_data.items():
        try:
            indicators = calc_all_indicators(df)
            if indicators is None:
                continue

            stock_entry = {
                "ticker": ticker,
                "sector": ticker_sectors.get(ticker, "Unknown"),
                **indicators,
            }
            stock_data.append(stock_entry)

        except Exception as e:
            logger.warning(f"Failed to compute indicators for {ticker}: {e}")

    logger.info(f"Computed indicators for {len(stock_data)} stocks")

    # Compute relative strength
    rs_data = compute_universe_rs(
        all_price_data,
        spy_df,
        sector_etf_data,
        ticker_sectors,
    )

    # Merge RS into stock_data
    rs_by_ticker = {r["ticker"]: r for r in rs_data}
    for stock in stock_data:
        rs = rs_by_ticker.get(stock["ticker"], {})
        stock["rs_vs_spy"] = rs.get("rs_vs_spy")
        stock["rs_vs_sector"] = rs.get("rs_vs_sector")
        stock["rs_percentile"] = rs.get("rs_percentile", 0)
        stock["momentum_4w"] = rs.get("momentum_4w")
        stock["momentum_13w"] = rs.get("momentum_13w")

    # Sector rankings
    sector_rankings = rank_sectors(rs_data, ticker_sectors)
    hot = get_hot_sectors(sector_rankings)

    # ------------------------------------------------------------------
    # 5. Apply filters
    # ------------------------------------------------------------------
    candidates = apply_filters(stock_data)
    logger.info(
        f"Filters passed: {len(candidates)}/{len(stock_data)} stocks"
    )

    # ------------------------------------------------------------------
    # 6. Rank and limit
    # ------------------------------------------------------------------
    candidates = rank_candidates(candidates)
    top = candidates[:TOP_CANDIDATES_LIMIT]

    elapsed = time.time() - start
    logger.info(f"Scan complete in {elapsed:.1f}s — {len(top)} top candidates")

    return {
        "candidates": top,
        "sector_rankings": sector_rankings,
        "hot_sectors": hot,
        "universe_size": len(tickers),
        "scanned": len(stock_data),
        "passed_filter": len(candidates),
        "scan_time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def apply_filters(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply technical/fundamental filters to the scanned universe.

    A stock passes if:
    - ADX > MIN_ADX (trending)
    - RS percentile > MIN_RS_PERCENTILE
    - ATR% < MAX_ATR_PCT (not too volatile)
    - Volume ratio > MIN_VOLUME_RATIO (liquid)
    - EMA cross is NOT death_cross
    - RSI is not overbought (< 80)
    """
    passed: List[Dict[str, Any]] = []

    for s in stocks:
        reasons: List[str] = []

        # Must have key indicators
        adx = s.get("adx")
        rs_pct = s.get("rs_percentile", 0)
        atr_pct = s.get("atr_pct")
        vol_ratio = s.get("volume_ratio")
        ema_cross = s.get("ema_cross")
        rsi = s.get("rsi_14")

        if adx is None or atr_pct is None:
            continue

        # Trending
        if adx < MIN_ADX:
            continue
        reasons.append(f"ADX={adx:.0f}")

        # Relative strength
        if rs_pct < MIN_RS_PERCENTILE:
            continue
        reasons.append(f"RS={rs_pct:.0f}%ile")

        # Volatility cap
        if atr_pct > MAX_ATR_PCT:
            continue

        # Volume
        if vol_ratio is not None and vol_ratio < MIN_VOLUME_RATIO:
            continue

        # Not in death cross
        if ema_cross == "death_cross":
            continue

        # Not overbought
        if rsi is not None and rsi > 80:
            continue

        # Bullish signals (additive reasons)
        if ema_cross == "golden_cross":
            reasons.append("Golden Cross")
        elif ema_cross == "bullish":
            reasons.append("EMA bullish")

        macd = s.get("macd_signal")
        if macd == "bullish":
            reasons.append("MACD bullish")

        if rsi and 40 <= rsi <= 60:
            reasons.append("RSI neutral zone")

        if vol_ratio and vol_ratio > 1.5:
            reasons.append(f"High volume ({vol_ratio:.1f}x)")

        s["filter_reasons"] = reasons
        passed.append(s)

    return passed


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank filtered candidates by a composite sort score.

    Composite = weighted sum of RS percentile, ADX, and momentum.
    """
    for c in candidates:
        rs_pct = c.get("rs_percentile", 0)
        adx = c.get("adx", 0) or 0
        m4w = c.get("momentum_4w", 0) or 0

        # Normalise ADX to 0-100 (it's already roughly that scale)
        # RS percentile is 0-100
        # Momentum can be anything, cap at ±50
        m4w_capped = max(-50, min(50, m4w))
        m4w_norm = (m4w_capped + 50) / 100 * 100  # 0-100 scale

        composite = (rs_pct * 0.5) + (adx * 0.3) + (m4w_norm * 0.2)
        c["composite_rank_score"] = round(composite, 2)

    candidates.sort(key=lambda x: x.get("composite_rank_score", 0), reverse=True)

    for i, c in enumerate(candidates, 1):
        c["rank"] = i

    return candidates


# ---------------------------------------------------------------------------
# ML pre-filter
# ---------------------------------------------------------------------------

def filter_by_ml_direction(
    candidates: List[Dict[str, Any]],
    model_manager: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Pre-filter candidates using the ML direction model.

    Removes candidates where the direction model predicts a strong
    bearish signal (probability of positive direction < threshold).

    Parameters
    ----------
    candidates : Filtered candidates from ``apply_filters()``.
    model_manager : A loaded ``ModelManager`` instance.

    Returns
    -------
    ``(filtered_candidates, ml_predictions_map)``
    - ``filtered_candidates`` : candidates that passed ML filter
    - ``ml_predictions_map``  : ``{ticker: predictions_dict}`` for all
      candidates (including those filtered out), to be passed to
      ``scorer.score_candidates()``
    """
    if not ENABLE_ML_FILTER or model_manager is None:
        return candidates, {}

    if not model_manager.models.get("direction"):
        logger.info("Direction model not loaded — skipping ML filter")
        return candidates, {}

    logger.info(f"Running ML direction filter on {len(candidates)} candidates")

    ml_predictions_map: Dict[str, Dict[str, Any]] = {}
    passed: List[Dict[str, Any]] = []

    for c in candidates:
        ticker = c.get("ticker", "???")

        try:
            # Direction prediction
            direction = model_manager.predict_direction(c)
            volatility = model_manager.predict_volatility(c)

            preds: Dict[str, Any] = {}
            if direction:
                preds["direction"] = direction
            if volatility:
                preds["volatility"] = volatility

            ml_predictions_map[ticker] = preds

            # Filter: reject if direction model is confidently bearish
            if direction and direction.get("signal") == "bearish":
                dir_prob = direction.get("probability", 0.5)
                # Only reject if model is confident about bearish
                if dir_prob < ML_DIRECTION_THRESHOLD:
                    logger.debug(
                        f"ML filter rejected {ticker} — "
                        f"bearish confidence {dir_prob:.2f}"
                    )
                    c["ml_filtered"] = True
                    continue

            passed.append(c)

        except Exception as e:
            logger.debug(f"ML filter skipped for {ticker}: {e}")
            passed.append(c)  # keep on error

    rejected = len(candidates) - len(passed)
    logger.info(
        f"ML filter: {len(passed)} passed, {rejected} rejected"
    )

    return passed, ml_predictions_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_scan_result(elapsed: float) -> Dict[str, Any]:
    return {
        "candidates": [],
        "sector_rankings": [],
        "hot_sectors": [],
        "universe_size": 0,
        "scanned": 0,
        "passed_filter": 0,
        "scan_time_s": round(elapsed, 1),
    }
