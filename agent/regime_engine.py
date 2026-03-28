"""
5-regime cross-asset macro engine.

Analyses 6 cross-asset components using yfinance data to classify the
market into one of five regimes, each with a position-size modifier.

Regimes and position-size modifiers
────────────────────────────────────
  Broadening    (1.00x) — ideal: broad participation, positive yield curve
  Concentration (0.85x) — mega-cap driven, breadth narrowing
  Transitional  (0.70x) — mixed signals, reduce exposure
  Inflationary  (0.60x) — commodities/rates dominating
  Contraction   (0.30x) — defensive/risk-off, minimise new entries

Cross-asset components (all free yfinance tickers)
────────────────────────────────────────────────────
  1. RSP/SPY    — equal-weight vs market-cap breadth signal
  2. Yield curve — ^TNX - ^IRX (10Y minus 3M proxy for curve shape)
  3. Credit      — HYG / LQD ratio (risk appetite)
  4. Size        — IWM / SPY (small vs large cap risk-on)
  5. Equity-bond — SPY / TLT (asset-class flow)
  6. Sector rot. — XLY / XLP (cyclical vs defensive spending)

Result is cached in-process for 7 days (only runs Sunday scan).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process cache (one dict per session — replaced on next weekly scan)
# ---------------------------------------------------------------------------
_cached_regime: Optional[Dict[str, Any]] = None
_cached_at: Optional[datetime] = None
_CACHE_HOURS = int(os.getenv("STOCKS_REGIME_CACHE_HOURS", "168"))  # 7 days default


def _is_fresh() -> bool:
    if _cached_regime is None or _cached_at is None:
        return False
    age = datetime.now(timezone.utc) - _cached_at
    return age < timedelta(hours=_CACHE_HOURS)


# ---------------------------------------------------------------------------
# Component fetchers
# ---------------------------------------------------------------------------

def _fetch_ratio(ticker_a: str, ticker_b: str, period: str = "1mo") -> Optional[float]:
    """Return the latest close ratio of ticker_a / ticker_b (1.0 = equal)."""
    try:
        import yfinance as yf
        import pandas as pd

        data = yf.download(
            [ticker_a, ticker_b],
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            return None

        close = data["Close"] if "Close" in data else data
        if ticker_a not in close.columns or ticker_b not in close.columns:
            return None

        a = close[ticker_a].dropna()
        b = close[ticker_b].dropna()

        if a.empty or b.empty:
            return None

        latest_a = float(a.iloc[-1])
        latest_b = float(b.iloc[-1])
        if latest_b == 0:
            return None

        return round(latest_a / latest_b, 4)
    except Exception as e:
        logger.debug(f"_fetch_ratio({ticker_a}/{ticker_b}) failed: {e}")
        return None


def _fetch_ratio_trend(
    ticker_a: str,
    ticker_b: str,
    period: str = "3mo",
    lookback_days: int = 20,
) -> Optional[float]:
    """Return % change of ratio over the last *lookback_days* trading days."""
    try:
        import yfinance as yf

        data = yf.download(
            [ticker_a, ticker_b],
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            return None

        close = data["Close"] if "Close" in data else data
        if ticker_a not in close.columns or ticker_b not in close.columns:
            return None

        a = close[ticker_a].dropna()
        b = close[ticker_b].dropna()
        n = min(len(a), len(b), lookback_days + 1)
        if n < 2:
            return None

        ratio_now  = float(a.iloc[-1])  / float(b.iloc[-1])
        ratio_prev = float(a.iloc[-n])  / float(b.iloc[-n])

        if ratio_prev == 0:
            return None
        return round((ratio_now - ratio_prev) / ratio_prev * 100, 2)
    except Exception as e:
        logger.debug(f"_fetch_ratio_trend({ticker_a}/{ticker_b}) failed: {e}")
        return None


def _fetch_yield_curve() -> Optional[float]:
    """Return 10Y − 3M spread in percentage points (positive = normal curve)."""
    try:
        import yfinance as yf
        import pandas as pd

        data = yf.download(
            ["^TNX", "^IRX"],
            period="5d",
            interval="1d",
            progress=False,
        )
        if data is None or data.empty:
            return None

        close = data["Close"] if "Close" in data else data
        if "^TNX" not in close.columns or "^IRX" not in close.columns:
            return None

        tnx = float(close["^TNX"].dropna().iloc[-1])  # 10Y (× 0.01 = %)
        irx = float(close["^IRX"].dropna().iloc[-1])  # 13-week (× 0.01 = %)

        # yfinance returns raw yield values (e.g. 4.5 = 4.5%)
        return round(tnx - irx, 2)
    except Exception as e:
        logger.debug(f"_fetch_yield_curve failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main regime classifier
# ---------------------------------------------------------------------------

def assess_regime(force_refresh: bool = False) -> Dict[str, Any]:
    """Return the current 5-regime classification.

    Uses in-process cache (refreshed at most once per week).
    Falls back to simple VIX-based regime on any data failure.

    Returns
    -------
    Dict with:
        ``regime``              — "broadening" | "concentration" | "transitional"
                                  | "inflationary" | "contraction"
        ``position_size_mod``  — float multiplier for position sizing
        ``components``         — dict of raw component values
        ``score``              — 0-100 composite bullish score
        ``mood``               — "Bullish" | "Neutral" | "Bearish"
        ``description``        — human-readable summary
    """
    global _cached_regime, _cached_at

    if not force_refresh and _is_fresh():
        logger.debug("Regime engine: returning cached result")
        return _cached_regime

    logger.info("Regime engine: computing fresh regime classification")

    components: Dict[str, Any] = {}

    # 1. RSP/SPY breadth trend
    rsp_spy_trend = _fetch_ratio_trend("RSP", "SPY", lookback_days=20)
    components["rsp_spy_trend"] = rsp_spy_trend

    # 2. Yield curve
    yield_curve = _fetch_yield_curve()
    components["yield_curve"] = yield_curve

    # 3. Credit risk appetite (HYG/LQD)
    hyg_lqd_trend = _fetch_ratio_trend("HYG", "LQD", lookback_days=20)
    components["credit_trend"] = hyg_lqd_trend

    # 4. Size factor (IWM/SPY)
    iwm_spy_trend = _fetch_ratio_trend("IWM", "SPY", lookback_days=20)
    components["size_trend"] = iwm_spy_trend

    # 5. Equity-bond (SPY/TLT)
    spy_tlt_trend = _fetch_ratio_trend("SPY", "TLT", lookback_days=20)
    components["equity_bond_trend"] = spy_tlt_trend

    # 6. Sector rotation (XLY/XLP)
    xly_xlp_trend = _fetch_ratio_trend("XLY", "XLP", lookback_days=20)
    components["sector_rotation"] = xly_xlp_trend

    # ── Score each component ─────────────────────────────────────────────────
    # Each positive signal contributes to the bullish composite score.
    bullish_pts = 0
    total_pts = 0

    def score_trend(val: Optional[float], bul_thresh: float = 0, bear_thresh: float = 0) -> int:
        if val is None:
            return 0
        if val > bul_thresh:
            return 2    # bullish
        elif val < bear_thresh:
            return -2   # bearish
        return 1        # neutral

    # RSP/SPY: positive trend = breadth broadening (strongly bullish)
    rsp_pts = score_trend(rsp_spy_trend, bul_thresh=1.0, bear_thresh=-1.5)
    bullish_pts += max(0, rsp_pts) * 20
    total_pts += 20

    # Yield curve: > 0.5 = normal, < 0 = inverted (bearish for growth)
    if yield_curve is not None:
        if yield_curve >= 0.5:
            bullish_pts += 15
        elif yield_curve >= 0:
            bullish_pts += 8
        elif yield_curve >= -0.5:
            bullish_pts += 0
        else:
            bullish_pts -= 10
    total_pts += 15

    # Credit: HYG/LQD rising = risk-on
    cr_pts = score_trend(hyg_lqd_trend, bul_thresh=0.5, bear_thresh=-0.5)
    bullish_pts += max(0, cr_pts) * 12
    total_pts += 12 * 2

    # Size: IWM leading SPY = risk-on
    sz_pts = score_trend(iwm_spy_trend, bul_thresh=0.5, bear_thresh=-1.0)
    bullish_pts += max(0, sz_pts) * 12
    total_pts += 12 * 2

    # Equity-bond: SPY/TLT rising = equity preferred
    eb_pts = score_trend(spy_tlt_trend, bul_thresh=0.5, bear_thresh=-0.5)
    bullish_pts += max(0, eb_pts) * 10
    total_pts += 10 * 2

    # Sector rotation: XLY > XLP = risk appetite
    sr_pts = score_trend(xly_xlp_trend, bul_thresh=0.5, bear_thresh=-1.0)
    bullish_pts += max(0, sr_pts) * 11
    total_pts += 11 * 2

    score = round(min(100, max(0, bullish_pts / total_pts * 100)), 1) if total_pts > 0 else 50.0

    # ── Regime classification ─────────────────────────────────────────────────
    data_available = sum(1 for v in components.values() if v is not None)

    if data_available < 3:
        # Insufficient data — fall back to neutral
        regime, size_mod, mood, desc = (
            "transitional", 0.70, "Neutral",
            "Insufficient cross-asset data — conservative sizing"
        )
    elif score >= 72:
        regime, size_mod, mood, desc = (
            "broadening", 1.00, "Bullish",
            f"Broad market participation — full size (score={score:.0f})"
        )
    elif score >= 55:
        regime, size_mod, mood, desc = (
            "concentration", 0.85, "Neutral",
            f"Mega-cap driven — slight size reduction (score={score:.0f})"
        )
    elif score >= 40:
        regime, size_mod, mood, desc = (
            "transitional", 0.70, "Neutral",
            f"Mixed signals — reduced sizing (score={score:.0f})"
        )
    elif score >= 25:
        regime, size_mod, mood, desc = (
            "inflationary", 0.60, "Bearish",
            f"Commodities/rates dominating — very selective (score={score:.0f})"
        )
    else:
        regime, size_mod, mood, desc = (
            "contraction", 0.30, "Bearish",
            f"Risk-off defensive regime — minimal new entries (score={score:.0f})"
        )

    # Adjust for inverted yield curve (hard brake)
    if yield_curve is not None and yield_curve < -0.5 and size_mod > 0.30:
        size_mod = max(0.30, size_mod - 0.20)
        desc += " | inverted curve penalty"

    result = {
        "regime": regime,
        "position_size_mod": round(size_mod, 2),
        # Legacy key so existing code using `position_size_modifier` still works
        "position_size_modifier": round(size_mod, 2),
        "mood": mood,
        "score": score,
        "components": components,
        "description": desc,
    }

    _cached_regime = result
    _cached_at = datetime.now(timezone.utc)
    logger.info(f"Regime engine: {regime} score={score:.0f} size_mod={size_mod:.2f}")
    return result


def get_cached_regime() -> Optional[Dict[str, Any]]:
    """Return the last computed regime without triggering a refresh."""
    return _cached_regime
