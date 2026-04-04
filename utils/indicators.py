"""
Technical indicators for stock analysis.

All functions are pure — they take a pandas DataFrame with OHLCV columns
and return computed indicator values. No API calls, no side effects.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (yfinance uses these)
# ---------------------------------------------------------------------------
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOLUME = "Volume"


def _validate_ohlcv(df: Optional[pd.DataFrame], min_rows: int = 2) -> bool:
    """Check that *df* is a non-empty OHLCV DataFrame with enough rows."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    required = {CLOSE}
    if not required.issubset(df.columns):
        return False
    if len(df) < min_rows:
        return False
    return True


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def calc_rsi(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """Relative Strength Index (0-100).

    Parameters
    ----------
    df : DataFrame with at least a ``Close`` column.
    period : Look-back window (default 14).

    Returns
    -------
    Series of RSI values, or ``None`` on invalid input.
    """
    if not _validate_ohlcv(df, min_rows=period + 1):
        logger.warning("calc_rsi: invalid input")
        return None
    try:
        return ta.momentum.RSIIndicator(
            close=df[CLOSE], window=period
        ).rsi()
    except Exception as e:
        logger.error(f"calc_rsi failed: {e}")
        return None


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[pd.DataFrame]:
    """MACD line, signal line, and histogram.

    Returns
    -------
    DataFrame with columns ``macd``, ``signal``, ``histogram``,
    or ``None`` on invalid input.
    """
    if not _validate_ohlcv(df, min_rows=slow + signal):
        logger.warning("calc_macd: invalid input")
        return None
    try:
        macd_ind = ta.trend.MACD(
            close=df[CLOSE],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal,
        )
        return pd.DataFrame({
            "macd": macd_ind.macd(),
            "signal": macd_ind.macd_signal(),
            "histogram": macd_ind.macd_diff(),
        })
    except Exception as e:
        logger.error(f"calc_macd failed: {e}")
        return None


def calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """Average True Range.

    Parameters
    ----------
    df : DataFrame with ``High``, ``Low``, ``Close`` columns.
    period : Look-back window (default 14).
    """
    if not _validate_ohlcv(df, min_rows=period + 1):
        logger.warning("calc_atr: invalid input")
        return None
    try:
        for col in (HIGH, LOW, CLOSE):
            if col not in df.columns:
                logger.warning(f"calc_atr: missing column {col}")
                return None
        return ta.volatility.AverageTrueRange(
            high=df[HIGH], low=df[LOW], close=df[CLOSE], window=period
        ).average_true_range()
    except Exception as e:
        logger.error(f"calc_atr failed: {e}")
        return None


def calc_adx(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """Average Directional Index (0-100).

    Measures trend strength regardless of direction.
    """
    if not _validate_ohlcv(df, min_rows=period * 2):
        logger.warning("calc_adx: invalid input")
        return None
    try:
        for col in (HIGH, LOW, CLOSE):
            if col not in df.columns:
                logger.warning(f"calc_adx: missing column {col}")
                return None
        return ta.trend.ADXIndicator(
            high=df[HIGH], low=df[LOW], close=df[CLOSE], window=period
        ).adx()
    except Exception as e:
        logger.error(f"calc_adx failed: {e}")
        return None


def calc_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: int = 2,
) -> Optional[pd.DataFrame]:
    """Bollinger Bands: upper, middle, lower, and bandwidth.

    Returns
    -------
    DataFrame with columns ``upper``, ``middle``, ``lower``, ``width``
    where ``width = (upper - lower) / middle``.
    """
    if not _validate_ohlcv(df, min_rows=period):
        logger.warning("calc_bollinger: invalid input")
        return None
    try:
        bb = ta.volatility.BollingerBands(
            close=df[CLOSE], window=period, window_dev=std
        )
        upper = bb.bollinger_hband()
        middle = bb.bollinger_mavg()
        lower = bb.bollinger_lband()
        width = bb.bollinger_wband()
        return pd.DataFrame({
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "width": width,
        })
    except Exception as e:
        logger.error(f"calc_bollinger failed: {e}")
        return None


def calc_ema(df: pd.DataFrame, period: int = 20) -> Optional[pd.Series]:
    """Exponential Moving Average.

    Parameters
    ----------
    period : Common values are 9, 20, 50, 200.
    """
    if not _validate_ohlcv(df, min_rows=period):
        logger.warning("calc_ema: invalid input")
        return None
    try:
        return ta.trend.EMAIndicator(
            close=df[CLOSE], window=period
        ).ema_indicator()
    except Exception as e:
        logger.error(f"calc_ema failed: {e}")
        return None


def calc_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    """Volume-Weighted Average Price.

    Requires ``High``, ``Low``, ``Close``, ``Volume`` columns.
    Uses cumulative VWAP over the full DataFrame.
    """
    required = {HIGH, LOW, CLOSE, VOLUME}
    if not _validate_ohlcv(df) or not required.issubset(df.columns):
        logger.warning("calc_vwap: invalid input")
        return None
    try:
        typical_price = (df[HIGH] + df[LOW] + df[CLOSE]) / 3.0
        cum_vol = df[VOLUME].cumsum()
        cum_tp_vol = (typical_price * df[VOLUME]).cumsum()
        vwap = cum_tp_vol / cum_vol
        vwap = vwap.replace([np.inf, -np.inf], np.nan)
        return vwap
    except Exception as e:
        logger.error(f"calc_vwap failed: {e}")
        return None


def calc_obv(df: pd.DataFrame) -> Optional[pd.Series]:
    """On-Balance Volume.

    Cumulative volume adjusted by price direction.
    """
    if not _validate_ohlcv(df) or VOLUME not in df.columns:
        logger.warning("calc_obv: invalid input")
        return None
    try:
        return ta.volume.OnBalanceVolumeIndicator(
            close=df[CLOSE], volume=df[VOLUME]
        ).on_balance_volume()
    except Exception as e:
        logger.error(f"calc_obv failed: {e}")
        return None


def calc_relative_volume(df: pd.DataFrame, period: int = 50) -> Optional[float]:
    """Relative volume: latest volume / average volume over *period* days.

    Returns
    -------
    float ratio (e.g. 1.5 means 50% above average), or ``None``.
    """
    if not _validate_ohlcv(df, min_rows=period) or VOLUME not in df.columns:
        logger.warning("calc_relative_volume: invalid input")
        return None
    try:
        avg_vol = df[VOLUME].iloc[-period:].mean()
        if avg_vol == 0 or np.isnan(avg_vol):
            return None
        latest_vol = df[VOLUME].iloc[-1]
        return round(float(latest_vol / avg_vol), 2)
    except Exception as e:
        logger.error(f"calc_relative_volume failed: {e}")
        return None


def calc_distance_52w(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Distance from 52-week (252 trading days) high and low.

    Returns
    -------
    dict with ``high_pct`` (negative = below high) and
    ``low_pct`` (positive = above low), or ``None``.
    """
    if not _validate_ohlcv(df, min_rows=20):
        logger.warning("calc_distance_52w: invalid input")
        return None
    try:
        lookback = min(252, len(df))
        window = df.iloc[-lookback:]
        high_52w = window[HIGH].max() if HIGH in df.columns else window[CLOSE].max()
        low_52w = window[LOW].min() if LOW in df.columns else window[CLOSE].min()
        current = df[CLOSE].iloc[-1]

        if high_52w == 0 or low_52w == 0:
            return None

        high_pct = round(((current - high_52w) / high_52w) * 100, 2)
        low_pct = round(((current - low_52w) / low_52w) * 100, 2)
        return {"high_pct": high_pct, "low_pct": low_pct}
    except Exception as e:
        logger.error(f"calc_distance_52w failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Golden / Death Cross detection
# ---------------------------------------------------------------------------

def detect_ema_cross(
    df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200,
) -> Optional[str]:
    """Detect golden cross or death cross using EMAs.

    Returns
    -------
    ``'golden_cross'`` if fast EMA just crossed above slow EMA,
    ``'death_cross'``  if fast EMA just crossed below slow EMA,
    ``'bullish'``      if fast > slow (no recent cross),
    ``'bearish'``      if fast < slow (no recent cross),
    or ``None`` on invalid input.
    """
    ema_fast = calc_ema(df, fast_period)
    ema_slow = calc_ema(df, slow_period)
    if ema_fast is None or ema_slow is None:
        return None
    try:
        fast_now = ema_fast.iloc[-1]
        slow_now = ema_slow.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_prev = ema_slow.iloc[-2]

        if fast_prev <= slow_prev and fast_now > slow_now:
            return "golden_cross"
        elif fast_prev >= slow_prev and fast_now < slow_now:
            return "death_cross"
        elif fast_now > slow_now:
            return "bullish"
        else:
            return "bearish"
    except Exception as e:
        logger.error(f"detect_ema_cross failed: {e}")
        return None


# ---------------------------------------------------------------------------
# MACD signal helper
# ---------------------------------------------------------------------------

def macd_signal_direction(df: pd.DataFrame) -> Optional[str]:
    """Return the current MACD signal direction.

    Returns
    -------
    ``'bullish'``  if MACD histogram is positive and rising,
    ``'bearish'``  if histogram is negative and falling,
    ``'neutral'``  otherwise, or ``None`` on failure.
    """
    macd_df = calc_macd(df)
    if macd_df is None:
        return None
    try:
        hist = macd_df["histogram"].dropna()
        if len(hist) < 2:
            return "neutral"
        current = hist.iloc[-1]
        previous = hist.iloc[-2]
        if current > 0 and current > previous:
            return "bullish"
        elif current < 0 and current < previous:
            return "bearish"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"macd_signal_direction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# ATR as percentage of price
# ---------------------------------------------------------------------------

def calc_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """ATR as a percentage of the current closing price.

    Useful for comparing volatility across different price levels.
    """
    atr = calc_atr(df, period)
    if atr is None:
        return None
    try:
        atr_val = atr.iloc[-1]
        price = df[CLOSE].iloc[-1]
        if price == 0 or np.isnan(price):
            return None
        return round(float(atr_val / price) * 100, 3)
    except Exception as e:
        logger.error(f"calc_atr_pct failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Volatility Contraction Pattern (VCP) Detection
# ---------------------------------------------------------------------------

def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Optional[Dict[str, Any]]:
    """Detect a high-quality Volatility Contraction Pattern (VCP).

    Requirements (Mark Minervini methodology):
    1. Proper uptrend: price > SMA50 > SMA150 > SMA200
    2. Within 25% of the 52-week high (only buy leaders)
    3. Three-stage contracting pullbacks — each stage tighter than the last
    4. Final stage (the current consolidation) must be < 12% range
    5. Volume drying up into the tight base

    Returns
    -------
    Dict with:
    - ``is_vcp``            : bool
    - ``pivot``             : exact breakout price to watch (high of tight base)
    - ``base_low``          : stop reference (low of tight base)
    - ``contraction_pct``   : tightness of the final base (%)  — lower is better
    - ``base_depth_pct``    : depth of the full correction (%)
    - ``reason``            : human-readable explanation
    or ``None`` if input is invalid.
    """
    if not _validate_ohlcv(df, min_rows=220):
        return None

    try:
        close = df[CLOSE].iloc[-1]
        sma50  = df[CLOSE].rolling(50).mean().iloc[-1]
        sma150 = df[CLOSE].rolling(150).mean().iloc[-1]
        sma200 = df[CLOSE].rolling(200).mean().iloc[-1]

        # ── 1. Proper uptrend ────────────────────────────────────────────────
        if not (close > sma50 > sma150 > sma200):
            return {"is_vcp": False, "reason": "Not in uptrend (need price > SMA50 > SMA150 > SMA200)"}

        # ── 2. Within 25% of 52-week high (leadership quality) ───────────────
        high_52w = df[HIGH].iloc[-252:].max() if HIGH in df.columns else df[CLOSE].iloc[-252:].max()
        dist_from_high_pct = (close - high_52w) / high_52w * 100  # negative
        if dist_from_high_pct < -25:
            return {"is_vcp": False, "reason": f"Too far from 52w high ({dist_from_high_pct:.1f}%)"}

        # ── 3. Three-stage contraction ───────────────────────────────────────
        recent = df.iloc[-lookback:]
        seg = lookback // 3
        s1 = recent.iloc[:seg]
        s2 = recent.iloc[seg : seg * 2]
        s3 = recent.iloc[seg * 2:]   # most recent = should be tightest

        def _range_pct(seg: pd.DataFrame) -> float:
            h = seg[HIGH].max() if HIGH in seg.columns else seg[CLOSE].max()
            lo = seg[LOW].min() if LOW in seg.columns else seg[CLOSE].min()
            return float((h - lo) / lo * 100) if lo > 0 else 999.0

        r1, r2, r3 = _range_pct(s1), _range_pct(s2), _range_pct(s3)

        if not (r1 > r2 > r3):
            return {"is_vcp": False, "reason": f"Contractions not sequential ({r1:.1f}% → {r2:.1f}% → {r3:.1f}%)"}

        # Final stage must be tight enough to be tradeable
        if r3 > 12.0:
            return {"is_vcp": False, "reason": f"Final base too loose ({r3:.1f}%, need < 12%)"}

        # ── 4. Volume drying up in the base ──────────────────────────────────
        vol1_avg = float(s1[VOLUME].mean()) if VOLUME in s1.columns else 1.0
        vol3_avg = float(s3[VOLUME].mean()) if VOLUME in s3.columns else 0.0
        if vol3_avg >= vol1_avg * 0.95:
            return {"is_vcp": False, "reason": "Volume not drying up in base"}

        # ── 5. Pivot and base-low ────────────────────────────────────────────
        pivot    = float(s3[HIGH].max() if HIGH in s3.columns else s3[CLOSE].max())
        base_low = float(s3[LOW].min()  if LOW  in s3.columns else s3[CLOSE].min())

        base_depth = _range_pct(recent)

        return {
            "is_vcp":          True,
            "pivot":           round(pivot, 2),
            "base_low":        round(base_low, 2),
            "contraction_pct": round(r3, 1),     # tightness of the current base
            "base_depth_pct":  round(base_depth, 1),
            "reason":          f"VCP — {r3:.1f}% tight base, pivot ${pivot:.2f}",
        }

    except Exception as e:
        logger.error(f"detect_vcp failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Composite: compute everything in one pass
# ---------------------------------------------------------------------------

def calc_all_indicators(df: pd.DataFrame) -> Optional[Dict[str, Union[float, str, None]]]:
    """Compute all indicators for a single stock DataFrame.

    Returns
    -------
    Dictionary with all indicator values, suitable for scoring and
    database storage. Returns ``None`` if input is invalid.

    Keys
    ----
    rsi_14, macd_signal, adx, atr_pct, bb_width, volume_ratio,
    distance_52w_high, distance_52w_low, ema_50, ema_200, ema_cross,
    obv_latest, vwap_latest, close
    """
    if not _validate_ohlcv(df, min_rows=50):
        logger.warning("calc_all_indicators: need at least 50 rows")
        return None

    try:
        rsi = calc_rsi(df)
        macd_dir = macd_signal_direction(df)
        adx = calc_adx(df)
        atr_pct = calc_atr_pct(df)
        bb = calc_bollinger(df)
        vol_ratio = calc_relative_volume(df)
        dist_52w = calc_distance_52w(df)
        ema_50 = calc_ema(df, 50)
        ema_200 = calc_ema(df, 200)
        ema_cross = detect_ema_cross(df)
        obv = calc_obv(df)
        vwap = calc_vwap(df)

        result: Dict[str, Union[float, str, None]] = {
            "close": round(float(df[CLOSE].iloc[-1]), 2),
            "rsi_14": round(float(rsi.iloc[-1]), 2) if rsi is not None else None,
            "macd_signal": macd_dir,
            "adx": round(float(adx.iloc[-1]), 2) if adx is not None else None,
            "atr_pct": atr_pct,
            "bb_width": round(float(bb["width"].iloc[-1]), 4) if bb is not None else None,
            "volume_ratio": vol_ratio,
            "distance_52w_high": dist_52w["high_pct"] if dist_52w else None,
            "distance_52w_low": dist_52w["low_pct"] if dist_52w else None,
            "ema_50": round(float(ema_50.iloc[-1]), 2) if ema_50 is not None else None,
            "ema_200": round(float(ema_200.iloc[-1]), 2) if ema_200 is not None else None,
            "ema_cross": ema_cross,
            "obv_latest": round(float(obv.iloc[-1]), 0) if obv is not None else None,
            "vwap_latest": round(float(vwap.iloc[-1]), 2) if vwap is not None else None,
        }
        
        # VCP — returns pivot price and base_low for entry/stop calculation
        vcp = detect_vcp(df)
        if vcp and vcp.get("is_vcp"):
            result["vcp_detected"]    = True
            result["vcp_pivot"]       = vcp.get("pivot")        # breakout trigger level
            result["vcp_base_low"]    = vcp.get("base_low")     # stop reference
            result["vcp_contraction"] = vcp.get("contraction_pct")  # tightness %
            result["vcp_base_depth"]  = vcp.get("base_depth_pct")
        else:
            result["vcp_detected"]    = False
            result["vcp_pivot"]       = None
            result["vcp_base_low"]    = None
            result["vcp_contraction"] = None
            result["vcp_base_depth"]  = None

        return result

    except Exception as e:
        logger.error(f"calc_all_indicators failed: {e}", exc_info=True)
        return None
