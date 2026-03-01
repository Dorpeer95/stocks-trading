"""
Feature configuration for ML models.

Defines feature sets for each model, builds feature vectors from
stock data dicts, and validates vectors before inference.

All 4 models:
  - direction:       P(stock up 5% in 10 trading days)
  - volatility:      P(stock moves > 2 ATR in 5 days)
  - earnings:        P(post-earnings move is positive)
  - sector_rotation: P(sector outperforms SPY next 4 weeks)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions per model
#
# Each list is the ORDERED set of feature names.  The training script
# and the inference code MUST use the same order.  Changing order or
# adding features requires a new model version.
# ---------------------------------------------------------------------------

DIRECTION_FEATURES: List[str] = [
    "rsi_14",
    "adx",
    "atr_pct",
    "bb_width",
    "volume_ratio",
    "distance_52w_high",
    "distance_52w_low",
    "ema_cross_encoded",        # -1 bearish, 0 neutral, 1 bullish, 2 golden
    "macd_signal_encoded",      # -1 bearish, 0 neutral, 1 bullish
    "rs_percentile",
    "momentum_4w",
    "momentum_13w",
    "momentum_26w",
    "rs_vs_spy",
    "rs_vs_sector",
]

VOLATILITY_FEATURES: List[str] = [
    "atr_pct",
    "bb_width",
    "volume_ratio",
    "adx",
    "rsi_14",
    "distance_52w_high",
    "momentum_4w",
    "days_to_earnings",         # 0 if unknown
    "vix_current",
]

EARNINGS_FEATURES: List[str] = [
    "rsi_14",
    "adx",
    "momentum_4w",
    "momentum_13w",
    "rs_percentile",
    "volume_ratio",
    "beat_streak",              # consecutive beats (neg = misses)
    "pe_ratio",
    "forward_pe",
    "revenue_growth",
    "profit_margin",
]

SECTOR_FEATURES: List[str] = [
    "sector_avg_rs",
    "sector_median_rs",
    "sector_momentum_4w",
    "vix_current",
    "treasury_10y",
    "oil_change_pct",
    "dollar_change_pct",
    "spy_momentum_4w",
]

# Mapping from model name → feature list
MODEL_FEATURES: Dict[str, List[str]] = {
    "direction": DIRECTION_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "earnings": EARNINGS_FEATURES,
    "sector_rotation": SECTOR_FEATURES,
}


# ---------------------------------------------------------------------------
# Categorical → numeric encoders
# ---------------------------------------------------------------------------

def _encode_ema_cross(value: Optional[str]) -> float:
    """Encode EMA cross string to numeric."""
    mapping = {
        "death_cross": -2.0,
        "bearish": -1.0,
        None: 0.0,
        "bullish": 1.0,
        "golden_cross": 2.0,
    }
    return mapping.get(value, 0.0)


def _encode_macd_signal(value: Optional[str]) -> float:
    """Encode MACD signal string to numeric."""
    mapping = {
        "bearish": -1.0,
        "neutral": 0.0,
        None: 0.0,
        "bullish": 1.0,
    }
    return mapping.get(value, 0.0)


# ---------------------------------------------------------------------------
# Feature vector builders
# ---------------------------------------------------------------------------

def build_direction_vector(stock: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build a feature vector for the direction model.

    Parameters
    ----------
    stock : Dict with indicator + RS data (output of scanner/scorer).

    Returns
    -------
    1-D numpy array of shape ``(len(DIRECTION_FEATURES),)`` or ``None``
    if critical features are missing.
    """
    try:
        vector = np.array([
            _safe_float(stock.get("rsi_14"), 50.0),
            _safe_float(stock.get("adx"), 20.0),
            _safe_float(stock.get("atr_pct"), 3.0),
            _safe_float(stock.get("bb_width"), 0.1),
            _safe_float(stock.get("volume_ratio"), 1.0),
            _safe_float(stock.get("distance_52w_high"), -10.0),
            _safe_float(stock.get("distance_52w_low"), 30.0),
            _encode_ema_cross(stock.get("ema_cross")),
            _encode_macd_signal(stock.get("macd_signal")),
            _safe_float(stock.get("rs_percentile"), 50.0),
            _safe_float(stock.get("momentum_4w"), 0.0),
            _safe_float(stock.get("momentum_13w"), 0.0),
            _safe_float(stock.get("momentum_26w"), 0.0),
            _safe_float(stock.get("rs_vs_spy"), 0.0),
            _safe_float(stock.get("rs_vs_sector"), 0.0),
        ], dtype=np.float32)

        assert len(vector) == len(DIRECTION_FEATURES)
        return vector

    except Exception as e:
        logger.error(f"build_direction_vector failed: {e}")
        return None


def build_volatility_vector(
    stock: Dict[str, Any],
    macro: Optional[Dict[str, Any]] = None,
    earnings_info: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Build a feature vector for the volatility model."""
    try:
        days_to_earn = 30.0  # default: no imminent earnings
        if earnings_info and earnings_info.get("days_to_earnings") is not None:
            days_to_earn = float(earnings_info["days_to_earnings"])

        vix = 15.0
        if macro and "vix" in macro:
            vix = float(macro["vix"].get("current", 15.0))

        vector = np.array([
            _safe_float(stock.get("atr_pct"), 3.0),
            _safe_float(stock.get("bb_width"), 0.1),
            _safe_float(stock.get("volume_ratio"), 1.0),
            _safe_float(stock.get("adx"), 20.0),
            _safe_float(stock.get("rsi_14"), 50.0),
            _safe_float(stock.get("distance_52w_high"), -10.0),
            _safe_float(stock.get("momentum_4w"), 0.0),
            days_to_earn,
            vix,
        ], dtype=np.float32)

        assert len(vector) == len(VOLATILITY_FEATURES)
        return vector

    except Exception as e:
        logger.error(f"build_volatility_vector failed: {e}")
        return None


def build_earnings_vector(
    stock: Dict[str, Any],
    fundamentals: Optional[Dict[str, Any]] = None,
    beat_streak: int = 0,
) -> Optional[np.ndarray]:
    """Build a feature vector for the earnings model."""
    try:
        f = fundamentals or {}

        vector = np.array([
            _safe_float(stock.get("rsi_14"), 50.0),
            _safe_float(stock.get("adx"), 20.0),
            _safe_float(stock.get("momentum_4w"), 0.0),
            _safe_float(stock.get("momentum_13w"), 0.0),
            _safe_float(stock.get("rs_percentile"), 50.0),
            _safe_float(stock.get("volume_ratio"), 1.0),
            float(beat_streak),
            _safe_float(f.get("pe_ratio"), 20.0),
            _safe_float(f.get("forward_pe"), 20.0),
            _safe_float(f.get("revenue_growth"), 0.0),
            _safe_float(f.get("profit_margin"), 0.1),
        ], dtype=np.float32)

        assert len(vector) == len(EARNINGS_FEATURES)
        return vector

    except Exception as e:
        logger.error(f"build_earnings_vector failed: {e}")
        return None


def build_sector_vector(
    sector_data: Dict[str, Any],
    macro: Optional[Dict[str, Any]] = None,
    spy_momentum_4w: float = 0.0,
) -> Optional[np.ndarray]:
    """Build a feature vector for the sector rotation model.

    Parameters
    ----------
    sector_data : Sector ranking dict with ``avg_rs``, ``median_rs``.
    macro : Macro data dict from ``fetch_macro_data()``.
    spy_momentum_4w : SPY 4-week return.
    """
    try:
        m = macro or {}

        vix = _safe_float(
            m.get("vix", {}).get("current") if isinstance(m.get("vix"), dict) else None,
            15.0,
        )
        treasury = _safe_float(
            m.get("treasury_10y", {}).get("current") if isinstance(m.get("treasury_10y"), dict) else None,
            4.0,
        )
        oil_chg = _safe_float(
            m.get("oil", {}).get("change_pct") if isinstance(m.get("oil"), dict) else None,
            0.0,
        )
        dollar_chg = _safe_float(
            m.get("dollar", {}).get("change_pct") if isinstance(m.get("dollar"), dict) else None,
            0.0,
        )

        vector = np.array([
            _safe_float(sector_data.get("avg_rs"), 0.0),
            _safe_float(sector_data.get("median_rs"), 0.0),
            _safe_float(sector_data.get("momentum_4w"), 0.0),
            vix,
            treasury,
            oil_chg,
            dollar_chg,
            spy_momentum_4w,
        ], dtype=np.float32)

        assert len(vector) == len(SECTOR_FEATURES)
        return vector

    except Exception as e:
        logger.error(f"build_sector_vector failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Generic vector builder
# ---------------------------------------------------------------------------

def build_feature_vector(
    model_name: str,
    stock: Dict[str, Any],
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """Build the feature vector for any model by name.

    Parameters
    ----------
    model_name : One of ``'direction'``, ``'volatility'``, ``'earnings'``,
                 ``'sector_rotation'``.
    stock : Data dict.
    **kwargs : Extra context (``macro``, ``fundamentals``, ``beat_streak``,
               ``earnings_info``, ``sector_data``, ``spy_momentum_4w``).
    """
    builders = {
        "direction": lambda: build_direction_vector(stock),
        "volatility": lambda: build_volatility_vector(
            stock,
            macro=kwargs.get("macro"),
            earnings_info=kwargs.get("earnings_info"),
        ),
        "earnings": lambda: build_earnings_vector(
            stock,
            fundamentals=kwargs.get("fundamentals"),
            beat_streak=kwargs.get("beat_streak", 0),
        ),
        "sector_rotation": lambda: build_sector_vector(
            stock,
            macro=kwargs.get("macro"),
            spy_momentum_4w=kwargs.get("spy_momentum_4w", 0.0),
        ),
    }

    builder = builders.get(model_name)
    if builder is None:
        logger.error(f"Unknown model: {model_name}")
        return None

    return builder()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(
    vector: Optional[np.ndarray],
    model_name: str,
) -> Tuple[bool, str]:
    """Validate a feature vector before inference.

    Returns (is_valid, message).
    """
    if vector is None:
        return False, "Vector is None"

    expected_features = MODEL_FEATURES.get(model_name)
    if expected_features is None:
        return False, f"Unknown model: {model_name}"

    if len(vector) != len(expected_features):
        return False, (
            f"Expected {len(expected_features)} features, "
            f"got {len(vector)}"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
        nan_idx = np.where(np.isnan(vector) | np.isinf(vector))[0]
        bad_features = [expected_features[i] for i in nan_idx]
        return False, f"NaN/Inf in features: {bad_features}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default
