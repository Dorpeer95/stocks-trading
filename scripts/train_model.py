#!/usr/bin/env python3
"""
Local training pipeline for stocks-agent ML models.

Run on your Mac (not on the server).  Trains 4 XGBoost classifiers
and optionally uploads them to Supabase Storage.

Usage
-----
    # Train all models
    python scripts/train_model.py --all

    # Train a specific model
    python scripts/train_model.py --model direction

    # Train + upload to Supabase
    python scripts/train_model.py --all --upload

    # Evaluate only (no training)
    python scripts/train_model.py --model direction --evaluate-only

Requirements
------------
    pip install xgboost scikit-learn pandas numpy yfinance python-dotenv
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agent.feature_config import (
    DIRECTION_FEATURES,
    VOLATILITY_FEATURES,
    EARNINGS_FEATURES,
    SECTOR_FEATURES,
    MODEL_FEATURES,
    build_direction_vector,
    build_volatility_vector,
    build_earnings_vector,
    build_sector_vector,
    _safe_float,
    _encode_ema_cross,
    _encode_macd_signal,
)
from utils.indicators import calc_all_indicators
from utils.data_loader import fetch_batch_prices, fetch_sp500_list, fetch_price_data
from utils.sectors import compute_universe_rs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Target definitions
DIRECTION_HORIZON = 10       # trading days
DIRECTION_TARGET_PCT = 5.0   # up 5%
VOLATILITY_HORIZON = 5       # trading days
VOLATILITY_ATR_MULT = 2.0    # move > 2 × ATR

# Training data lookback
TRAINING_PERIOD = "2y"       # 2 years of data
VALIDATION_SPLIT = 0.2       # 20% validation


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    model_name: str,
    tickers: Optional[List[str]] = None,
    period: str = TRAINING_PERIOD,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare labeled training data for a model.

    Parameters
    ----------
    model_name : ``'direction'``, ``'volatility'``, ``'earnings'``,
                 or ``'sector_rotation'``.
    tickers : Stock tickers to use.  Defaults to S&P 500 sample.
    period : yfinance period for historical data.

    Returns
    -------
    ``(X, y)`` where X is ``(n_samples, n_features)`` and y is ``(n_samples,)``
    with binary labels.
    """
    if tickers is None:
        sp500 = fetch_sp500_list()
        if not sp500:
            logger.error("Failed to fetch S&P 500 list")
            return None, None
        # Use a representative sample to keep training fast
        tickers = [s["ticker"] for s in sp500[:100]]
        logger.info(f"Using {len(tickers)} tickers for training")

    if model_name == "direction":
        return _prepare_direction_data(tickers, period)
    elif model_name == "volatility":
        return _prepare_volatility_data(tickers, period)
    elif model_name == "earnings":
        return _prepare_earnings_data(tickers, period)
    elif model_name == "sector_rotation":
        return _prepare_sector_data(tickers, period)
    else:
        logger.error(f"Unknown model: {model_name}")
        return None, None


def _prepare_direction_data(
    tickers: List[str],
    period: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build direction training data.

    For each stock on each day (with enough lookback), compute the
    feature vector and label: 1 if stock is up ≥ 5% in the next 10 days.
    """
    logger.info("Preparing direction training data...")
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    # Download in batches
    batch_size = 25
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"  Batch {i // batch_size + 1}: {len(batch)} tickers")

        batch_data = fetch_batch_prices(batch, period=period)
        if not batch_data:
            continue

        # Fetch SPY for RS calculation
        spy_df = fetch_price_data("SPY", period=period)

        for ticker, df in batch_data.items():
            if df is None or len(df) < 252:  # need ~1y lookback
                continue

            try:
                close = df["Close"]
                # Slide a window across the history
                for day_idx in range(252, len(df) - DIRECTION_HORIZON):
                    window = df.iloc[:day_idx + 1]
                    future_close = close.iloc[day_idx + DIRECTION_HORIZON]
                    current_close = close.iloc[day_idx]

                    # Label
                    pct_change = ((future_close - current_close) / current_close) * 100
                    label = 1 if pct_change >= DIRECTION_TARGET_PCT else 0

                    # Compute indicators on window
                    indicators = calc_all_indicators(window)
                    if indicators is None:
                        continue

                    # RS vs SPY (simplified for training)
                    rs_spy = 0.0
                    if spy_df is not None and day_idx < len(spy_df):
                        spy_window = spy_df.iloc[:day_idx + 1]
                        if len(spy_window) >= 130:
                            from utils.sectors import _pct_return
                            stock_ret = _pct_return(window["Close"], 65)
                            spy_ret = _pct_return(spy_window["Close"], 65)
                            if stock_ret is not None and spy_ret is not None:
                                rs_spy = stock_ret - spy_ret

                    stock_dict = {
                        **indicators,
                        "rs_percentile": 50.0,  # approximate
                        "momentum_4w": _pct_return_safe(close, day_idx, 20),
                        "momentum_13w": _pct_return_safe(close, day_idx, 65),
                        "momentum_26w": _pct_return_safe(close, day_idx, 130),
                        "rs_vs_spy": rs_spy,
                        "rs_vs_sector": 0.0,
                    }

                    vec = build_direction_vector(stock_dict)
                    if vec is not None:
                        features_list.append(vec)
                        labels_list.append(label)

            except Exception as e:
                logger.warning(f"  Skipped {ticker}: {e}")

        # Memory management
        del batch_data
        gc.collect()
        time.sleep(1)  # rate limit

    if not features_list:
        logger.error("No training data produced")
        return None, None

    X = np.stack(features_list)
    y = np.array(labels_list)
    logger.info(
        f"Direction data: {X.shape[0]} samples, "
        f"{X.shape[1]} features, "
        f"{y.mean():.1%} positive rate"
    )
    return X, y


def _prepare_volatility_data(
    tickers: List[str],
    period: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build volatility training data.

    Label: 1 if stock moves > 2 × ATR in next 5 days.
    """
    logger.info("Preparing volatility training data...")
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    batch_size = 25
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_data = fetch_batch_prices(batch, period=period)
        if not batch_data:
            continue

        for ticker, df in batch_data.items():
            if df is None or len(df) < 252:
                continue

            try:
                from utils.indicators import calc_atr

                close = df["Close"]
                atr_series = calc_atr(df, period=14)
                if atr_series is None:
                    continue

                for day_idx in range(252, len(df) - VOLATILITY_HORIZON):
                    current_close = close.iloc[day_idx]
                    atr_val = atr_series.iloc[day_idx]

                    if pd.isna(atr_val) or atr_val <= 0:
                        continue

                    # Future max move
                    future_slice = close.iloc[day_idx + 1:day_idx + 1 + VOLATILITY_HORIZON]
                    max_move = max(
                        abs(future_slice.max() - current_close),
                        abs(future_slice.min() - current_close),
                    )

                    label = 1 if max_move > (VOLATILITY_ATR_MULT * atr_val) else 0

                    # Build features
                    window = df.iloc[:day_idx + 1]
                    indicators = calc_all_indicators(window)
                    if indicators is None:
                        continue

                    stock_dict = {
                        **indicators,
                        "momentum_4w": _pct_return_safe(close, day_idx, 20),
                    }

                    vec = build_volatility_vector(stock_dict)
                    if vec is not None:
                        features_list.append(vec)
                        labels_list.append(label)

            except Exception as e:
                logger.warning(f"  Skipped {ticker}: {e}")

        del batch_data
        gc.collect()
        time.sleep(1)

    if not features_list:
        return None, None

    X = np.stack(features_list)
    y = np.array(labels_list)
    logger.info(
        f"Volatility data: {X.shape[0]} samples, "
        f"{y.mean():.1%} positive rate"
    )
    return X, y


def _prepare_earnings_data(
    tickers: List[str],
    period: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build earnings training data.

    Uses historical earnings dates: label 1 if stock closed higher
    5 days after earnings vs day before.
    """
    logger.info("Preparing earnings training data...")
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    import yfinance as yf

    for ticker in tickers[:50]:  # limit for speed
        try:
            t = yf.Ticker(ticker)
            earnings_dates = t.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                continue

            df = fetch_price_data(ticker, period=period)
            if df is None or len(df) < 100:
                continue

            close = df["Close"]

            for idx in earnings_dates.index:
                earn_date = pd.Timestamp(idx).tz_localize(None)

                # Find the closest trading day
                try:
                    day_before_idx = df.index.get_indexer([earn_date], method="pad")[0]
                except Exception:
                    continue

                if day_before_idx < 50 or day_before_idx + 5 >= len(df):
                    continue

                pre_price = close.iloc[day_before_idx]
                post_price = close.iloc[day_before_idx + 5]

                label = 1 if post_price > pre_price else 0

                # Features from pre-earnings window
                window = df.iloc[:day_before_idx + 1]
                indicators = calc_all_indicators(window)
                if indicators is None:
                    continue

                # Get beat streak from earnings data
                est = earnings_dates.iloc[:].get("EPS Estimate")
                act = earnings_dates.iloc[:].get("Reported EPS")
                beat_streak = 0
                if est is not None and act is not None:
                    for _, row in earnings_dates.iterrows():
                        e_val = row.get("EPS Estimate")
                        a_val = row.get("Reported EPS")
                        if pd.notna(e_val) and pd.notna(a_val):
                            if a_val > e_val:
                                beat_streak += 1
                            else:
                                break

                stock_dict = {
                    **indicators,
                    "rs_percentile": 50.0,
                    "momentum_4w": _pct_return_safe(close, day_before_idx, 20),
                    "momentum_13w": _pct_return_safe(close, day_before_idx, 65),
                }

                fund = t.info or {}
                vec = build_earnings_vector(
                    stock_dict,
                    fundamentals={
                        "pe_ratio": fund.get("trailingPE"),
                        "forward_pe": fund.get("forwardPE"),
                        "revenue_growth": fund.get("revenueGrowth"),
                        "profit_margin": fund.get("profitMargins"),
                    },
                    beat_streak=beat_streak,
                )
                if vec is not None:
                    features_list.append(vec)
                    labels_list.append(label)

        except Exception as e:
            logger.warning(f"  Skipped {ticker} earnings: {e}")

        time.sleep(0.5)

    if not features_list:
        return None, None

    X = np.stack(features_list)
    y = np.array(labels_list)
    logger.info(
        f"Earnings data: {X.shape[0]} samples, "
        f"{y.mean():.1%} positive rate"
    )
    return X, y


def _prepare_sector_data(
    tickers: List[str],
    period: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build sector rotation training data.

    For each sector, on each week: label 1 if sector ETF outperformed
    SPY over the next 4 weeks.
    """
    logger.info("Preparing sector rotation training data...")
    from utils.data_loader import SECTOR_ETF_MAP

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    spy_df = fetch_price_data("SPY", period=period)
    if spy_df is None:
        return None, None

    etf_tickers = list(set(SECTOR_ETF_MAP.values()))
    etf_data = fetch_batch_prices(etf_tickers, period=period)
    if not etf_data:
        return None, None

    spy_close = spy_df["Close"]

    for etf_ticker, df in etf_data.items():
        if df is None or len(df) < 150:
            continue

        close = df["Close"]

        # Weekly samples (every 5 trading days)
        for day_idx in range(130, len(df) - 20, 5):
            try:
                current = close.iloc[day_idx]
                future = close.iloc[day_idx + 20]  # 4 weeks forward

                spy_current = spy_close.iloc[day_idx] if day_idx < len(spy_close) else None
                spy_future = spy_close.iloc[day_idx + 20] if day_idx + 20 < len(spy_close) else None

                if spy_current is None or spy_future is None:
                    continue

                etf_return = (future - current) / current
                spy_return = (spy_future - spy_current) / spy_current
                label = 1 if etf_return > spy_return else 0

                # Sector features (simplified)
                window = df.iloc[:day_idx + 1]

                from utils.sectors import _pct_return
                avg_rs = (_pct_return(close[:day_idx + 1], 65) or 0) - (_pct_return(spy_close[:day_idx + 1], 65) or 0)

                sector_dict = {
                    "avg_rs": avg_rs,
                    "median_rs": avg_rs,  # approximation
                    "momentum_4w": _pct_return(close[:day_idx + 1], 20) or 0,
                }

                spy_m4w = _pct_return(spy_close[:day_idx + 1], 20) or 0

                vec = build_sector_vector(sector_dict, spy_momentum_4w=spy_m4w)
                if vec is not None:
                    features_list.append(vec)
                    labels_list.append(label)

            except Exception:
                continue

    if not features_list:
        return None, None

    X = np.stack(features_list)
    y = np.array(labels_list)
    logger.info(
        f"Sector data: {X.shape[0]} samples, "
        f"{y.mean():.1%} positive rate"
    )
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """Train an XGBoost classifier.

    Returns (model, metrics_dict).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    from xgboost import XGBClassifier

    logger.info(f"Training '{model_name}' on {X.shape[0]} samples...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    # XGBoost hyperparameters (conservative for small datasets)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_val, y_proba), 4),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
    }

    logger.info(
        f"  {model_name} metrics: "
        f"acc={metrics['accuracy']}, "
        f"f1={metrics['f1']}, "
        f"auc={metrics['auc_roc']}"
    )

    # Feature importances
    feature_names = MODEL_FEATURES.get(model_name, [])
    if len(feature_names) == X.shape[1]:
        importances = model.feature_importances_
        top_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        logger.info(f"  Top features: {top_features}")

    return model, metrics


def save_model(model: Any, model_name: str) -> str:
    """Save a trained model to disk.

    Returns the file path.
    """
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    size_kb = path.stat().st_size // 1024
    logger.info(f"Saved {model_name}.pkl ({size_kb} KB)")
    return str(path)


def evaluate_model(model_name: str) -> Optional[Dict[str, float]]:
    """Load and evaluate an existing model.

    Returns metrics dict or None.
    """
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        logger.error(f"Model not found: {path}")
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)

    X, y = prepare_training_data(model_name)
    if X is None or y is None:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(f"  {model_name} eval: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Upload to Supabase
# ---------------------------------------------------------------------------

def upload_model(
    model_name: str,
    metrics: Dict[str, float],
    feature_names: List[str],
) -> bool:
    """Upload a trained model to Supabase Storage and record metadata."""
    local_path = str(MODEL_DIR / f"{model_name}.pkl")
    version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    remote_path = f"{model_name}/{version}.pkl"

    # Upload binary
    from agent.ai_model import upload_to_storage

    if not upload_to_storage(local_path, remote_path):
        return False

    # Record metadata
    file_size = Path(local_path).stat().st_size // 1024

    from agent.persistence import insert_model_version, activate_model

    model_record = {
        "model_name": model_name,
        "version": version,
        "accuracy": metrics.get("accuracy"),
        "precision_score": metrics.get("precision"),
        "recall_score": metrics.get("recall"),
        "f1_score": metrics.get("f1"),
        "auc_roc": metrics.get("auc_roc"),
        "training_samples": metrics.get("train_samples"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
        },
        "storage_path": remote_path,
        "file_size_kb": file_size,
        "trained_at": datetime.utcnow().isoformat(),
    }

    if not insert_model_version(model_record):
        return False

    logger.info(f"Uploaded {model_name} v{version}")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_return_safe(series: pd.Series, current_idx: int, lookback: int) -> float:
    """Safe percentage return calculation for training data."""
    if current_idx < lookback:
        return 0.0
    try:
        old = float(series.iloc[current_idx - lookback])
        new = float(series.iloc[current_idx])
        if old == 0 or np.isnan(old) or np.isnan(new):
            return 0.0
        return round(((new - old) / old) * 100, 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train stocks-agent ML models")
    parser.add_argument("--model", type=str, help="Model name to train")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--upload", action="store_true", help="Upload to Supabase after training")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing model")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (default: S&P 500 sample)")
    args = parser.parse_args()

    models_to_train = []
    if args.all:
        models_to_train = ["direction", "volatility", "earnings", "sector_rotation"]
    elif args.model:
        models_to_train = [args.model]
    else:
        parser.print_help()
        return

    if args.tickers:
        tickers = args.tickers.split(",")
    else:
        tickers = None

    for model_name in models_to_train:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  MODEL: {model_name}")
        logger.info(f"{'=' * 60}")

        if args.evaluate_only:
            evaluate_model(model_name)
            continue

        # Prepare data
        X, y = prepare_training_data(model_name, tickers=tickers)
        if X is None or y is None:
            logger.error(f"Failed to prepare data for {model_name}")
            continue

        # Train
        model, metrics = train_model(model_name, X, y)

        # Save
        save_model(model, model_name)

        # Upload
        if args.upload:
            feature_names = MODEL_FEATURES.get(model_name, [])
            upload_model(model_name, metrics, feature_names)

        # Cleanup
        del X, y, model
        gc.collect()

    logger.info("\n✅ Training complete")


if __name__ == "__main__":
    main()
