#!/usr/bin/env python3
"""
Generate ML models using synthetic training data.

This is used when the yfinance API is unavailable (connection issues,
rate limits, etc.) to bootstrap the earnings and sector_rotation models
so the rest of the pipeline can load and run them.

The models are intentionally conservative — they will produce near-50/50
predictions until retrained on real data via:
    python scripts/train_model.py --model earnings
    python scripts/train_model.py --model sector_rotation

Usage:
    python scripts/train_synthetic.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.feature_config import EARNINGS_FEATURES, SECTOR_FEATURES


def generate_synthetic_data(n_features: int, n_samples: int = 2000, seed: int = 42):
    """Generate synthetic X, y data with realistic feature distributions."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)
    # Introduce mild signal: positive correlation with sum of first 3 features
    signal = X[:, :3].sum(axis=1)
    prob = 1 / (1 + np.exp(-0.3 * signal))  # sigmoid with weak signal
    y = (rng.random(n_samples) < prob).astype(int)
    return X, y


def train_and_save(model_name: str, features: list, seed: int = 42):
    """Train an XGBoost model on synthetic data and save it."""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y = generate_synthetic_data(len(features), seed=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

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

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    path = model_dir / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    size_kb = path.stat().st_size // 1024
    acc = model.score(X_val, y_val)
    print(f"  ✅ {model_name}.pkl saved ({size_kb} KB, synthetic acc={acc:.2%})")
    print(f"     Features: {len(features)}, Samples: {len(X)}")
    print(f"     ⚠️  Retrain on real data: python scripts/train_model.py --model {model_name}")


def main():
    print("Training missing models with synthetic data...\n")
    train_and_save("earnings", EARNINGS_FEATURES, seed=43)
    print()
    train_and_save("sector", SECTOR_FEATURES, seed=44)
    print("\n✅ Done. Both models are now loadable by ModelManager.")


if __name__ == "__main__":
    main()
