"""
AI model manager — loads XGBoost models and runs inference.

Models are stored as ``.pkl`` files.  On the server they are downloaded
from Supabase Storage on startup / on schedule.  During local
development they can be loaded from disk.

Memory notes
------------
Each XGBoost binary classifier is ~2-3 MB in RAM.  With 4 models
loaded, expect ~12 MB overhead.  Models are lazily loaded and can
be explicitly unloaded via ``ModelManager.unload_all()``.
"""

import gc
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent.feature_config import (
    MODEL_FEATURES,
    build_feature_vector,
    validate_features,
)
from agent.persistence import get_active_model, insert_model_version

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("STOCKS_MODEL_DIR", "models"))
MODEL_NAMES = ["direction", "volatility", "earnings", "sector_rotation"]

# Probability thresholds for converting predictions to signals
DIRECTION_THRESHOLD = 0.55      # P(up) > 55% → bullish
VOLATILITY_THRESHOLD = 0.60     # P(big move) > 60% → high vol expected
EARNINGS_THRESHOLD = 0.55       # P(positive reaction) > 55% → bullish
SECTOR_THRESHOLD = 0.55         # P(outperform) > 55% → overweight


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages loading, inference, and lifecycle of XGBoost models."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}          # model_name → XGBClassifier
        self._metadata: Dict[str, Dict] = {}        # model_name → version info
        self._loaded = False

    @property
    def models(self) -> Dict[str, Any]:
        """Public read-only access to loaded models dict."""
        return self._models

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_disk(self, model_dir: Optional[Path] = None) -> int:
        """Load all available models from local disk.

        Parameters
        ----------
        model_dir : Directory containing ``.pkl`` files named
                    ``direction.pkl``, ``volatility.pkl``, etc.

        Returns
        -------
        Number of models loaded.
        """
        d = model_dir or MODEL_DIR
        d.mkdir(parents=True, exist_ok=True)
        loaded = 0

        for name in MODEL_NAMES:
            path = d / f"{name}.pkl"
            if not path.exists():
                logger.debug(f"Model file not found: {path}")
                continue

            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)

                self._models[name] = model
                self._metadata[name] = {
                    "source": "disk",
                    "path": str(path),
                    "feature_count": len(MODEL_FEATURES.get(name, [])),
                }
                loaded += 1
                logger.info(f"Loaded model '{name}' from {path}")

            except Exception as e:
                logger.error(f"Failed to load model '{name}': {e}")

        self._loaded = loaded > 0
        logger.info(f"ModelManager: {loaded}/{len(MODEL_NAMES)} models loaded")
        return loaded

    def load_from_supabase(self) -> int:
        """Download active models from Supabase Storage.

        For each model name, looks up the active version in
        ``stocks.model_versions``, downloads the ``.pkl`` from
        Supabase Storage, and loads it.

        Returns
        -------
        Number of models loaded.
        """
        loaded = 0

        for name in MODEL_NAMES:
            try:
                meta = get_active_model(name)
                if meta is None:
                    logger.debug(f"No active model version for '{name}'")
                    continue

                storage_path = meta.get("storage_path")
                if not storage_path:
                    logger.warning(f"No storage_path for model '{name}'")
                    continue

                # Download from Supabase Storage
                model_bytes = _download_from_storage(storage_path)
                if model_bytes is None:
                    continue

                model = pickle.loads(model_bytes)
                self._models[name] = model
                self._metadata[name] = {
                    "source": "supabase",
                    "version": meta.get("version"),
                    "accuracy": meta.get("accuracy"),
                    "trained_at": meta.get("trained_at"),
                    "feature_count": meta.get("feature_count"),
                }
                loaded += 1
                logger.info(
                    f"Loaded model '{name}' v{meta.get('version')} "
                    f"from Supabase (acc={meta.get('accuracy')})"
                )

            except Exception as e:
                logger.error(f"Failed to load model '{name}' from Supabase: {e}")

        self._loaded = loaded > 0
        logger.info(f"ModelManager (Supabase): {loaded}/{len(MODEL_NAMES)} models loaded")
        return loaded

    def is_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if models are loaded.

        If *model_name* is given, check that specific model.
        Otherwise check if any model is loaded.
        """
        if model_name:
            return model_name in self._models
        return self._loaded

    def unload_all(self) -> None:
        """Unload all models to free RAM (~12 MB)."""
        self._models.clear()
        self._metadata.clear()
        self._loaded = False
        gc.collect()
        logger.info("ModelManager: all models unloaded")

    def get_metadata(self) -> Dict[str, Dict]:
        """Return metadata for all loaded models."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_direction(
        self,
        stock: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Predict P(stock up 5% in 10 trading days).

        Returns
        -------
        Dict with ``probability``, ``signal`` ('bullish'/'bearish'/'neutral'),
        ``confidence`` (0-100).  ``None`` if model not loaded.
        """
        return self._predict("direction", stock)

    def predict_volatility(
        self,
        stock: Dict[str, Any],
        macro: Optional[Dict] = None,
        earnings_info: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Predict P(stock moves > 2 ATR in 5 days)."""
        return self._predict(
            "volatility", stock, macro=macro, earnings_info=earnings_info
        )

    def predict_earnings(
        self,
        stock: Dict[str, Any],
        fundamentals: Optional[Dict] = None,
        beat_streak: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Predict P(post-earnings move is positive)."""
        return self._predict(
            "earnings", stock, fundamentals=fundamentals, beat_streak=beat_streak
        )

    def predict_sector_rotation(
        self,
        sector_data: Dict[str, Any],
        macro: Optional[Dict] = None,
        spy_momentum_4w: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Predict P(sector outperforms SPY next 4 weeks)."""
        return self._predict(
            "sector_rotation",
            sector_data,
            macro=macro,
            spy_momentum_4w=spy_momentum_4w,
        )

    def predict_batch(
        self,
        model_name: str,
        stocks: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Optional[Dict[str, Any]]]:
        """Run predictions for a batch of stocks.

        Returns list of prediction dicts (same order as input).
        """
        results: List[Optional[Dict[str, Any]]] = []
        for stock in stocks:
            pred = self._predict(model_name, stock, **kwargs)
            results.append(pred)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(
        self,
        model_name: str,
        data: Dict[str, Any],
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Generic prediction for any model."""
        model = self._models.get(model_name)
        if model is None:
            logger.debug(f"Model '{model_name}' not loaded — skipping")
            return None

        # Build feature vector
        vector = build_feature_vector(model_name, data, **kwargs)
        is_valid, msg = validate_features(vector, model_name)
        if not is_valid:
            logger.warning(f"Invalid features for '{model_name}': {msg}")
            return None

        try:
            # XGBoost predict_proba returns [[P(0), P(1)]]
            proba = model.predict_proba(vector.reshape(1, -1))
            prob_positive = float(proba[0][1])

            # Convert to signal
            threshold = _get_threshold(model_name)
            if prob_positive >= threshold:
                signal = "bullish"
            elif prob_positive <= (1 - threshold):
                signal = "bearish"
            else:
                signal = "neutral"

            # Confidence: distance from 0.5 scaled to 0-100
            confidence = round(abs(prob_positive - 0.5) * 200, 1)

            return {
                "model": model_name,
                "probability": round(prob_positive, 4),
                "signal": signal,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Prediction failed for '{model_name}': {e}")
            return None


# ---------------------------------------------------------------------------
# Supabase Storage download helper
# ---------------------------------------------------------------------------

def _download_from_storage(storage_path: str) -> Optional[bytes]:
    """Download a file from Supabase Storage.

    Uses the ``models`` bucket.
    """
    try:
        from agent.persistence import _get_client

        client = _get_client()
        bucket = client.storage.from_("models")
        data = bucket.download(storage_path)
        logger.debug(f"Downloaded {storage_path} ({len(data)} bytes)")
        return data

    except Exception as e:
        logger.error(f"Failed to download {storage_path}: {e}")
        return None


def upload_to_storage(
    local_path: str,
    remote_path: str,
) -> bool:
    """Upload a model file to Supabase Storage.

    Uses the ``models`` bucket.
    """
    try:
        from agent.persistence import _get_client

        client = _get_client()
        bucket = client.storage.from_("models")

        with open(local_path, "rb") as f:
            data = f.read()

        bucket.upload(remote_path, data, {"content-type": "application/octet-stream"})
        logger.info(f"Uploaded {local_path} → {remote_path} ({len(data)} bytes)")
        return True

    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_threshold(model_name: str) -> float:
    """Return the probability threshold for a model."""
    return {
        "direction": DIRECTION_THRESHOLD,
        "volatility": VOLATILITY_THRESHOLD,
        "earnings": EARNINGS_THRESHOLD,
        "sector_rotation": SECTOR_THRESHOLD,
    }.get(model_name, 0.55)
