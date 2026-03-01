"""
Supabase persistence layer for stocks-agent.

All CRUD operations for the ``stocks`` schema.
Every function returns a success boolean and logs errors internally.
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[Client] = None


def init_supabase() -> Client:
    """Initialise and return the Supabase client (singleton).

    Reads ``STOCKS_SUPABASE_URL`` and ``STOCKS_SUPABASE_KEY``
    from the environment.

    Raises
    ------
    ValueError
        If required environment variables are missing.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("STOCKS_SUPABASE_URL")
    key = os.getenv("STOCKS_SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "STOCKS_SUPABASE_URL and STOCKS_SUPABASE_KEY must be set"
        )

    _client = create_client(url, key)
    logger.info("Supabase client initialised")
    return _client


def _get_client() -> Client:
    """Return the existing Supabase client, or initialise one."""
    if _client is None:
        return init_supabase()
    return _client


# ---------------------------------------------------------------------------
# Helper: table reference with schema
# ---------------------------------------------------------------------------

def _table(name: str) -> Any:
    """Return a table reference in the ``stocks`` schema."""
    client = _get_client()
    return client.schema("stocks").table(name)


# ---------------------------------------------------------------------------
# stocks.universe
# ---------------------------------------------------------------------------

def upsert_universe(stocks: List[Dict[str, Any]]) -> bool:
    """Upsert S&P 500 constituents into ``stocks.universe``.

    Parameters
    ----------
    stocks : List of dicts with keys ``ticker``, ``company_name``,
             ``sector``, ``industry`` and optionally ``market_cap_b``,
             ``avg_volume_50d``.

    Returns
    -------
    ``True`` on success, ``False`` on failure.
    """
    if not stocks:
        logger.warning("upsert_universe: empty list")
        return False
    try:
        now = datetime.utcnow().isoformat()
        rows = []
        for s in stocks:
            rows.append({
                "ticker": s["ticker"],
                "company_name": s["company_name"],
                "sector": s["sector"],
                "industry": s["industry"],
                "market_cap_b": s.get("market_cap_b"),
                "avg_volume_50d": s.get("avg_volume_50d"),
                "in_sp500": True,
                "is_active": True,
                "updated_at": now,
            })

        result = _table("universe").upsert(
            rows, on_conflict="ticker"
        ).execute()

        logger.info(f"Upserted {len(rows)} stocks into universe")
        return True
    except Exception as e:
        logger.error(f"upsert_universe failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# stocks.daily_scores
# ---------------------------------------------------------------------------

def insert_daily_scores(scores: List[Dict[str, Any]]) -> bool:
    """Batch-insert daily scores for all scanned stocks.

    Parameters
    ----------
    scores : List of dicts matching the ``stocks.daily_scores`` schema.

    Returns
    -------
    ``True`` on success, ``False`` on failure.
    """
    if not scores:
        logger.warning("insert_daily_scores: empty list")
        return False
    try:
        # Batch in chunks of 100 to avoid payload size limits
        chunk_size = 100
        for i in range(0, len(scores), chunk_size):
            chunk = scores[i : i + chunk_size]
            _table("daily_scores").insert(chunk).execute()

        logger.info(f"Inserted {len(scores)} daily scores")
        return True
    except Exception as e:
        logger.error(f"insert_daily_scores failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# stocks.opportunities
# ---------------------------------------------------------------------------

def insert_opportunity(opp: Dict[str, Any]) -> bool:
    """Insert a new trade opportunity.

    Parameters
    ----------
    opp : Dict matching the ``stocks.opportunities`` schema.

    Returns
    -------
    ``True`` on success, ``False`` on failure.
    """
    try:
        _table("opportunities").insert(opp).execute()
        logger.info(
            f"Inserted opportunity: {opp.get('ticker')} "
            f"confidence={opp.get('confidence')}"
        )
        return True
    except Exception as e:
        logger.error(f"insert_opportunity failed: {e}", exc_info=True)
        return False


def insert_opportunities(opps: List[Dict[str, Any]]) -> bool:
    """Batch-insert multiple opportunities."""
    if not opps:
        return True
    try:
        _table("opportunities").insert(opps).execute()
        logger.info(f"Inserted {len(opps)} opportunities")
        return True
    except Exception as e:
        logger.error(f"insert_opportunities failed: {e}", exc_info=True)
        return False


def get_pending_opportunities() -> List[Dict[str, Any]]:
    """Get opportunities with status ``pending``."""
    try:
        result = (
            _table("opportunities")
            .select("*")
            .eq("status", "pending")
            .order("confidence", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"get_pending_opportunities failed: {e}", exc_info=True)
        return []


def update_opportunity(opp_id: str, updates: Dict[str, Any]) -> bool:
    """Update an opportunity by ID."""
    try:
        _table("opportunities").update(updates).eq("id", opp_id).execute()
        return True
    except Exception as e:
        logger.error(f"update_opportunity failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# stocks.positions
# ---------------------------------------------------------------------------

def get_open_positions() -> List[Dict[str, Any]]:
    """Get all positions with status ``open``.

    Returns
    -------
    List of position dicts, or empty list on failure.
    """
    try:
        result = (
            _table("positions")
            .select("*")
            .eq("status", "open")
            .order("entry_date", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"get_open_positions failed: {e}", exc_info=True)
        return []


def insert_position(position: Dict[str, Any]) -> bool:
    """Insert a new advisory position."""
    try:
        _table("positions").insert(position).execute()
        logger.info(f"Inserted position: {position.get('ticker')}")
        return True
    except Exception as e:
        logger.error(f"insert_position failed: {e}", exc_info=True)
        return False


def update_position(position_id: str, updates: Dict[str, Any]) -> bool:
    """Update a position by ID.

    Common updates: ``current_price``, ``unrealized_pnl``,
    ``trailing_stop``, ``status``, ``exit_price``.
    """
    try:
        updates["updated_at"] = datetime.utcnow().isoformat()
        _table("positions").update(updates).eq("id", position_id).execute()
        logger.debug(f"Updated position {position_id}")
        return True
    except Exception as e:
        logger.error(f"update_position failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# stocks.trades
# ---------------------------------------------------------------------------

def insert_trade(trade: Dict[str, Any]) -> bool:
    """Insert a closed trade record."""
    try:
        _table("trades").insert(trade).execute()
        logger.info(
            f"Inserted trade: {trade.get('ticker')} "
            f"pnl={trade.get('realized_pnl')}"
        )
        return True
    except Exception as e:
        logger.error(f"insert_trade failed: {e}", exc_info=True)
        return False


def get_trade_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent closed trades."""
    try:
        result = (
            _table("trades")
            .select("*")
            .order("exit_date", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"get_trade_history failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# stocks.market_events
# ---------------------------------------------------------------------------

def insert_event(event: Dict[str, Any]) -> bool:
    """Insert a detected market event."""
    try:
        _table("market_events").insert(event).execute()
        logger.info(
            f"Inserted event: {event.get('event_type')} "
            f"severity={event.get('severity')}"
        )
        return True
    except Exception as e:
        logger.error(f"insert_event failed: {e}", exc_info=True)
        return False


def get_recent_events(days: int = 7) -> List[Dict[str, Any]]:
    """Get market events from the last N days."""
    try:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        result = (
            _table("market_events")
            .select("*")
            .gte("event_date", cutoff)
            .order("event_date", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"get_recent_events failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# stocks.equity_snapshots
# ---------------------------------------------------------------------------

def insert_equity_snapshot(snapshot: Dict[str, Any]) -> bool:
    """Insert a weekly equity/portfolio snapshot."""
    try:
        _table("equity_snapshots").insert(snapshot).execute()
        logger.info(
            f"Inserted equity snapshot: "
            f"total={snapshot.get('total_value')}"
        )
        return True
    except Exception as e:
        logger.error(f"insert_equity_snapshot failed: {e}", exc_info=True)
        return False


def get_latest_snapshot() -> Optional[Dict[str, Any]]:
    """Get the most recent equity snapshot."""
    try:
        result = (
            _table("equity_snapshots")
            .select("*")
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
        )
        data = result.data
        return data[0] if data else None
    except Exception as e:
        logger.error(f"get_latest_snapshot failed: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# stocks.model_versions
# ---------------------------------------------------------------------------

def get_active_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Get the currently active version of a model."""
    try:
        result = (
            _table("model_versions")
            .select("*")
            .eq("model_name", model_name)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        data = result.data
        return data[0] if data else None
    except Exception as e:
        logger.error(f"get_active_model failed: {e}", exc_info=True)
        return None


def insert_model_version(model: Dict[str, Any]) -> bool:
    """Insert a new model version record."""
    try:
        _table("model_versions").insert(model).execute()
        logger.info(
            f"Inserted model version: {model.get('model_name')} "
            f"v{model.get('version')}"
        )
        return True
    except Exception as e:
        logger.error(f"insert_model_version failed: {e}", exc_info=True)
        return False


def activate_model(model_name: str, version_id: str) -> bool:
    """Set a specific model version as active (deactivate others)."""
    try:
        # Deactivate all versions of this model
        _table("model_versions").update(
            {"is_active": False}
        ).eq("model_name", model_name).execute()

        # Activate the specified version
        _table("model_versions").update(
            {"is_active": True, "deployed_at": datetime.utcnow().isoformat()}
        ).eq("id", version_id).execute()

        logger.info(f"Activated model {model_name} version {version_id}")
        return True
    except Exception as e:
        logger.error(f"activate_model failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Data maintenance
# ---------------------------------------------------------------------------

def purge_old_scores(days: int = 180) -> int:
    """Delete daily_scores older than *days*.

    Returns the number of rows deleted, or -1 on failure.
    """
    try:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        result = (
            _table("daily_scores")
            .delete()
            .lt("score_date", cutoff)
            .execute()
        )
        count = len(result.data) if result.data else 0
        logger.info(f"Purged {count} daily_scores older than {days} days")
        return count
    except Exception as e:
        logger.error(f"purge_old_scores failed: {e}", exc_info=True)
        return -1


def get_storage_estimate() -> Optional[Dict[str, int]]:
    """Estimate row counts for all stocks tables.

    Useful for monitoring Supabase storage usage.
    """
    tables = [
        "universe",
        "daily_scores",
        "opportunities",
        "positions",
        "market_events",
        "equity_snapshots",
        "model_versions",
        "trades",
    ]
    result: Dict[str, int] = {}
    try:
        for tbl in tables:
            resp = (
                _table(tbl)
                .select("id", count="exact")
                .execute()
            )
            result[tbl] = resp.count if resp.count is not None else 0
        return result
    except Exception as e:
        logger.error(f"get_storage_estimate failed: {e}", exc_info=True)
        return None
