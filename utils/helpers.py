"""
Safe type-conversion helpers for Supabase data.

Supabase returns ``None`` for NULL columns even when ``.get("key", default)``
is used — the key *exists* in the dict with value ``None``, so the default
is never reached.  These helpers guarantee a valid numeric return.
"""

from typing import Any


def safe_float(val: Any, default: float = 0.0) -> float:
    """Convert *val* to float, returning *default* on None / error."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Convert *val* to int, returning *default* on None / error."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default
