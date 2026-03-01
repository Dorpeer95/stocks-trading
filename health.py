"""
Health check endpoint for stocks-agent.

Runs a lightweight FastAPI server on port 8001 (crypto bot uses 8000).
Provides health and status endpoints for monitoring.
Includes a memory watchdog that alerts when RSS approaches limits.
"""

import logging
import os
import time
from typing import Any, Dict

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="stocks-agent",
    description="Stock trading advisory bot health check",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEMORY_WARNING_MB = 380
MEMORY_CRITICAL_MB = 430

# ---------------------------------------------------------------------------
# Startup tracking
# ---------------------------------------------------------------------------
_start_time: float = time.time()
_status_data: Dict[str, Any] = {
    "open_positions": 0,
    "last_scan": None,
    "next_weekly_scan": None,
    "last_morning_briefing": None,
    "last_error": None,
    "memory_warnings": 0,
    "emergency_mode": False,
}


def update_status(key: str, value: Any) -> None:
    """Update a status field (called by other modules)."""
    _status_data[key] = value


def get_status(key: str, default: Any = None) -> Any:
    """Read a status field."""
    return _status_data.get(key, default)


def get_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    try:
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return -1.0


def check_memory() -> Dict[str, Any]:
    """Check memory and return status dict.

    Returns
    -------
    Dict with ``memory_mb``, ``warning``, ``critical``, ``emergency_mode``.
    """
    mem = get_memory_mb()
    warning = mem > MEMORY_WARNING_MB
    critical = mem > MEMORY_CRITICAL_MB

    if warning and not _status_data.get("emergency_mode"):
        _status_data["memory_warnings"] = _status_data.get("memory_warnings", 0) + 1
        logger.warning(f"Memory warning: {mem:.0f} MB (threshold: {MEMORY_WARNING_MB} MB)")

    if critical:
        _status_data["emergency_mode"] = True
        logger.critical(
            f"Memory CRITICAL: {mem:.0f} MB — enabling emergency mode"
        )

    return {
        "memory_mb": mem,
        "warning": warning,
        "critical": critical,
        "emergency_mode": _status_data.get("emergency_mode", False),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    """Basic health check.

    Returns
    -------
    JSON with ``status``, ``uptime`` (seconds), ``memory_mb``.
    """
    uptime = round(time.time() - _start_time, 1)
    mem_info = check_memory()
    status = "degraded" if mem_info["emergency_mode"] else "ok"
    return JSONResponse(
        {
            "status": status,
            "service": "stocks-agent",
            "uptime_seconds": uptime,
            "memory_mb": mem_info["memory_mb"],
            "memory_warning": mem_info["warning"],
            "emergency_mode": mem_info["emergency_mode"],
        },
        status_code=200 if status == "ok" else 503,
    )


@app.get("/status")
async def status() -> JSONResponse:
    """Detailed bot status.

    Returns
    -------
    JSON with position count, last scan time, next run times, memory.
    """
    mem_info = check_memory()
    return JSONResponse({
        "status": "degraded" if mem_info["emergency_mode"] else "ok",
        "memory_mb": mem_info["memory_mb"],
        "memory_warning": mem_info["warning"],
        "uptime_seconds": round(time.time() - _start_time, 1),
        **_status_data,
    })


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_health_server() -> None:
    """Start the health check server (blocking). Run in a thread."""
    import uvicorn

    port = int(os.getenv("STOCKS_HEALTH_PORT", "8001"))
    logger.info(f"Starting health server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
