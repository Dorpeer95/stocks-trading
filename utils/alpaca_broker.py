"""
Alpaca Broker Integration

Handles communicating with the Alpaca Trade API to submit
market and limit orders autonomously.
"""

import logging
import os
from typing import Any, Dict, Optional

from alpaca_trade_api.rest import REST
from alpaca_trade_api.entity import Order

logger = logging.getLogger(__name__)

# Config
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
DRY_RUN = os.getenv("STOCKS_DRY_RUN", "false").lower() == "true"

# Initialize REST client
alpaca_client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try:
        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        alpaca_client = REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=base_url
        )
        logger.info(f"[Alpaca] Client initialized ({'paper' if ALPACA_PAPER else 'live'})")
    except Exception as e:
        logger.error(f"[Alpaca] Failed to initialize client: {e}")


def execute_trade(
    ticker: str,
    action: str,
    shares: float,
    order_type: str = "market",
    limit_price: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Submits a trade order to Alpaca.

    Parameters
    ----------
    ticker : str
        The stock symbol (e.g., 'AAPL').
    action : str
        'buy' or 'sell'.
    shares : float
        Number of shares/fractional shares to trade.
    order_type : str
        'market', 'limit', etc. Default 'market'.
    limit_price : float, optional
        Required only if order_type is 'limit'.

    Returns
    -------
    dict
        The parsed order response from Alpaca, or None if failed.
    """
    if DRY_RUN:
        logger.info(f"[DRY-RUN] Would submit Alpaca order: {action.upper()} {shares} shares of {ticker} ({order_type})")
        return {
            "id": "dry_run_id_12345",
            "symbol": ticker,
            "qty": shares,
            "side": action.lower(),
            "type": order_type,
            "status": "filled"
        }

    if not alpaca_client:
        logger.warning(f"[Alpaca] Attempted {action} order for {ticker} but client is not configured!")
        return None

    try:
        # Ensure shares amount is handled correctly for submitting.
        # Alpaca supports fractional shares for market orders, but it's best to pass as float or string.
        qty_val = float(shares)

        kwargs = {
            "symbol": ticker,
            "qty": qty_val,
            "side": action.lower(),
            "type": order_type,
            "time_in_force": "gtc" if order_type == "limit" else "day",
        }

        if order_type == "limit" and limit_price:
            kwargs["limit_price"] = round(float(limit_price), 2)

        order: Order = alpaca_client.submit_order(**kwargs)

        logger.info(
            f"[Alpaca] Submitted order: {action.upper()} {qty_val} {ticker} "
            f"| Type: {order_type} | Status: {order.status}"
        )

        return order._raw

    except Exception as e:
        logger.error(f"[Alpaca] Error submitting {action} order for {ticker}: {e}")
        return None


def get_account_status() -> Optional[Dict[str, Any]]:
    """Fetches high-level account status from Alpaca (Buying power, equity, etc)."""
    if not alpaca_client:
        return None
    try:
        account = alpaca_client.get_account()
        return account._raw
    except Exception as e:
        logger.error(f"[Alpaca] Error fetching account status: {e}")
        return None
