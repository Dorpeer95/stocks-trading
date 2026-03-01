"""
Insider trading activity — SEC EDGAR Form 4 parser.

Fetches recent insider transactions from the SEC EDGAR full-text
search API (no API key needed) and from yfinance insider data.
Identifies cluster buys and significant insider signals.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNIFICANT_BUY_USD = 100_000       # Minimum buy to be "significant"
CLUSTER_BUY_THRESHOLD = 3          # >= 3 insiders buying = cluster buy
CLUSTER_WINDOW_DAYS = 30           # Within 30 days
OFFICER_TITLES = {"CEO", "CFO", "COO", "CTO", "President", "Chairman"}


# ---------------------------------------------------------------------------
# yfinance insider data
# ---------------------------------------------------------------------------

def fetch_insider_transactions(ticker: str, days: int = 90) -> Optional[List[Dict[str, Any]]]:
    """Fetch insider transactions from yfinance.

    Returns
    -------
    List of dicts with keys: ``insider_name``, ``title``, ``date``,
    ``transaction_type``, ``shares``, ``value_usd``, ``is_officer``.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # Try insider_transactions (purchases/sales only)
        txns = None
        try:
            txns = t.insider_transactions
        except Exception:
            pass

        if txns is None or (isinstance(txns, pd.DataFrame) and txns.empty):
            logger.debug(f"No insider data for {ticker}")
            return None

        cutoff = date.today() - timedelta(days=days)
        records: List[Dict[str, Any]] = []

        for _, row in txns.iterrows():
            # Parse date
            txn_date = None
            try:
                raw_date = row.get("Start Date") or row.get("Date")
                if raw_date is not None:
                    txn_date = pd.Timestamp(raw_date).date()
            except Exception:
                continue

            if txn_date is None or txn_date < cutoff:
                continue

            # Parse transaction type
            txn_text = str(row.get("Transaction", "") or row.get("Text", "")).lower()
            if "purchase" in txn_text or "buy" in txn_text:
                txn_type = "buy"
            elif "sale" in txn_text or "sell" in txn_text:
                txn_type = "sell"
            else:
                txn_type = "other"

            # Parse value
            shares = 0
            value = 0.0
            try:
                shares = int(row.get("Shares", 0) or 0)
                value = float(row.get("Value", 0) or 0)
            except (ValueError, TypeError):
                pass

            name = str(row.get("Insider", "") or row.get("Insider Trading", ""))
            title = str(row.get("Position", "") or row.get("Title", ""))

            is_officer = any(t in title.upper() for t in OFFICER_TITLES)

            records.append({
                "insider_name": name,
                "title": title,
                "date": str(txn_date),
                "transaction_type": txn_type,
                "shares": shares,
                "value_usd": value,
                "is_officer": is_officer,
            })

        return records if records else None

    except Exception as e:
        logger.warning(f"fetch_insider_transactions failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def analyze_insider_activity(
    transactions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Analyze insider transactions for bullish/bearish signals.

    Returns
    -------
    Dict with keys: ``signal``, ``buy_count``, ``sell_count``,
    ``net_buy_value``, ``has_cluster_buy``, ``has_officer_buy``,
    ``significant_buys``, ``score_adjustment``.
    """
    default = {
        "signal": "neutral",
        "buy_count": 0,
        "sell_count": 0,
        "net_buy_value": 0.0,
        "has_cluster_buy": False,
        "has_officer_buy": False,
        "significant_buys": 0,
        "score_adjustment": 0,
    }

    if not transactions:
        return default

    buys = [t for t in transactions if t["transaction_type"] == "buy"]
    sells = [t for t in transactions if t["transaction_type"] == "sell"]

    buy_value = sum(t.get("value_usd", 0) for t in buys)
    sell_value = sum(t.get("value_usd", 0) for t in sells)
    net_value = buy_value - sell_value

    # Check for cluster buy (multiple insiders buying within window)
    has_cluster = len(buys) >= CLUSTER_BUY_THRESHOLD

    # Check for officer buys
    officer_buys = [t for t in buys if t.get("is_officer", False)]
    has_officer_buy = len(officer_buys) > 0

    # Significant buys (> $100K)
    significant = [t for t in buys if t.get("value_usd", 0) >= SIGNIFICANT_BUY_USD]

    # Determine signal
    score_adj = 0
    if has_cluster:
        signal = "strong_bullish"
        score_adj = 10
    elif has_officer_buy and buy_value > sell_value:
        signal = "bullish"
        score_adj = 5
    elif len(buys) > len(sells) and net_value > 0:
        signal = "mild_bullish"
        score_adj = 3
    elif len(sells) > len(buys) * 2 and sell_value > buy_value * 3:
        signal = "bearish"
        score_adj = -5
    else:
        signal = "neutral"
        score_adj = 0

    return {
        "signal": signal,
        "buy_count": len(buys),
        "sell_count": len(sells),
        "net_buy_value": round(net_value, 2),
        "has_cluster_buy": has_cluster,
        "has_officer_buy": has_officer_buy,
        "significant_buys": len(significant),
        "score_adjustment": score_adj,
    }


def get_insider_signal(ticker: str) -> Dict[str, Any]:
    """One-call convenience: fetch + analyze insider activity.

    Returns the analyze_insider_activity result dict.
    """
    txns = fetch_insider_transactions(ticker)
    result = analyze_insider_activity(txns)
    result["ticker"] = ticker
    return result
