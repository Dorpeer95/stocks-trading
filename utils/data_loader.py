"""
Data loader — fetches price data, fundamentals, and macro indicators.

Wraps yfinance with retry logic, caching, and error handling.
All public functions return ``None`` on failure.
"""

import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _cache_key(prefix: str, *args: Any) -> str:
    return f"{prefix}:{'|'.join(str(a) for a in args)}"


def _get_cached(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    if datetime.utcnow() > entry["expires"]:
        del _cache[key]
        return None
    return entry["value"]


def _set_cached(key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    _cache[key] = {
        "value": value,
        "expires": datetime.utcnow() + timedelta(seconds=ttl),
    }


def clear_cache() -> None:
    """Clear all cached data."""
    _cache.clear()
    logger.info("Data loader cache cleared")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _retry(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    """Retry a function with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_retries} "
                            f"failed: {e} — retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} "
                            f"attempts: {last_err}"
                        )
            return None

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# S&P 500 universe
# ---------------------------------------------------------------------------

@_retry(max_retries=3)
def fetch_sp500_list() -> Optional[List[Dict[str, str]]]:
    """Fetch the current S&P 500 constituents from Wikipedia.

    Returns
    -------
    List of dicts with keys ``ticker``, ``company_name``, ``sector``,
    ``industry``.  Returns ``None`` on failure.
    """
    cache_key = _cache_key("sp500_list")
    cached = _get_cached(cache_key)
    if cached is not None:
        logger.debug("fetch_sp500_list: returning cached")
        return cached

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]

    # Wikipedia table columns may change; handle common variants
    col_map = {
        "Symbol": "ticker",
        "Ticker": "ticker",
        "Security": "company_name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "industry",
    }

    # Rename only columns that exist
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    required = {"ticker", "company_name", "sector", "industry"}
    if not required.issubset(df.columns):
        logger.error(
            f"fetch_sp500_list: missing columns. Got {list(df.columns)}"
        )
        return None

    # Clean tickers (e.g. BRK.B → BRK-B for yfinance)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    result = df[["ticker", "company_name", "sector", "industry"]].to_dict(
        orient="records"
    )

    _set_cached(cache_key, result, ttl=86400)  # 24-hour cache
    logger.info(f"Fetched S&P 500 list: {len(result)} stocks")
    return result


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

@_retry(max_retries=3)
def fetch_price_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV price data for a single ticker.

    Parameters
    ----------
    ticker : Stock symbol (e.g. ``'AAPL'``).
    period : yfinance period string (``'1d'``, ``'5d'``, ``'1mo'``, ``'1y'``, etc.).
    interval : Bar interval (``'1d'``, ``'1h'``, etc.).

    Returns
    -------
    DataFrame with ``Open, High, Low, Close, Volume`` columns,
    or ``None`` on failure.
    """
    cache_key = _cache_key("price", ticker, period, interval)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)

    if df is None or df.empty:
        logger.warning(f"No price data for {ticker}")
        return None

    # Standardise column names (yfinance sometimes adds extras)
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Drop rows where Close is NaN
    df = df.dropna(subset=["Close"])

    if df.empty:
        logger.warning(f"Price data for {ticker} was all NaN")
        return None

    _set_cached(cache_key, df)
    logger.debug(f"Fetched price data for {ticker}: {len(df)} bars")
    return df


@_retry(max_retries=3)
def fetch_batch_prices(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> Optional[Dict[str, pd.DataFrame]]:
    """Batch-download OHLCV data for many tickers in one API call.

    Uses ``yf.download`` for efficiency (single HTTP request).

    Returns
    -------
    Dict mapping ticker → DataFrame, or ``None`` on failure.
    Tickers that failed silently are omitted from the result.
    """
    if not tickers:
        return {}

    cache_key = _cache_key("batch", ",".join(sorted(tickers)), period)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    logger.info(
        f"Batch downloading {len(tickers)} tickers, period={period}"
    )

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if data is None or data.empty:
        logger.error("Batch download returned empty data")
        return None

    result: Dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        # yf.download with 1 ticker returns flat columns
        ticker = tickers[0]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        df = data[keep].dropna(subset=["Close"] if "Close" in keep else [])
        if not df.empty:
            result[ticker] = df
    else:
        for ticker in tickers:
            try:
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].copy()
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].dropna(subset=["Close"] if "Close" in keep else [])
                if not df.empty:
                    result[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to extract {ticker} from batch: {e}")

    logger.info(f"Batch download complete: {len(result)}/{len(tickers)} tickers OK")
    _set_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

MACRO_TICKERS = {
    "vix": "^VIX",
    "oil": "CL=F",
    "gold": "GC=F",
    "dollar": "DX-Y.NYB",
    "treasury_10y": "^TNX",
    "spy": "SPY",
}


@_retry(max_retries=3)
def fetch_macro_data(period: str = "3mo") -> Optional[Dict[str, Dict[str, float]]]:
    """Fetch key macro indicators.

    Returns
    -------
    Dict with keys ``vix``, ``oil``, ``gold``, ``dollar``, ``treasury_10y``,
    ``spy``.  Each value is a dict with ``current``, ``prev_close``,
    ``change_pct``.  Returns ``None`` on failure.
    """
    cache_key = _cache_key("macro", period)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Dict[str, float]] = {}

    for name, symbol in MACRO_TICKERS.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period=period)
            if hist is None or len(hist) < 2:
                logger.warning(f"Macro data missing for {name} ({symbol})")
                continue

            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change_pct = round(((current - prev) / prev) * 100, 2) if prev != 0 else 0.0

            result[name] = {
                "current": round(current, 2),
                "prev_close": round(prev, 2),
                "change_pct": change_pct,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch macro {name}: {e}")

    if not result:
        return None

    _set_cached(cache_key, result)
    logger.info(f"Fetched macro data: {list(result.keys())}")
    return result


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

@_retry(max_retries=2)
def fetch_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch fundamental data for a single ticker.

    Returns
    -------
    Dict with keys: ``market_cap_b``, ``pe_ratio``, ``forward_pe``,
    ``sector``, ``industry``, ``avg_volume_50d``, ``short_pct_float``,
    ``earnings_date``, ``dividend_yield``, ``revenue_growth``,
    ``profit_margin``.
    """
    cache_key = _cache_key("fundamentals", ticker)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info

        if not info or info.get("regularMarketPrice") is None:
            logger.warning(f"No fundamental data for {ticker}")
            return None

        market_cap = info.get("marketCap")
        market_cap_b = round(market_cap / 1e9, 2) if market_cap else None

        result = {
            "ticker": ticker,
            "market_cap_b": market_cap_b,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "avg_volume_50d": info.get("averageVolume"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "dividend_yield": info.get("dividendYield"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
        }

        # Earnings date
        try:
            calendar = t.calendar
            if calendar is not None and not calendar.empty:
                if "Earnings Date" in calendar.index:
                    result["earnings_date"] = str(calendar.loc["Earnings Date"].iloc[0])
                elif isinstance(calendar, dict) and "Earnings Date" in calendar:
                    dates = calendar["Earnings Date"]
                    result["earnings_date"] = str(dates[0]) if dates else None
                else:
                    result["earnings_date"] = None
            else:
                result["earnings_date"] = None
        except Exception:
            result["earnings_date"] = None

        _set_cached(cache_key, result, ttl=3600)  # 1 hour cache
        logger.debug(f"Fetched fundamentals for {ticker}")
        return result

    except Exception as e:
        logger.error(f"fetch_fundamentals failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Sector ETF mapping
# ---------------------------------------------------------------------------

SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def get_sector_etf(sector: str) -> Optional[str]:
    """Return the SPDR sector ETF ticker for a GICS sector name."""
    return SECTOR_ETF_MAP.get(sector)
