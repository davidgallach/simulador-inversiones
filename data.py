"""Historical price download via yfinance, resampled to first business day of each month."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def looks_like_isin(symbol: str) -> bool:
    return bool(ISIN_PATTERN.match(symbol.strip().upper()))


class DataError(Exception):
    """Raised when historical data cannot be obtained for the requested ticker."""


@dataclass(frozen=True)
class PriceData:
    ticker: str
    monthly: pd.Series  # indexed by first available trading day of each month


def _resolve_isin(isin: str) -> str | None:
    """Best-effort ISIN → Yahoo ticker. yfinance has no first-class ISIN lookup;
    we try the symbol directly (some ETFs are listed under their ISIN) and bail
    out if nothing comes back. The caller is expected to surface a Spanish error
    asking for a Yahoo ticker if this returns None."""
    try:
        info = yf.Ticker(isin)
        hist = info.history(period="5d", auto_adjust=True)
        if hist is not None and not hist.empty:
            return isin
    except Exception:
        return None
    return None


def _to_monthly(daily: pd.DataFrame) -> pd.Series:
    """Pick the first available trading day per month from a daily OHLC frame."""
    if "Close" in daily.columns:
        close = daily["Close"]
    else:
        # auto_adjust=True returns 'Close' but be defensive
        close = daily.iloc[:, 0]

    close = close.dropna()
    if close.empty:
        return close

    # Ensure tz-naive index for stable grouping/display
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    grouped = close.groupby([close.index.year, close.index.month])
    first_per_month = grouped.head(1)
    first_per_month.name = "price"
    return first_per_month.sort_index()


def fetch_prices(symbol: str, years: int) -> PriceData:
    """Download `years` of daily prices, resample to monthly (first trading day),
    and return a PriceData. Raises DataError on failure or insufficient history."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise DataError("Símbolo vacío.")

    if looks_like_isin(symbol):
        resolved = _resolve_isin(symbol)
        if resolved is None:
            raise DataError("ISIN_UNRESOLVED")
        symbol = resolved

    period = f"{max(years, 1)}y"
    try:
        daily = yf.download(
            symbol,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise DataError(f"NO_DATA: {exc}") from exc

    if daily is None or daily.empty:
        raise DataError("NO_DATA")

    # yfinance with a single ticker may return a MultiIndex on columns; flatten.
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    monthly = _to_monthly(daily)
    if monthly.empty:
        raise DataError("NO_DATA")

    months_needed = years * 12
    if len(monthly) < months_needed:
        raise DataError("INSUFFICIENT_HISTORY")

    # Keep exactly the last `months_needed` months so the simulation window
    # matches what the user asked for.
    monthly = monthly.iloc[-months_needed:]
    return PriceData(ticker=symbol, monthly=monthly)
