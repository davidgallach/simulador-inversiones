"""Historical price download via yfinance, resampled to first business day of each month.

Also synthesizes daily-reset leveraged price series (e.g. 3× SPY) from a base
asset's daily prices. See `synthesize_leveraged_prices` for the mechanics."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Defensive floor on the daily leveraged multiplier (1 + L·r) so a pathological
# day cannot make the synthetic price non-positive (which would break monthly
# return derivation downstream). With real broad-index data this floor never
# fires; it's a pure safety net, not wipeout behavior.
_LEV_DAILY_FLOOR = 1e-4
MIN_LEVERAGE = -10
MAX_LEVERAGE = 10


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


def _monthly_from_close(close: pd.Series) -> pd.Series:
    """Pick the first available trading day per month from a daily close series."""
    close = close.dropna()
    if close.empty:
        return close
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    grouped = close.groupby([close.index.year, close.index.month])
    first_per_month = grouped.head(1)
    first_per_month.name = "price"
    return first_per_month.sort_index()


def _fetch_daily_close(symbol: str, years: int) -> tuple[str, pd.Series]:
    """Resolve the symbol (ISIN if needed), download daily prices via yfinance,
    return (resolved_symbol, daily Close series). Raises DataError on failure."""
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

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    close = daily["Close"] if "Close" in daily.columns else daily.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise DataError("NO_DATA")
    return symbol, close


def _trim_to_window(monthly: pd.Series, years: int) -> pd.Series:
    if monthly.empty:
        raise DataError("NO_DATA")
    months_needed = years * 12
    if len(monthly) < months_needed:
        raise DataError("INSUFFICIENT_HISTORY")
    return monthly.iloc[-months_needed:]


def synthesize_leveraged_prices(
    base_daily_close: pd.Series, leverage: int
) -> pd.Series:
    """Synthetic daily-reset L× price series built from a base daily close series.

    Each day: r_lev[t] = L · r_base[t]; price compounds as
        P_lev[t] = P_lev[t-1] · (1 + r_lev[t])
    starting at the base's first price. The daily multiplier is floored at
    `_LEV_DAILY_FLOOR` to guard against pathological negative prices; this
    floor never fires on real broad-index data with |L| <= 10."""
    if not isinstance(leverage, (int, np.integer)):
        raise ValueError("leverage debe ser entero")
    if not (MIN_LEVERAGE <= int(leverage) <= MAX_LEVERAGE):
        raise ValueError(
            f"leverage fuera de rango ({MIN_LEVERAGE}..{MAX_LEVERAGE})"
        )

    base = base_daily_close.dropna().astype(float)
    if base.size < 2:
        raise DataError("NO_DATA")

    r = base.pct_change().fillna(0.0).to_numpy()
    multiplier = np.maximum(1.0 + int(leverage) * r, _LEV_DAILY_FLOOR)
    prices = float(base.iloc[0]) * np.cumprod(multiplier)
    return pd.Series(prices, index=base.index, name="price")


def fetch_prices(symbol: str, years: int) -> PriceData:
    """Download `years` of daily prices, resample to monthly (first trading day),
    and return a PriceData. Raises DataError on failure or insufficient history."""
    resolved, close = _fetch_daily_close(symbol, years)
    monthly = _trim_to_window(_monthly_from_close(close), years)
    return PriceData(ticker=resolved, monthly=monthly)


def fetch_leveraged_prices(
    base_symbol: str, leverage: int, years: int
) -> PriceData:
    """Download the base asset, synthesize a daily L× series, then monthly-resample.
    The display ticker is annotated, e.g. 'SPY × +3'."""
    resolved, close = _fetch_daily_close(base_symbol, years)
    leveraged_daily = synthesize_leveraged_prices(close, leverage)
    monthly = _trim_to_window(_monthly_from_close(leveraged_daily), years)
    display = f"{resolved} × {int(leverage):+d}"
    return PriceData(ticker=display, monthly=monthly)
