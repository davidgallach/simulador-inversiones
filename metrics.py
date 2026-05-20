"""Performance metrics for a DCA strategy run.

All metrics operate on a `StrategyResult` (see strategies.py) or on raw arrays.

Return conventions for the portfolio:
- We use *time-weighted* monthly returns to strip out cashflow distortion:
      r_t = (V_t - C_t) / V_{t-1} - 1
  where C_t is the cash deployed at month t (the start of the period). This
  isolates the market return on shares that were already held.
- Annualizations use 12 monthly periods.
"""
from __future__ import annotations

import math

import numpy as np
import numpy_financial as npf

from strategies import StrategyResult


# --------- Cashflows ---------

def cashflows(result: StrategyResult) -> np.ndarray:
    """Per-month cashflows from the investor's perspective.

    Inflows to the portfolio are negative (money leaving the investor's wallet).
    The final value is a positive inflow back to the investor at the last month.
    Money sitting in the táctico pool is NOT counted (spec: leftover pool is ignored).
    """
    deployed_per_month = np.diff(result.invested, prepend=0.0)
    cf = -deployed_per_month.astype(float)
    cf[-1] += result.final_value
    return cf


# --------- IRR / CAGR ---------

def cagr(result: StrategyResult) -> float:
    """Money-weighted return, annualized. Returns float('nan') if undefined."""
    cf = cashflows(result)
    # IRR needs both signs in the cashflow stream.
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return float("nan")
    monthly_rate = npf.irr(cf)
    if monthly_rate is None or not np.isfinite(monthly_rate):
        return float("nan")
    return (1.0 + monthly_rate) ** 12 - 1.0


# --------- Time-weighted monthly returns ---------

def time_weighted_returns(result: StrategyResult) -> np.ndarray:
    """Monthly portfolio returns adjusted for cashflows. Length = N-1."""
    values = result.values
    deployed = np.diff(result.invested, prepend=0.0)
    n = values.size
    if n < 2:
        return np.array([])
    rets = np.empty(n - 1)
    for t in range(1, n):
        v_prev = values[t - 1]
        if v_prev <= 0:
            rets[t - 1] = 0.0
        else:
            rets[t - 1] = (values[t] - deployed[t]) / v_prev - 1.0
    return rets


def total_return(result: StrategyResult) -> float:
    """Simple total return: (final_value - total_invested) / total_invested."""
    if result.total_invested <= 0:
        return float("nan")
    return result.final_value / result.total_invested - 1.0


def twr_cumulative(result: StrategyResult) -> float:
    """Cumulative time-weighted return: product of (1 + monthly TWR) - 1."""
    rets = time_weighted_returns(result)
    if rets.size == 0:
        return float("nan")
    return float(np.prod(1.0 + rets) - 1.0)


def twr_annualized(result: StrategyResult) -> float:
    """Annualized TWR. With N monthly returns spanning N months elapsed:
    annual = (1 + cumulative_twr)^(12/N) - 1."""
    rets = time_weighted_returns(result)
    if rets.size == 0:
        return float("nan")
    cum = float(np.prod(1.0 + rets) - 1.0)
    return (1.0 + cum) ** (12.0 / rets.size) - 1.0


def volatility_annualized(result: StrategyResult) -> float:
    rets = time_weighted_returns(result)
    if rets.size < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(12))


def sharpe_ratio(result: StrategyResult, risk_free_annual: float = 0.0) -> float:
    """Annualized Sharpe ratio. Risk-free defaults to 0 (per spec)."""
    rets = time_weighted_returns(result)
    if rets.size < 2:
        return float("nan")
    std = np.std(rets, ddof=1)
    if std == 0:
        return float("nan")
    rf_monthly = (1.0 + risk_free_annual) ** (1.0 / 12.0) - 1.0
    excess = rets - rf_monthly
    return float(np.mean(excess) / std * math.sqrt(12))


# --------- Drawdown ---------

def max_drawdown(values) -> float:
    """Maximum drawdown of a value series, as a negative fraction (e.g. -0.42).

    Computed as min over t of (V_t - running_max_{<=t}) / running_max_{<=t}.
    Returns 0.0 if the series never falls below its running peak.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    # Ignore leading zeros that would blow up the ratio.
    running_max = np.maximum.accumulate(v)
    safe = running_max > 0
    if not np.any(safe):
        return 0.0
    dd = np.zeros_like(v)
    dd[safe] = (v[safe] - running_max[safe]) / running_max[safe]
    return float(dd.min())


# --------- Probability helpers (used by Monte Carlo tab) ---------

def prob_below(final_values, threshold: float) -> float:
    arr = np.asarray(final_values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr < threshold))


def prob_profit_above(final_values, total_invested: float, x: float) -> float:
    """Fraction of simulations where final_value - total_invested > x."""
    arr = np.asarray(final_values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.mean((arr - total_invested) > x))


def percentiles(final_values, percentile_list=(10, 20, 30, 40, 60, 70, 80, 90)) -> dict:
    arr = np.asarray(final_values, dtype=float).ravel()
    if arr.size == 0:
        return {p: float("nan") for p in percentile_list}
    qs = np.percentile(arr, percentile_list)
    return {p: float(q) for p, q in zip(percentile_list, qs)}
