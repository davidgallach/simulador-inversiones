"""DCA Puro and DCA Táctico, applied month-by-month to a price series.

Conventions:
- `prices` is a 1D array-like of monthly prices, length N (N >= 1). Index 0 is month 0.
- Both strategies invest the initial amount at price[0] in month 0. No monthly
  contribution is processed at month 0 by either strategy.
- From month 1 onward:
    * DCA Puro always invests `monthly` at price[t].
    * DCA Táctico looks at asset_return = price[t]/price[t-1] - 1.
        - return < 0: invest monthly + pool at price[t], empty the pool.
        - return >= 0: add monthly to pool, do not invest.
- Returned `value` series has length N: value[t] = shares_at_t * price[t]. The
  pool is intentionally NOT counted in the final value if it has leftover cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrategyResult:
    values: np.ndarray          # portfolio value at each month, length N
    invested: np.ndarray        # cumulative money deployed at each month, length N
    shares: np.ndarray          # cumulative shares held at each month, length N
    pool: np.ndarray            # cash sitting in the táctico pool, length N (Puro = zeros)
    total_invested: float       # final money actually deployed
    committed: float            # initial + monthly * (N-1)
    final_value: float          # values[-1]


def _as_prices(prices) -> np.ndarray:
    arr = np.asarray(prices, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("prices is empty")
    if np.any(arr <= 0):
        raise ValueError("prices must be strictly positive")
    return arr


def dca_puro(prices, initial: float, monthly: float) -> StrategyResult:
    p = _as_prices(prices)
    n = p.size
    shares = np.empty(n)
    invested = np.empty(n)

    # Month 0: only the initial deposit.
    shares[0] = initial / p[0] if initial > 0 else 0.0
    invested[0] = float(initial)

    # Months 1..N-1: deploy the monthly contribution at that month's price.
    for t in range(1, n):
        added_shares = monthly / p[t] if monthly > 0 else 0.0
        shares[t] = shares[t - 1] + added_shares
        invested[t] = invested[t - 1] + monthly

    values = shares * p
    pool = np.zeros(n)
    committed = float(initial) + float(monthly) * (n - 1)
    return StrategyResult(
        values=values,
        invested=invested,
        shares=shares,
        pool=pool,
        total_invested=float(invested[-1]),
        committed=committed,
        final_value=float(values[-1]),
    )


def dca_tactico(prices, initial: float, monthly: float) -> StrategyResult:
    p = _as_prices(prices)
    n = p.size
    shares = np.empty(n)
    invested = np.empty(n)
    pool = np.empty(n)

    # Month 0: only the initial deposit. Pool stays empty.
    shares[0] = initial / p[0] if initial > 0 else 0.0
    invested[0] = float(initial)
    pool[0] = 0.0

    for t in range(1, n):
        asset_return = p[t] / p[t - 1] - 1.0
        if asset_return < 0.0:
            cash_to_deploy = pool[t - 1] + monthly
            added_shares = cash_to_deploy / p[t] if cash_to_deploy > 0 else 0.0
            shares[t] = shares[t - 1] + added_shares
            invested[t] = invested[t - 1] + cash_to_deploy
            pool[t] = 0.0
        else:
            shares[t] = shares[t - 1]
            invested[t] = invested[t - 1]
            pool[t] = pool[t - 1] + monthly

    values = shares * p
    committed = float(initial) + float(monthly) * (n - 1)
    return StrategyResult(
        values=values,
        invested=invested,
        shares=shares,
        pool=pool,
        total_invested=float(invested[-1]),
        committed=committed,
        final_value=float(values[-1]),
    )
