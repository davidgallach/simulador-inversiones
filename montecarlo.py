"""Vectorized bootstrap Monte Carlo for DCA Puro and DCA Táctico.

Convention (per spec): we sample `n_months = años_simular × 12` monthly returns
with replacement from the historical monthly returns and build a synthetic price
path starting at 1.0. The resulting path therefore has `n_months + 1` evaluation
dates (month 0 .. month n_months), and the strategies are run over that path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


N_SIMS = 10_000
PERCENTILES_FAN = (10, 30, 50, 70, 90)


@dataclass(frozen=True)
class StrategySim:
    final_values: np.ndarray           # shape (n_sims,)
    total_invested: np.ndarray         # shape (n_sims,) — money actually deployed per sim
    percentile_bands: dict             # {p: ndarray of length n_steps}
    trajectories: np.ndarray           # shape (n_sims, n_steps) — full portfolio paths
    invested_trajectories: np.ndarray  # shape (n_sims, n_steps) — cumulative € deployed
    n_steps: int


@dataclass(frozen=True)
class SimResult:
    puro: StrategySim
    tactico: StrategySim
    n_sims: int
    n_months: int                      # number of sampled returns (path = n_months + 1 long)


def monthly_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float).ravel()
    if p.size < 2:
        return np.array([])
    return p[1:] / p[:-1] - 1.0


def _vectorized_puro(prices: np.ndarray, initial: float, monthly: float):
    """prices shape (n_sims, n_steps). Returns (values, invested) same shape."""
    n_sims, n_steps = prices.shape
    shares = np.empty((n_sims, n_steps))
    invested = np.empty((n_sims, n_steps))

    shares[:, 0] = initial / prices[:, 0] if initial > 0 else 0.0
    invested[:, 0] = initial

    if n_steps > 1:
        if monthly > 0:
            increments = monthly / prices[:, 1:]
            shares[:, 1:] = shares[:, [0]] + np.cumsum(increments, axis=1)
        else:
            shares[:, 1:] = shares[:, [0]]
        invested[:, 1:] = initial + monthly * np.arange(1, n_steps)

    values = shares * prices
    return values, invested


def _vectorized_tactico(
    prices: np.ndarray,
    initial: float,
    monthly: float,
    progress_cb: Callable[[float], None] | None = None,
):
    """prices shape (n_sims, n_steps). Returns (values, invested) same shape.

    The decision rule is sequential in t but vectorized across sims:
    O(n_sims) work per time step, n_steps iterations.
    """
    n_sims, n_steps = prices.shape
    shares = np.empty((n_sims, n_steps))
    invested = np.empty((n_sims, n_steps))

    shares[:, 0] = initial / prices[:, 0] if initial > 0 else 0.0
    invested[:, 0] = initial
    pool = np.zeros(n_sims)
    cum_shares = shares[:, 0].copy()
    cum_invested = invested[:, 0].copy()

    # Asset returns shape (n_sims, n_steps - 1): return[:, t-1] is for transition (t-1)->t.
    if n_steps > 1:
        asset_returns = prices[:, 1:] / prices[:, :-1] - 1.0

    # Report progress every ~5% of the time steps.
    report_every = max(1, (n_steps - 1) // 20)

    for t in range(1, n_steps):
        r = asset_returns[:, t - 1]
        invest_mask = r < 0.0

        cash_deployed = np.where(invest_mask, pool + monthly, 0.0)
        added_shares = np.where(invest_mask, cash_deployed / prices[:, t], 0.0)
        cum_shares = cum_shares + added_shares
        cum_invested = cum_invested + cash_deployed
        pool = np.where(invest_mask, 0.0, pool + monthly)

        shares[:, t] = cum_shares
        invested[:, t] = cum_invested

        if progress_cb is not None and (t % report_every == 0 or t == n_steps - 1):
            progress_cb(t / (n_steps - 1))

    values = shares * prices
    return values, invested


def _bands(values: np.ndarray, percentiles=PERCENTILES_FAN) -> dict:
    """Per-time-step percentiles across simulations. values shape (n_sims, n_steps)."""
    out = {}
    for p in percentiles:
        out[int(p)] = np.percentile(values, p, axis=0)
    return out


def simulate(
    historical_monthly_returns: np.ndarray,
    initial: float,
    monthly: float,
    n_months: int,
    n_sims: int = N_SIMS,
    seed: int | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> SimResult:
    """Run the full Monte Carlo.

    progress_cb(fraction) is called periodically with values in [0, 1]. The
    fraction represents combined progress across Puro and Táctico.
    """
    if historical_monthly_returns.size == 0:
        raise ValueError("No hay rendimientos históricos para muestrear.")
    if n_months < 1:
        raise ValueError("n_months debe ser >= 1")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, historical_monthly_returns.size, size=(n_sims, n_months))
    sampled_returns = historical_monthly_returns[idx]

    # Price paths shape (n_sims, n_months + 1), starting at 1.0.
    n_steps = n_months + 1
    prices = np.empty((n_sims, n_steps))
    prices[:, 0] = 1.0
    prices[:, 1:] = np.exp(np.cumsum(np.log1p(sampled_returns), axis=1))

    if progress_cb is not None:
        progress_cb(0.05)

    values_puro, invested_puro = _vectorized_puro(prices, initial, monthly)
    if progress_cb is not None:
        progress_cb(0.45)  # Puro is fast; reserve most of the bar for Táctico.

    def _tactico_cb(frac: float) -> None:
        if progress_cb is not None:
            progress_cb(0.45 + 0.55 * frac)

    values_tactico, invested_tactico = _vectorized_tactico(
        prices, initial, monthly, progress_cb=_tactico_cb
    )

    puro_sim = StrategySim(
        final_values=values_puro[:, -1],
        total_invested=invested_puro[:, -1],
        percentile_bands=_bands(values_puro),
        trajectories=values_puro,
        invested_trajectories=invested_puro,
        n_steps=n_steps,
    )
    tactico_sim = StrategySim(
        final_values=values_tactico[:, -1],
        total_invested=invested_tactico[:, -1],
        percentile_bands=_bands(values_tactico),
        trajectories=values_tactico,
        invested_trajectories=invested_tactico,
        n_steps=n_steps,
    )
    return SimResult(
        puro=puro_sim,
        tactico=tactico_sim,
        n_sims=n_sims,
        n_months=n_months,
    )
