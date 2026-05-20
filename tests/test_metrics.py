"""Tests for metrics: CAGR/IRR, drawdown, probability helper."""
from __future__ import annotations

import math

import numpy as np
import pytest

from metrics import (
    cagr,
    cashflows,
    max_drawdown,
    prob_loss_given_ruin,
    prob_profit_above,
    prob_ruin_path,
    volatility_annualized,
)
from strategies import StrategyResult, dca_puro


def _make_result(values, invested) -> StrategyResult:
    values = np.asarray(values, dtype=float)
    invested = np.asarray(invested, dtype=float)
    return StrategyResult(
        values=values,
        invested=invested,
        shares=np.zeros_like(values),
        pool=np.zeros_like(values),
        total_invested=float(invested[-1]),
        committed=float(invested[-1]),
        final_value=float(values[-1]),
    )


# --------- CAGR / IRR ---------

def test_cagr_lump_sum_known_case():
    """Invest 100 at month 0, get 110 back at month 12.
    Monthly IRR = (110/100)^(1/12) - 1; annualized = 10%."""
    n = 13  # months 0..12 inclusive
    invested = np.concatenate([[100.0], np.full(n - 1, 100.0)])
    values = np.zeros(n)
    values[-1] = 110.0
    res = _make_result(values, invested)
    result = cagr(res)
    assert math.isclose(result, 0.10, abs_tol=1e-6)


def test_cagr_puro_constant_price_is_zero():
    prices = np.full(24, 100.0)
    res = dca_puro(prices, initial=1000.0, monthly=200.0)
    # Final value == total invested → 0% return.
    assert abs(cagr(res)) < 1e-6


def test_cashflows_signs():
    prices = np.array([100.0, 100.0, 100.0])
    res = dca_puro(prices, initial=1000.0, monthly=200.0)
    cf = cashflows(res)
    # Three months: -1000, -200, -200 + final_value
    assert cf[0] == pytest.approx(-1000.0)
    assert cf[1] == pytest.approx(-200.0)
    assert cf[2] == pytest.approx(-200.0 + res.final_value)


# --------- Drawdown ---------

def test_max_drawdown_known_series():
    """Peak at 2, trough at 0.5 → drawdown = (0.5 - 2) / 2 = -0.75."""
    series = [1.0, 2.0, 0.5, 1.0]
    assert max_drawdown(series) == pytest.approx(-0.75)


def test_max_drawdown_monotonically_increasing_is_zero():
    series = np.linspace(1.0, 2.0, 10)
    assert max_drawdown(series) == pytest.approx(0.0)


def test_max_drawdown_handles_leading_zeros():
    """Early months may have V=0 before any deposit; that shouldn't blow up."""
    series = [0.0, 0.0, 100.0, 50.0, 75.0]
    # Peak after deposits is 100, trough 50 → -0.5.
    assert max_drawdown(series) == pytest.approx(-0.5)


# --------- Volatility sanity ---------

def test_volatility_constant_value_is_zero():
    prices = np.full(24, 100.0)
    res = dca_puro(prices, initial=1000.0, monthly=200.0)
    vol = volatility_annualized(res)
    assert vol == pytest.approx(0.0, abs=1e-9)


# --------- Probability helper ---------

def test_prob_profit_above_known_array():
    """final_values=[100, 200, 300], total_invested=100, x=50 →
    we need final - invested > 50, i.e. final > 150 → 2 of 3 → 2/3."""
    arr = np.array([100.0, 200.0, 300.0])
    assert prob_profit_above(arr, total_invested=100.0, x=50.0) == pytest.approx(2 / 3)


def test_prob_profit_above_strict_inequality():
    """Boundary case: strictly greater than X."""
    arr = np.array([100.0, 200.0])
    # final - invested = [0, 100]; x=100 → strict > so zero pass.
    assert prob_profit_above(arr, total_invested=100.0, x=100.0) == pytest.approx(0.0)
    # x=99 → one passes.
    assert prob_profit_above(arr, total_invested=100.0, x=99.0) == pytest.approx(0.5)


# --------- Path-based ruin ---------

def test_prob_ruin_path_detects_intermediate_dip():
    """A sim that dips below the threshold and recovers still counts as ruin."""
    traj = np.array([
        [1000.0, 1500.0, 2000.0],   # never dips → no ruin
        [1000.0,    0.5, 2000.0],   # dips at t=1 and recovers → ruin
        [1000.0, 1000.0,    0.2],   # dips at the end → ruin
    ])
    assert prob_ruin_path(traj, threshold=1.0) == pytest.approx(2 / 3)


def test_prob_ruin_path_excludes_month_zero():
    """If initial = 0, V_0 = 0 must not be flagged. Only t >= 1 counts."""
    traj = np.array([
        [0.0, 100.0, 200.0],   # never < 1 after t=0 → no ruin
        [0.0,   0.5, 200.0],   # dips at t=1 → ruin
    ])
    assert prob_ruin_path(traj, threshold=1.0) == pytest.approx(0.5)


def test_prob_loss_given_ruin_conditional():
    """Among ruin sims, fraction ending with final < total_invested."""
    traj = np.array([
        [1000.0,    0.5, 1500.0],   # ruin, final 1500 > invested 1200 → no loss
        [1000.0,    0.5,  500.0],   # ruin, final 500 < 1200 → loss
        [1000.0, 1000.0,  800.0],   # no ruin (dropped, but never < 1)
    ])
    finals = traj[:, -1]
    invested = np.array([1200.0, 1200.0, 1200.0])
    # 2 sims hit ruin, 1 has a loss → 0.5.
    assert prob_loss_given_ruin(traj, finals, invested) == pytest.approx(0.5)


def test_prob_loss_given_ruin_no_ruin_is_nan():
    """If no sim hits ruin, the conditional is undefined."""
    traj = np.array([[1000.0, 1100.0, 1200.0]])
    finals = traj[:, -1]
    assert math.isnan(prob_loss_given_ruin(traj, finals, 1000.0))
