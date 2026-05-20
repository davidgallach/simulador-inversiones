"""Tests for DCA Puro and DCA Táctico strategies."""
from __future__ import annotations

import numpy as np
import pytest

from strategies import dca_puro, dca_tactico


# ---------- DCA Puro ----------

def test_puro_constant_price_final_value_equals_total_invested():
    prices = np.full(12, 100.0)
    res = dca_puro(prices, initial=1000.0, monthly=200.0)
    expected_invested = 1000.0 + 200.0 * 11
    assert res.total_invested == pytest.approx(expected_invested)
    assert res.final_value == pytest.approx(expected_invested)


def test_puro_monotonically_increasing_final_value_exceeds_invested():
    prices = np.linspace(100.0, 200.0, 24)
    res = dca_puro(prices, initial=1000.0, monthly=200.0)
    # Every share bought at <= 200 ends up worth 200, so final value > total invested.
    assert res.final_value > res.total_invested
    # And we deployed initial + 23 monthlies.
    assert res.total_invested == pytest.approx(1000.0 + 200.0 * 23)


# ---------- DCA Táctico ----------

def test_tactico_month0_only_invests_initial():
    """Month 0 must invest only the initial deposit. The monthly contribution
    is NOT added to the portfolio and NOT added to the pool at month 0."""
    prices = np.array([100.0, 110.0])
    res = dca_tactico(prices, initial=1000.0, monthly=200.0)
    # After month 0: shares = 10, pool = 0, invested = 1000
    # (We can read these off index 0 directly.)
    assert res.shares[0] == pytest.approx(10.0)
    assert res.pool[0] == pytest.approx(0.0)
    assert res.invested[0] == pytest.approx(1000.0)


def test_tactico_monotonically_increasing_no_investments_after_month0():
    """If every monthly return is >= 0, the pool just grows and is ignored at the end."""
    prices = np.linspace(100.0, 200.0, 24)
    res = dca_tactico(prices, initial=1000.0, monthly=200.0)
    # Only the initial deposit ever hits the portfolio.
    assert res.total_invested == pytest.approx(1000.0)
    # Shares from month 0 onward never change.
    assert np.all(res.shares == res.shares[0])
    # Pool grows by `monthly` every month from month 1 onward (23 contributions).
    assert res.pool[-1] == pytest.approx(200.0 * 23)
    # Final value uses prices only, pool is ignored.
    assert res.final_value == pytest.approx(res.shares[-1] * prices[-1])


def test_tactico_monotonically_decreasing_invests_every_month():
    """If every monthly return is < 0, the pool is emptied every month from month 1."""
    prices = np.linspace(200.0, 100.0, 24)
    res = dca_tactico(prices, initial=1000.0, monthly=200.0)
    # All cash is deployed: initial + 23 monthlies, no leftover pool.
    assert res.pool[-1] == pytest.approx(0.0)
    assert res.total_invested == pytest.approx(1000.0 + 200.0 * 23)


def test_tactico_decision_uses_previous_month_return():
    """The decision at month t uses price[t]/price[t-1] - 1.

    Handcrafted series:
        prices = [100, 110, 99, 110, 100]
    Returns by month:
        t=1: 110/100 - 1 = +0.10  -> add monthly to pool, don't invest
        t=2:  99/110 - 1 < 0       -> deploy pool + monthly at price 99
        t=3: 110/99  - 1 > 0       -> add monthly to pool
        t=4: 100/110 - 1 < 0       -> deploy pool + monthly at price 100
    """
    prices = np.array([100.0, 110.0, 99.0, 110.0, 100.0])
    initial = 1000.0
    monthly = 200.0
    res = dca_tactico(prices, initial=initial, monthly=monthly)

    # After month 1: pool has one monthly, shares unchanged, invested unchanged.
    assert res.pool[1] == pytest.approx(monthly)
    assert res.invested[1] == pytest.approx(initial)
    assert res.shares[1] == pytest.approx(res.shares[0])

    # After month 2: pool emptied, deployed (pool+monthly) = 2*monthly at price 99.
    deployed_at_2 = 2 * monthly
    expected_new_shares = deployed_at_2 / 99.0
    assert res.pool[2] == pytest.approx(0.0)
    assert res.invested[2] == pytest.approx(initial + deployed_at_2)
    assert res.shares[2] == pytest.approx(res.shares[0] + expected_new_shares)

    # After month 3: pool grew by monthly, no investment.
    assert res.pool[3] == pytest.approx(monthly)
    assert res.invested[3] == pytest.approx(res.invested[2])

    # After month 4: pool emptied, deployed 2*monthly at price 100.
    deployed_at_4 = 2 * monthly
    assert res.pool[4] == pytest.approx(0.0)
    assert res.invested[4] == pytest.approx(res.invested[2] + deployed_at_4)


def test_tactico_does_not_use_t0_return():
    """No special-case rule applies at month 0; the first decision is at month 1
    using price[1]/price[0]. Verified indirectly by checking that the shares at
    month 0 equal initial/price[0] regardless of any subsequent return."""
    for prices in [
        np.array([100.0, 200.0]),
        np.array([100.0, 50.0]),
        np.array([100.0, 100.0]),
    ]:
        res = dca_tactico(prices, initial=1000.0, monthly=200.0)
        assert res.shares[0] == pytest.approx(10.0)
        assert res.invested[0] == pytest.approx(1000.0)
        assert res.pool[0] == pytest.approx(0.0)
