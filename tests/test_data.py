"""Tests for synthetic leveraged price construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import synthesize_leveraged_prices


def test_flat_base_stays_flat():
    """Zero daily returns → leveraged series is also constant."""
    base = pd.Series([100.0] * 10)
    out = synthesize_leveraged_prices(base, 3)
    assert np.allclose(out.values, 100.0)


def test_single_up_day_amplified_by_L():
    """Base +10% on day 1, then flat. With L=2, leveraged should be +20%."""
    base = pd.Series([100.0, 110.0, 110.0])
    out = synthesize_leveraged_prices(base, 2)
    assert out.iloc[0] == pytest.approx(100.0)
    assert out.iloc[1] == pytest.approx(120.0)
    assert out.iloc[2] == pytest.approx(120.0)


def test_inverse_leverage_negates_return():
    """L=-1: returns are flipped, so base +10% gives leveraged -10%."""
    base = pd.Series([100.0, 110.0])
    out = synthesize_leveraged_prices(base, -1)
    assert out.iloc[1] == pytest.approx(90.0)


def test_volatility_drag_emerges():
    """Base -10% then +11.11% (net 0%). Daily-reset L=2 must end below 100."""
    base = pd.Series([100.0, 90.0, 100.0])
    out = synthesize_leveraged_prices(base, 2)
    # Day 1: 100 · (1 + 2·(-0.10)) = 80
    # Day 2: 80  · (1 + 2·(10/90))  ≈ 97.78
    assert out.iloc[2] == pytest.approx(80.0 * (1 + 2 * (10.0 / 90.0)))
    assert out.iloc[2] < 100.0


def test_defensive_floor_keeps_price_positive():
    """If a single day would otherwise produce a non-positive multiplier
    (e.g. base -50% with L=3 → 1 - 1.5 = -0.5), the floor must keep the
    price strictly positive."""
    base = pd.Series([100.0, 50.0])  # -50%
    out = synthesize_leveraged_prices(base, 3)
    assert out.iloc[1] > 0
    # And it should be tiny (the floor is 1e-4 of previous price).
    assert out.iloc[1] < 1.0


def test_leverage_must_be_integer():
    base = pd.Series([100.0, 110.0])
    with pytest.raises(ValueError):
        synthesize_leveraged_prices(base, 2.5)  # type: ignore[arg-type]


def test_leverage_out_of_range_rejected():
    base = pd.Series([100.0, 110.0])
    with pytest.raises(ValueError):
        synthesize_leveraged_prices(base, 11)
    with pytest.raises(ValueError):
        synthesize_leveraged_prices(base, -11)


def test_leverage_one_reproduces_base_returns():
    """L=1 should match base returns exactly (price scale starts at base[0])."""
    base = pd.Series([100.0, 110.0, 99.0, 120.0])
    out = synthesize_leveraged_prices(base, 1)
    # Returns identical, starting price identical → series identical.
    np.testing.assert_allclose(out.values, base.values, rtol=1e-12)
