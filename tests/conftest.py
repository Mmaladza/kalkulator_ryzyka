"""Wspólne fixture'y pytest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def normal_returns() -> pd.Series:
    """5000 i.i.d. zwrotów dziennych N(0, 0.01) — wystarczająca próba do stabilnych kwantyli."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=0.01, size=5000)
    idx = pd.bdate_range("2005-01-03", periods=5000)
    return pd.Series(data, index=idx, name="sim")


@pytest.fixture
def two_asset_prices() -> pd.DataFrame:
    """Dwa syntetyczne szeregi cen o znanej korelacji."""
    rng = np.random.default_rng(1)
    n = 1000
    r1 = rng.normal(0.0005, 0.012, n)
    r2 = 0.6 * r1 + np.sqrt(1 - 0.6**2) * rng.normal(0.0003, 0.010, n)
    p1 = 100 * np.exp(np.cumsum(r1))
    p2 = 100 * np.exp(np.cumsum(r2))
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame({"A": p1, "B": p2}, index=idx)
