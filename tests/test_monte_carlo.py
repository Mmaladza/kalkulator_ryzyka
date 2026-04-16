"""Testy dla Monte Carlo VaR/CVaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.risk.monte_carlo import monte_carlo_var_cvar


def test_seeded_reproducibility(normal_returns):
    a = monte_carlo_var_cvar(normal_returns, 0.95, n_sims=5000, seed=123)
    b = monte_carlo_var_cvar(normal_returns, 0.95, n_sims=5000, seed=123)
    assert a == b


def test_mc_converges_to_normal_var(normal_returns):
    """Parametryczne MC na danych normalnych powinno zbiec do zamkniętej formy
    normalnego VaR."""
    var, _ = monte_carlo_var_cvar(
        normal_returns, 0.99, horizon_days=1, n_sims=200_000, seed=7
    )
    sigma = float(normal_returns.std(ddof=1))
    analytical = -float(normal_returns.mean()) + sigma * stats.norm.ppf(0.99)
    assert var == pytest.approx(analytical, rel=0.02)


def test_horizon_scaling_matches_sqrt_t(normal_returns):
    """10-dniowy MC VaR powinien być ~sqrt(10) * 1-dniowy VaR."""
    v1, _ = monte_carlo_var_cvar(
        normal_returns, 0.99, horizon_days=1, n_sims=100_000, seed=1
    )
    v10, _ = monte_carlo_var_cvar(
        normal_returns, 0.99, horizon_days=10, n_sims=100_000, seed=1
    )
    assert v10 == pytest.approx(v1 * np.sqrt(10), rel=0.05)
