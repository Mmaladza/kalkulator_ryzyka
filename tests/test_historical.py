"""Testy dla historycznego VaR/CVaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.risk.historical import historical_var_cvar, scale_horizon


def test_historical_var_matches_normal_quantile(normal_returns):
    """Dla 5 tys. próbek N(0, sigma) historyczny VaR powinien trafić w analityczne z_alpha."""
    var, cvar = historical_var_cvar(normal_returns, confidence=0.99)
    sigma = float(normal_returns.std(ddof=1))
    analytical_var = -stats.norm.ppf(0.01) * sigma
    assert var == pytest.approx(analytical_var, rel=0.05)
    assert cvar > var  # Z definicji CVaR musi być większy od VaR


def test_historical_cvar_matches_normal_tail_expectation(normal_returns):
    """Dla normalnego o zerowej średniej ES_alpha = sigma * phi(z_alpha) / (1 - alpha)."""
    _, cvar = historical_var_cvar(normal_returns, confidence=0.99)
    sigma = float(normal_returns.std(ddof=1))
    z = stats.norm.ppf(0.99)
    expected = sigma * stats.norm.pdf(z) / (1 - 0.99)
    assert cvar == pytest.approx(expected, rel=0.08)


def test_scale_horizon_square_root_rule():
    assert scale_horizon(0.02, 1) == pytest.approx(0.02)
    assert scale_horizon(0.02, 4) == pytest.approx(0.04)
    assert scale_horizon(0.02, 10) == pytest.approx(0.02 * np.sqrt(10))


def test_empty_input_raises():
    with pytest.raises(ValueError):
        historical_var_cvar(np.array([]), confidence=0.95)
