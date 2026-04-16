"""Testy dla parametrycznego VaR/CVaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.risk.parametric import (
    parametric_var_cvar_normal,
    parametric_var_cvar_t,
)


def test_normal_var_closed_form(normal_returns):
    var, cvar = parametric_var_cvar_normal(normal_returns, 0.99)
    mu = float(normal_returns.mean())
    sigma = float(normal_returns.std(ddof=1))
    z = stats.norm.ppf(0.99)
    assert var == pytest.approx(-mu + sigma * z, rel=1e-6)
    # Zamknięta forma CVaR
    expected_cvar = -mu + sigma * stats.norm.pdf(z) / (1 - 0.99)
    assert cvar == pytest.approx(expected_cvar, rel=1e-6)


def test_normal_vs_historical_convergence(normal_returns):
    """Dla dużych i.i.d. próbek normalnych wynik parametryczny == historyczny
    z dokładnością do szumu."""
    from src.risk.historical import historical_var_cvar

    p_var, _ = parametric_var_cvar_normal(normal_returns, 0.95)
    h_var, _ = historical_var_cvar(normal_returns, 0.95)
    assert p_var == pytest.approx(h_var, rel=0.05)


def test_t_fit_on_fat_tailed_sample():
    """Losujemy ze znanego rozkładu t i sprawdzamy, że odzyskamy sensowne df."""
    rng = np.random.default_rng(7)
    r = stats.t.rvs(df=5, size=10_000, random_state=rng) * 0.01
    var, cvar, fitted_df = parametric_var_cvar_t(r, 0.99)
    # Dopasowane df powinno być w rozsądnym sąsiedztwie 5; MLE na 10 tys.
    # próbek zwykle trafia w ~30%.
    assert 3.0 < fitted_df < 8.0
    assert cvar > var


def test_insufficient_samples_raises():
    with pytest.raises(ValueError):
        parametric_var_cvar_normal(np.array([0.01]), 0.99)
    with pytest.raises(ValueError):
        parametric_var_cvar_t(np.zeros(5), 0.99)
