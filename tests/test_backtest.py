"""Testy dla backtestu Kupca POF."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.risk.backtest import kupiec_pof


def test_on_spec_model_passes():
    """Gdy odsetek wyjątków równa się dokładnie 1 - alpha, LR = 0 i p = 1."""
    rng = np.random.default_rng(11)
    n = 10_000
    alpha = 0.99
    returns = rng.normal(0.0, 0.01, n)
    # Wymuszamy empiryczny odsetek wyjątków ≈ 1%.
    threshold = np.quantile(returns, 0.01)
    bt = kupiec_pof(returns, var_series=-threshold, confidence=alpha)
    assert bt.passed
    assert bt.kupiec_pvalue > 0.5  # Powinno być duże, gdy model jest „na spec"


def test_badly_underestimated_var_fails():
    """Jeśli VaR jest skrajnie za niski, wyjątków jest dużo i p-wartość spada."""
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0, 0.02, 2000)
    bt = kupiec_pof(returns, var_series=0.005, confidence=0.99)
    assert not bt.passed
    assert bt.kupiec_pvalue < 0.01
    assert bt.n_exceptions > bt.expected_exceptions


def test_likelihood_ratio_matches_manual():
    """Sanity-check statystyki LR wobec ręcznej formuły dla znanych liczb."""
    # Konstrukcja: 1000 obs., 20 wyjątków, alpha = 0.99 -> oczekiwane 10.
    n, x, p = 1000, 20, 0.01
    phat = x / n
    lr_expected = -2.0 * (
        x * np.log(p) + (n - x) * np.log(1 - p)
        - x * np.log(phat) - (n - x) * np.log(1 - phat)
    )
    pvalue_expected = 1.0 - stats.chi2.cdf(lr_expected, df=1)

    # Konstrukcja wektora returns/var dającego dokładnie 20 wyjątków.
    returns = np.full(n, -0.001)  # nieszkodliwy zwrot
    returns[:x] = -0.10           # wyjątki: strata 0.10 > VaR = 0.05
    bt = kupiec_pof(returns, var_series=0.05, confidence=0.99)

    assert bt.n_exceptions == x
    assert bt.kupiec_statistic == pytest.approx(lr_expected, rel=1e-6)
    assert bt.kupiec_pvalue == pytest.approx(pvalue_expected, rel=1e-6)


def test_zero_exceptions_handled():
    returns = np.full(500, 0.001)  # żadnych strat
    bt = kupiec_pof(returns, var_series=0.05, confidence=0.99)
    assert bt.n_exceptions == 0
    assert np.isfinite(bt.kupiec_statistic)
    assert 0.0 <= bt.kupiec_pvalue <= 1.0
