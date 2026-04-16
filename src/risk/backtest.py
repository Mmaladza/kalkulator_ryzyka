"""Backtest VaR — test Kupca Proportion-of-Failures (POF).

Jeżeli model VaR na poziomie ufności alpha jest poprawnie wyspecyfikowany,
to wskaźnik przekroczenia VaR jest i.i.d. Bernoullim z p = 1 - alpha.
Test Kupca to test ilorazu wiarygodności H0: p = 1 - alpha vs. H1: p != 1 - alpha:

    LR_POF = -2 * ln[ (p^x * (1-p)^(n-x)) / (x/n)^x * ((n-x)/n)^(n-x) ]

Przy H0, LR_POF ~ chi^2(1). Nie odrzucamy H0 na poziomie 5%, gdy p-wartość > 0.05.

Regulatorzy używają testu Kupca (plus test niezależności Christoffersena)
do walidacji modeli wewnętrznych. Zobacz Basel Committee „Supervisory framework
for the use of backtesting in conjunction with the internal models approach
to market risk capital requirements" (1996).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.models import BacktestResult


def kupiec_pof(
    returns: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray | float,
    confidence: float,
    significance: float = 0.05,
) -> BacktestResult:
    """Wykonuje test Kupca POF.

    Parameters
    ----------
    returns
        Zrealizowane zwroty portfela.
    var_series
        Liczba (statyczny VaR z pojedynczego okna zastosowany do całego okresu)
        lub szereg kroczących / zmiennych w czasie prognoz VaR o tej samej
        długości co `returns`. Oba są wyrażone jako dodatnie wartości strat.
    confidence
        Poziom ufności VaR — np. 0.99. Oczekiwany odsetek wyjątków to
        1 - confidence.
    """
    r = np.asarray(returns, dtype=float)
    if np.ndim(var_series) == 0:
        v = np.full_like(r, float(var_series))
    else:
        v = np.asarray(var_series, dtype=float)
        if v.shape != r.shape:
            raise ValueError("returns i var_series muszą mieć tę samą długość")

    # Wyjątek to zrealizowana strata większa niż prognoza VaR.
    # Strata = -zwrot; wyjątek, gdy -r > v  <=>  r < -v.
    exceptions = (r < -v).astype(int)
    x = int(exceptions.sum())
    n = int(exceptions.size)
    p = 1.0 - confidence
    expected = n * p

    # Statystyka LR z bezpieczną obsługą x == 0 i x == n.
    if x == 0:
        lr = -2.0 * (n * np.log(1.0 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        phat = x / n
        lr = -2.0 * (
            x * np.log(p) + (n - x) * np.log(1.0 - p)
            - x * np.log(phat) - (n - x) * np.log(1.0 - phat)
        )

    pvalue = 1.0 - stats.chi2.cdf(lr, df=1)
    passed = pvalue > significance

    return BacktestResult(
        n_observations=n,
        n_exceptions=x,
        expected_exceptions=expected,
        exception_rate=x / n if n else 0.0,
        expected_rate=p,
        kupiec_statistic=float(lr),
        kupiec_pvalue=float(pvalue),
        passed=bool(passed),
    )
