"""VaR / CVaR metodą Monte Carlo.

Stosujemy parametryczne MC na szeregu zwrotów *portfela*: estymujemy
(mu, sigma) z historii, losujemy N ścieżek o długości wybranego horyzontu
z rozkładu N(mu, sigma), agregujemy każdą ścieżkę do łącznego zwrotu
logarytmicznego, po czym bierzemy kwantyl empiryczny / warunkową średnią
ogonową z symulowanego rozkładu strat.

To rozwiązanie jest bliższe temu, jak MC odpala się na biurku, niż pełne
wieloaktywowe GBM ze skorelowanymi szokami, ale wersja na poziomie portfela
oddaje to samo ryzyko pierwszego rzędu dla projektu pokazowego i jest
trywialna do uzasadnienia. Potencjalnym rozszerzeniem byłoby losowanie
skorelowanych wielowymiarowych szoków normalnych na poziomie aktywów
i ponowna agregacja zgodnie z wagami.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_var_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float,
    horizon_days: int = 1,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Parametryczne MC na szeregu zwrotów portfela.

    Zwraca VaR i CVaR jako dodatnie wartości strat dla pełnego horyzontu.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError("potrzebne są co najmniej 2 obserwacje")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    rng = np.random.default_rng(seed)

    # Kształt: (n_sims, horizon_days). Sumowanie po osi 1 daje łączny
    # zwrot logarytmiczny dla każdej symulowanej ścieżki.
    shocks = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon_days))
    horizon_returns = shocks.sum(axis=1)

    alpha = 1.0 - confidence
    threshold = np.quantile(horizon_returns, alpha)
    var = -float(threshold)
    tail = horizon_returns[horizon_returns <= threshold]
    cvar = -float(tail.mean()) if tail.size else var
    return var, cvar
