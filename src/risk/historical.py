"""VaR / CVaR metodą historyczną — nieparametryczna, oparta na kwantylu empirycznym.

Historyczny VaR na poziomie ufności alpha to (1 - alpha)-ty kwantyl empiryczny
rozkładu zwrotów (raportowany jako dodatnia strata). CVaR (Expected Shortfall)
to średnia strata warunkowa pod warunkiem, że strata przekroczyła VaR.

Zalety: brak założenia o rozkładzie, uchwycone empiryczne grube ogony.
Wady: zakłada, że przyszłość wygląda jak niedawna przeszłość; estymator
kwantyla jest zaszumiony w ogonie przy małej próbie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Zwraca (VaR, CVaR) jako dodatnie wartości strat.

    Używa domyślnej liniowej interpolacji kwantyla z numpy — zgodnie
    z RiskMetrics i większością podręczników.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        raise ValueError("Tablica returns jest pusta")

    alpha = 1.0 - confidence
    var_threshold = np.quantile(r, alpha)      # liczba ujemna dla strat
    var = -float(var_threshold)

    tail = r[r <= var_threshold]
    # Fallback na ekstremalne przypadki brzegowe (np. rozkład jednostronny):
    cvar = -float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def scale_horizon(var: float, horizon_days: int) -> float:
    """Skalowanie horyzontu regułą pierwiastka z czasu.

    Ściśle słuszne przy założeniu i.i.d. normalnych zwrotów, ale jest
    rynkowym standardem dla krótkich horyzontów (Bazylea używa go
    dla 10-dniowego VaR).
    """
    return var * float(np.sqrt(horizon_days))
