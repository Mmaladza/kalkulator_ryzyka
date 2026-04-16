"""Parametryczny VaR / CVaR (wariancja-kowariancja).

Dwie odmiany:
    - Normalna: zamknięta forma z użyciem kwantyla i gęstości rozkładu
      normalnego.
    - t-Studenta: ta sama maszyneria, ale z dopasowanymi stopniami swobody,
      która lepiej oddaje grube ogony obserwowane w realnych zwrotach akcji.

Dla straty L o średniej mu_L i odchyleniu sigma_L:
    VaR_alpha  = mu_L + sigma_L * z_alpha
    CVaR_alpha = mu_L + sigma_L * phi(z_alpha) / (1 - alpha)       (normal)

Ponieważ pracujemy na zwrotach r (gdzie P&L = -r, żeby straty były dodatnie):
    VaR  = -mu + sigma * z_alpha
    CVaR = -mu + sigma * phi(z_alpha) / (1 - alpha)

Dla rozkładu t-Studenta z nu stopniami swobody CVaR ma dobrze znaną zamkniętą
formę zawierającą gęstość t w jej kwantylu alpha; zob. McNeil, Frey & Embrechts
"Quantitative Risk Management" (2015), §2.2.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def parametric_var_cvar_normal(
    returns: pd.Series | np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError("potrzebne są co najmniej 2 obserwacje")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = stats.norm.ppf(confidence)
    phi_z = stats.norm.pdf(z)

    var = -mu + sigma * z
    cvar = -mu + sigma * phi_z / (1.0 - confidence)
    return var, cvar


def parametric_var_cvar_t(
    returns: pd.Series | np.ndarray,
    confidence: float,
) -> tuple[float, float, float]:
    """Dopasowuje rozkład t-Studenta i zwraca (VaR, CVaR, fitted_df).

    Stopnie swobody (df) są dopasowywane metodą największej wiarygodności
    (MLE) przez scipy. Dla dziennych zwrotów indeksów akcyjnych typowo lądują
    między 3 a 6 — wyraźnie grubsze ogony niż normalny.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 10:
        raise ValueError("potrzeba co najmniej 10 obserwacji do dopasowania t")

    df, loc, scale = stats.t.fit(r)
    # VaR: bezpośrednio z dopasowanego kwantyla
    loss_quantile = stats.t.ppf(1.0 - confidence, df, loc=loc, scale=scale)
    var = -float(loss_quantile)

    # CVaR dla t: E[L | L > VaR] przy dopasowanym rozkładzie.
    # Zamknięta forma dla standaryzowanego t (McNeil/Frey/Embrechts):
    #   ES_alpha = (pdf(t_alpha) / (1 - alpha)) * ((df + t_alpha**2) / (df - 1))
    t_alpha = stats.t.ppf(1.0 - confidence, df)  # kwantyl std-t
    pdf_val = stats.t.pdf(t_alpha, df)
    es_std = (pdf_val / (1.0 - confidence)) * ((df + t_alpha**2) / (df - 1.0))
    cvar = -loc + scale * es_std
    return var, cvar, float(df)
