"""Obliczanie zwrotów z szeregu / ramki cen.

Dlaczego dwa typy zwrotów:
    - zwroty logarytmiczne są addytywne w czasie (r_{t->t+h} = sum r_i),
      co jest wygodne przy skalowaniu horyzontu i przy założeniu normalności
      w parametrycznym VaR;
    - zwroty proste są addytywne ze względu na wagi w pojedynczym momencie
      (r_port = sum w_i * r_i), co odpowiada temu, jak faktycznie realizuje
      się P&L portfela.

Udostępniamy oba i pozwalamy użytkownikowi wybrać. UI domyślnie używa
zwrotów logarytmicznych.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import ReturnType


def compute_returns(prices: pd.DataFrame, kind: ReturnType = "log") -> pd.DataFrame:
    """Zwraca DataFrame zwrotów dla poszczególnych tickerów.

    Parameters
    ----------
    prices
        Panel cen zamknięcia — indeks to data, kolumny to tickery.
    kind
        "log" -> ln(P_t / P_{t-1}); "simple" -> P_t / P_{t-1} - 1.

    Wiersze zawierające NaN (zwykle pierwszy wiersz albo dzień, w którym
    jedna z giełd była zamknięta) są usuwane, żeby dalsze obliczenia
    kowariancji / zwrotów portfela operowały na w pełni dopasowanym panelu.
    """
    if prices.empty:
        raise ValueError("DataFrame z cenami jest pusty")

    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    elif kind == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Nieznany typ zwrotów: {kind}")

    return rets.dropna(how="any")


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Agreguje zwroty poszczególnych aktywów w szereg zwrotów portfela.

    Dla zwrotów prostych jest to wzór dokładny. Dla zwrotów logarytmicznych
    to aproksymacja (log(sum w_i * exp(r_i)) ≈ sum w_i * r_i, gdy r_i jest
    małe) — to standardowa praktyka dla horyzontów dziennych.
    """
    if len(weights) != returns.shape[1]:
        raise ValueError(
            f"długość weights {len(weights)} != liczba aktywów {returns.shape[1]}"
        )
    w = np.asarray(weights, dtype=float)
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="portfolio")
