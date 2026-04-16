"""Klasy danych domeny — wejścia i wyjścia wymieniane między warstwami.

Utrzymujemy je czyste (bez zależności od pandas / PyQt), dzięki czemu można
je tanio przekazywać między wątkami i testować warstwę ryzyka bez dotykania UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal


class DataSource(str, Enum):
    YFINANCE = "yfinance"
    STOOQ = "stooq"


class RiskMethod(str, Enum):
    HISTORICAL = "historical"
    PARAMETRIC_NORMAL = "parametric_normal"
    PARAMETRIC_T = "parametric_t"
    MONTE_CARLO = "monte_carlo"


ReturnType = Literal["log", "simple"]


@dataclass(frozen=True)
class Position:
    """Pojedyncza pozycja w portfelu — ticker i waga w [0, 1].

    Źródło danych może się różnić między tickerami, bo niektóre instrumenty
    (np. WIG20 ze Stooqa) są dostępne tylko u jednego z dwóch dostawców.
    """
    ticker: str
    weight: float
    source: DataSource = DataSource.YFINANCE

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("Ticker nie może być pusty")
        if self.weight < 0:
            raise ValueError(f"Waga musi być nieujemna, podano {self.weight}")


@dataclass(frozen=True)
class PortfolioSpec:
    """Pełny zestaw wejść dla obliczenia VaR/CVaR."""
    positions: tuple[Position, ...]
    start: date
    end: date
    return_type: ReturnType = "log"
    portfolio_value: float = 1_000_000.0  # PLN/USD — prezentowane jako liczba

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("Portfel musi mieć przynajmniej jedną pozycję")
        total = sum(p.weight for p in self.positions)
        if total <= 0:
            raise ValueError("Suma wag musi być dodatnia")
        if self.start >= self.end:
            raise ValueError("start musi być wcześniej niż end")

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(p.weight for p in self.positions)
        return tuple(p.weight / total for p in self.positions)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(p.ticker for p in self.positions)


@dataclass(frozen=True)
class RiskParams:
    """Parametry: poziom ufności / horyzont / metoda."""
    confidence: float          # np. 0.95, 0.99
    horizon_days: int = 1
    method: RiskMethod = RiskMethod.HISTORICAL
    mc_simulations: int = 10_000
    mc_seed: int | None = 42

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1.0:
            raise ValueError(
                f"confidence musi być w (0.5, 1.0), podano {self.confidence}"
            )
        if self.horizon_days < 1:
            raise ValueError("horizon_days musi być >= 1")


@dataclass(frozen=True)
class RiskResult:
    """Wynik — VaR i CVaR w ujęciu względnym i nominalnym.

    VaR jest raportowany jako liczba DODATNIA reprezentująca stratę
    (konwencja branżowa). Np. `var_relative = 0.023` oznacza stratę 2.3%
    przy zadanym poziomie ufności.
    """
    method: RiskMethod
    confidence: float
    horizon_days: int
    var_relative: float
    cvar_relative: float
    var_absolute: float
    cvar_absolute: float
    portfolio_value: float
    n_observations: int
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResult:
    """Wynik testu Kupca — Proportion of Failures (POF)."""
    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_pvalue: float
    passed: bool  # True, jeśli nie odrzucamy H0 na poziomie 5%
