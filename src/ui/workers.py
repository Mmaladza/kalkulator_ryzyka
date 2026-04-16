"""Wątki robocze QThread — ciężka praca poza pętlą zdarzeń Qt.

Pobranie danych (yfinance/Stooq) i Monte Carlo trwają zauważalnie długo
i blokowałyby GUI, gdyby leciały na wątku głównym. Pakujemy pełny pipeline
"pobierz ceny -> policz ryzyko -> zrób backtest" w QRunnable, który emituje
pojedynczy sygnał `finished` z paczką wyników.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from src.data.providers import fetch_prices
from src.models import (
    BacktestResult,
    PortfolioSpec,
    RiskParams,
    RiskResult,
)
from src.risk.backtest import kupiec_pof
from src.risk.portfolio import compute_portfolio_returns, compute_risk


@dataclass
class ComputationBundle:
    prices: pd.DataFrame
    portfolio_returns: pd.Series
    risk: RiskResult
    backtest: BacktestResult


class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # ComputationBundle
    failed = pyqtSignal(str)


class ComputationWorker(QRunnable):
    """Pobierz -> zwroty portfela -> VaR/CVaR -> backtest Kupca.

    Uruchamiany w QThreadPool. Wszystkie wyjścia emitowane są sygnałami,
    dzięki czemu wątek UI pozostaje responsywny, a worker nie trzyma
    żadnych widgetów Qt.
    """

    def __init__(self, spec: PortfolioSpec, params: RiskParams) -> None:
        super().__init__()
        self.spec = spec
        self.params = params
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            prices = fetch_prices(self.spec.positions, self.spec.start, self.spec.end)
            if prices.empty:
                raise RuntimeError("Brak danych cenowych — sprawdź tickery i daty.")

            port_rets = compute_portfolio_returns(prices, self.spec)
            risk = compute_risk(prices, self.spec, self.params)

            # Backtest używa STATYCZNEJ estymaty VaR wobec zrealizowanych
            # zwrotów in-sample. Dla projektu pokazowego jawnie zaznaczamy,
            # że normą regulatora jest backtest out-of-sample z oknem
            # kroczącym; test statyczny i tak pokazuje mechanikę.
            backtest = kupiec_pof(
                port_rets.to_numpy(),
                var_series=risk.var_relative,
                confidence=self.params.confidence,
            )

            bundle = ComputationBundle(
                prices=prices,
                portfolio_returns=port_rets,
                risk=risk,
                backtest=backtest,
            )
            self.signals.finished.emit(bundle)
        except Exception as exc:  # noqa: BLE001 — cokolwiek zdarzy się, pokazujemy w UI
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
