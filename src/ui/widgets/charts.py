"""Osadzone wykresy matplotlib — histogram zwrotów i krzywa kapitału.

Używamy backendu Qt Agg przez FigureCanvasQTAgg, dzięki czemu wykresy
żyją w oknie głównym jako zwykłe widgety (bez osobnych okien matplotlib).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QTabWidget, QWidget

from src.models import RiskResult


class ChartsWidget(QWidget):
    """Dwie zakładki: histogram zwrotów portfela i krzywa kapitału."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.hist_canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4), tight_layout=True))
        self.equity_canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4), tight_layout=True))
        self.tabs.addTab(self.hist_canvas, "Histogram zwrotów")
        self.tabs.addTab(self.equity_canvas, "Krzywa kapitału")

        self._render_empty()

    # -- API publiczne --------------------------------------------------

    def render(self, portfolio_returns: pd.Series, risk: RiskResult) -> None:
        self._render_histogram(portfolio_returns, risk)
        self._render_equity(portfolio_returns)

    # -- wewnętrzne -----------------------------------------------------

    def _render_empty(self) -> None:
        for canvas, title in (
            (self.hist_canvas, "Histogram zwrotów portfela"),
            (self.equity_canvas, "Krzywa kapitału"),
        ):
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Uruchom obliczenia, aby zobaczyć wykres.",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()

    def _render_histogram(self, returns: pd.Series, risk: RiskResult) -> None:
        fig = self.hist_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        data = returns.to_numpy()
        ax.hist(data, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white")

        # VaR/CVaR są raportowane jako dodatnie wartości strat; na osi zwrotów
        # odpowiadają im liczby ujemne.
        var_line = -risk.var_relative
        cvar_line = -risk.cvar_relative

        ax.axvline(var_line, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"VaR {risk.confidence*100:.1f}%  = {risk.var_relative*100:.2f}%")
        ax.axvline(cvar_line, color="#7f0000", linestyle=":", linewidth=1.8,
                   label=f"CVaR {risk.confidence*100:.1f}% = {risk.cvar_relative*100:.2f}%")
        ax.axvline(float(np.mean(data)), color="#2ca02c", linestyle="-", linewidth=1.0,
                   label=f"Średnia = {np.mean(data)*100:.3f}%")

        ax.set_title("Rozkład dziennych zwrotów portfela")
        ax.set_xlabel("Zwrot")
        ax.set_ylabel("Liczba obserwacji")
        ax.legend(loc="upper left", fontsize=9)
        self.hist_canvas.draw_idle()

    def _render_equity(self, returns: pd.Series) -> None:
        fig = self.equity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Traktujemy zwroty jak logarytmiczne przy liczeniu skumulowanego
        # iloczynu; to rozsądna aproksymacja nawet gdy użytkownik wybrał
        # zwroty proste (jakościowy kształt jest przy częstotliwości dziennej
        # identyczny).
        equity = np.exp(returns.cumsum())
        ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.4)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title("Krzywa kapitału (znormalizowana do 1.0)")
        ax.set_xlabel("Data")
        ax.set_ylabel("Wartość")
        fig.autofmt_xdate()
        self.equity_canvas.draw_idle()
