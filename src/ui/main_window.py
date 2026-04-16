"""Okno główne — komponuje cztery widgety i uruchamia obliczenia.

Układ:

    ┌───────────────────────────────────────────────┐
    │                   pasek menu                   │
    ├─────────────┬─────────────────────────────────┤
    │ panel       │                                  │
    │ parametrów  │          wykresy (zakładki)      │
    │             │                                  │
    ├─────────────┤                                  │
    │ tabela      │                                  │
    │ tickerów    │                                  │
    ├─────────────┴─────────────────────────────────┤
    │ [  Oblicz  ]     panel wyników (inline)        │
    └───────────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
from pathlib import Path

from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.charts import ChartsWidget
from src.ui.widgets.params_panel import ParamsPanel
from src.ui.widgets.results_panel import ResultsPanel
from src.ui.widgets.ticker_table import TickerTable
from src.ui.workers import ComputationBundle, ComputationWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kalkulator ryzyka — VaR / CVaR")
        self.resize(1280, 800)

        self._pool = QThreadPool.globalInstance()
        self._last_bundle: ComputationBundle | None = None

        self._build_ui()
        self._build_menu()

    # -- konstrukcja ----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Lewa kolumna: parametry + tickery jeden pod drugim
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.params_panel = ParamsPanel()
        self.ticker_table = TickerTable()
        left_layout.addWidget(self.params_panel)
        left_layout.addWidget(self.ticker_table, stretch=1)
        splitter.addWidget(left)

        # Prawa kolumna: wykresy u góry, wyniki niżej
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.charts = ChartsWidget()
        self.results_panel = ResultsPanel()
        right_layout.addWidget(self.charts, stretch=3)
        right_layout.addWidget(self.results_panel, stretch=2)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.compute_btn = QPushButton("Oblicz ryzyko")
        self.compute_btn.setMinimumHeight(36)
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        root.addWidget(self.compute_btn)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Gotowy.")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&Plik")
        export_action = QAction("Eksport wyników do CSV…", self)
        export_action.triggered.connect(self._export_csv)
        menu.addAction(export_action)
        menu.addSeparator()
        quit_action = QAction("Zakończ", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("Pomo&c")
        about_action = QAction("O programie", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # -- akcje ----------------------------------------------------------

    def _on_compute_clicked(self) -> None:
        try:
            positions = self.ticker_table.positions()
            spec = self.params_panel.build_spec(positions)
            params = self.params_panel.build_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Nieprawidłowe dane", str(exc))
            return

        self.compute_btn.setEnabled(False)
        self.statusBar().showMessage("Pobieranie danych i obliczenia…")

        worker = ComputationWorker(spec, params)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _on_finished(self, bundle: ComputationBundle) -> None:
        self._last_bundle = bundle
        self.results_panel.update_risk(bundle.risk)
        self.results_panel.update_backtest(bundle.backtest)
        self.charts.render(bundle.portfolio_returns, bundle.risk)
        self.statusBar().showMessage(
            f"Obliczono: {bundle.risk.n_observations} obserwacji, "
            f"{len(bundle.prices.columns)} instrument(ów)."
        )
        self.compute_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Błąd obliczeń", message)
        self.statusBar().showMessage("Błąd.")
        self.compute_btn.setEnabled(True)

    def _export_csv(self) -> None:
        if self._last_bundle is None:
            QMessageBox.information(
                self, "Brak wyników", "Najpierw wykonaj obliczenia."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz wyniki", "wyniki_ryzyka.csv", "CSV (*.csv)"
        )
        if not path:
            return
        r = self._last_bundle.risk
        b = self._last_bundle.backtest
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pole", "wartość"])
            w.writerow(["metoda", r.method.value])
            w.writerow(["poziom_ufności", r.confidence])
            w.writerow(["horyzont_dni", r.horizon_days])
            w.writerow(["var_relative", r.var_relative])
            w.writerow(["cvar_relative", r.cvar_relative])
            w.writerow(["var_absolute", r.var_absolute])
            w.writerow(["cvar_absolute", r.cvar_absolute])
            w.writerow(["obserwacje", r.n_observations])
            w.writerow(["kupiec_exceptions", b.n_exceptions])
            w.writerow(["kupiec_expected", b.expected_exceptions])
            w.writerow(["kupiec_lr", b.kupiec_statistic])
            w.writerow(["kupiec_pvalue", b.kupiec_pvalue])
            w.writerow(["kupiec_passed", b.passed])
        self.statusBar().showMessage(f"Wyniki zapisane do {path}.")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Kalkulator ryzyka",
            "Kalkulator VaR / CVaR\n\n"
            "Metody: historyczna, parametryczna (normalna i t-Studenta), "
            "Monte Carlo.\n"
            "Backtest: test Kupca POF.\n"
            "Źródła danych: yfinance, Stooq.\n\n"
            "Projekt portfolio — market risk.",
        )
