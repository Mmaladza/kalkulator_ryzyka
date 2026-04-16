"""Formularz po lewej — daty, metoda, poziom ufności, horyzont, wartość portfela."""

from __future__ import annotations

from datetime import date, timedelta

from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from src.models import PortfolioSpec, RiskMethod, RiskParams


METHOD_LABELS: dict[RiskMethod, str] = {
    RiskMethod.HISTORICAL: "Historyczna",
    RiskMethod.PARAMETRIC_NORMAL: "Parametryczna (normalna)",
    RiskMethod.PARAMETRIC_T: "Parametryczna (t-Studenta)",
    RiskMethod.MONTE_CARLO: "Monte Carlo",
}


class ParamsPanel(QWidget):
    """Wszystkie wejścia spoza portfela zebrane w QFormLayout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        today = date.today()
        two_years_ago = today - timedelta(days=2 * 365)

        self.start_edit = QDateEdit(QDate(two_years_ago.year, two_years_ago.month, two_years_ago.day))
        self.start_edit.setCalendarPopup(True)
        self.end_edit = QDateEdit(QDate(today.year, today.month, today.day))
        self.end_edit.setCalendarPopup(True)

        self.method_combo = QComboBox()
        for method, label in METHOD_LABELS.items():
            self.method_combo.addItem(label, userData=method)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.80, 0.999)
        self.confidence_spin.setSingleStep(0.005)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setValue(0.99)

        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 250)
        self.horizon_spin.setValue(1)
        self.horizon_spin.setSuffix(" dni")

        self.portfolio_value_spin = QDoubleSpinBox()
        self.portfolio_value_spin.setRange(1_000.0, 1e12)
        self.portfolio_value_spin.setSingleStep(10_000.0)
        self.portfolio_value_spin.setDecimals(2)
        self.portfolio_value_spin.setGroupSeparatorShown(True)
        self.portfolio_value_spin.setValue(1_000_000.0)
        self.portfolio_value_spin.setSuffix(" PLN")

        self.mc_sims_spin = QSpinBox()
        self.mc_sims_spin.setRange(1_000, 1_000_000)
        self.mc_sims_spin.setSingleStep(1_000)
        self.mc_sims_spin.setValue(10_000)

        form.addRow("Data początkowa:", self.start_edit)
        form.addRow("Data końcowa:", self.end_edit)
        form.addRow("Metoda:", self.method_combo)
        form.addRow("Poziom ufności:", self.confidence_spin)
        form.addRow("Horyzont:", self.horizon_spin)
        form.addRow("Wartość portfela:", self.portfolio_value_spin)
        form.addRow("Symulacje MC:", self.mc_sims_spin)

    # -- API publiczne --------------------------------------------------

    def current_method(self) -> RiskMethod:
        return self.method_combo.currentData()

    def build_spec(self, positions) -> PortfolioSpec:
        start = self.start_edit.date().toPyDate()
        end = self.end_edit.date().toPyDate()
        return PortfolioSpec(
            positions=tuple(positions),
            start=start,
            end=end,
            return_type="log",
            portfolio_value=float(self.portfolio_value_spin.value()),
        )

    def build_params(self) -> RiskParams:
        return RiskParams(
            confidence=float(self.confidence_spin.value()),
            horizon_days=int(self.horizon_spin.value()),
            method=self.current_method(),
            mc_simulations=int(self.mc_sims_spin.value()),
            mc_seed=42,
        )
