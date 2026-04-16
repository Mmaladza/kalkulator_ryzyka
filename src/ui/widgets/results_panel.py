"""Panel wyników po prawej — VaR/CVaR oraz wyniki backtestu."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from src.models import BacktestResult, RiskResult


METHOD_LABELS_PL: dict[str, str] = {
    "historical": "Historyczna",
    "parametric_normal": "Parametryczna (normalna)",
    "parametric_t": "Parametryczna (t-Studenta)",
    "monte_carlo": "Monte Carlo",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.3f}%"


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


class ResultsPanel(QWidget):
    """Dwa zgrupowane odczyty: Ryzyko i Backtest."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ---- grupa: ryzyko ----
        risk_group = QGroupBox("Wyniki VaR / CVaR")
        risk_form = QFormLayout(risk_group)

        self.method_label = QLabel("—")
        self.confidence_label = QLabel("—")
        self.horizon_label = QLabel("—")
        self.var_rel_label = QLabel("—")
        self.cvar_rel_label = QLabel("—")
        self.var_abs_label = QLabel("—")
        self.cvar_abs_label = QLabel("—")
        self.obs_label = QLabel("—")
        self.extras_label = QLabel("—")

        for lbl in (
            self.var_rel_label,
            self.cvar_rel_label,
            self.var_abs_label,
            self.cvar_abs_label,
        ):
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)

        risk_form.addRow("Metoda:", self.method_label)
        risk_form.addRow("Poziom ufności:", self.confidence_label)
        risk_form.addRow("Horyzont:", self.horizon_label)
        risk_form.addRow("VaR (% portfela):", self.var_rel_label)
        risk_form.addRow("CVaR (% portfela):", self.cvar_rel_label)
        risk_form.addRow("VaR (kwota):", self.var_abs_label)
        risk_form.addRow("CVaR (kwota):", self.cvar_abs_label)
        risk_form.addRow("Obserwacje:", self.obs_label)
        risk_form.addRow("Dodatkowe:", self.extras_label)
        layout.addWidget(risk_group)

        # ---- grupa: backtest ----
        bt_group = QGroupBox("Backtest VaR (test Kupca POF)")
        bt_form = QFormLayout(bt_group)
        self.bt_n_label = QLabel("—")
        self.bt_x_label = QLabel("—")
        self.bt_expected_label = QLabel("—")
        self.bt_rate_label = QLabel("—")
        self.bt_lr_label = QLabel("—")
        self.bt_pvalue_label = QLabel("—")
        self.bt_verdict_label = QLabel("—")
        bt_form.addRow("Obserwacje:", self.bt_n_label)
        bt_form.addRow("Wyjątki (realne):", self.bt_x_label)
        bt_form.addRow("Wyjątki (oczekiwane):", self.bt_expected_label)
        bt_form.addRow("Częstość wyjątków:", self.bt_rate_label)
        bt_form.addRow("Statystyka LR:", self.bt_lr_label)
        bt_form.addRow("p-wartość:", self.bt_pvalue_label)
        bt_form.addRow("Ocena:", self.bt_verdict_label)
        layout.addWidget(bt_group)

        layout.addStretch(1)

    # -- API publiczne --------------------------------------------------

    def update_risk(self, result: RiskResult) -> None:
        self.method_label.setText(METHOD_LABELS_PL.get(result.method.value, result.method.value))
        self.confidence_label.setText(f"{result.confidence * 100:.2f}%")
        self.horizon_label.setText(f"{result.horizon_days} dni")
        self.var_rel_label.setText(_fmt_pct(result.var_relative))
        self.cvar_rel_label.setText(_fmt_pct(result.cvar_relative))
        self.var_abs_label.setText(_fmt_money(result.var_absolute))
        self.cvar_abs_label.setText(_fmt_money(result.cvar_absolute))
        self.obs_label.setText(str(result.n_observations))
        if result.extras:
            extras_txt = ", ".join(f"{k} = {v:.3f}" for k, v in result.extras.items())
        else:
            extras_txt = "—"
        self.extras_label.setText(extras_txt)

    def update_backtest(self, bt: BacktestResult) -> None:
        self.bt_n_label.setText(str(bt.n_observations))
        self.bt_x_label.setText(str(bt.n_exceptions))
        self.bt_expected_label.setText(f"{bt.expected_exceptions:.2f}")
        self.bt_rate_label.setText(f"{bt.exception_rate * 100:.3f}% (oczek. {bt.expected_rate * 100:.3f}%)")
        self.bt_lr_label.setText(f"{bt.kupiec_statistic:.4f}")
        self.bt_pvalue_label.setText(f"{bt.kupiec_pvalue:.4f}")
        verdict = "ZALICZONY (p > 0.05)" if bt.passed else "ODRZUCONY (p ≤ 0.05)"
        self.bt_verdict_label.setText(verdict)
        color = "#2e7d32" if bt.passed else "#c62828"
        self.bt_verdict_label.setStyleSheet(f"color: {color}; font-weight: bold;")
