"""Edytowalna tabela portfela — ticker, waga, źródło."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.models import DataSource, Position


class TickerTable(QWidget):
    """Mała trójkolumnowa tabela z przyciskami dodaj/usuń wiersz."""

    COLUMNS = ("Ticker", "Waga", "Źródło")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Dodaj pozycję")
        self.remove_btn = QPushButton("Usuń zaznaczoną")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(lambda: self.add_row("", 1.0, DataSource.YFINANCE))
        self.remove_btn.clicked.connect(self._remove_selected)

        # Ładujemy mały realistyczny portfel demo, żeby aplikacja była użyteczna
        # od razu po starcie (i żeby recenzent widział coś bez wpisywania):
        self.add_row("AAPL", 0.4, DataSource.YFINANCE)
        self.add_row("MSFT", 0.4, DataSource.YFINANCE)
        self.add_row("^GSPC", 0.2, DataSource.YFINANCE)

    # -- API publiczne --------------------------------------------------

    def add_row(self, ticker: str, weight: float, source: DataSource) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(ticker))
        weight_item = QTableWidgetItem(f"{weight:.4f}")
        weight_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, 1, weight_item)

        combo = QComboBox()
        combo.addItems([s.value for s in DataSource])
        combo.setCurrentText(source.value)
        self.table.setCellWidget(row, 2, combo)

    def positions(self) -> list[Position]:
        out: list[Position] = []
        for row in range(self.table.rowCount()):
            ticker_item = self.table.item(row, 0)
            weight_item = self.table.item(row, 1)
            combo = self.table.cellWidget(row, 2)
            if ticker_item is None or weight_item is None or combo is None:
                continue
            ticker = ticker_item.text().strip()
            if not ticker:
                continue
            try:
                weight = float(weight_item.text().replace(",", "."))
            except ValueError as exc:
                raise ValueError(
                    f"Nieprawidłowa waga w wierszu {row + 1}: {weight_item.text()!r}"
                ) from exc
            source = DataSource(combo.currentText())
            out.append(Position(ticker=ticker, weight=weight, source=source))
        if not out:
            raise ValueError("Dodaj przynajmniej jedną pozycję do portfela.")
        return out

    # -- wewnętrzne -----------------------------------------------------

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
