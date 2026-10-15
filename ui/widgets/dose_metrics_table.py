from __future__ import annotations

from typing import Any, Callable

import numbers

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _SelectableRowWidget(QWidget):
    def __init__(self, on_clicked: Callable[[], None], parent: QWidget | None = None):
        super().__init__(parent)
        self._on_clicked = on_clicked
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def mousePressEvent(self, event):  # type: ignore[override]
        self._on_clicked()
        super().mousePressEvent(event)


class DoseMetricsTableWidget(QWidget):
    regionSelected = Signal(str)
    editAlphaBetaRequested = Signal(str)
    addMetricRequested = Signal(str)

    def __init__(self, title: str = "Contours", parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._row_regions: dict[int, str] = {}
        self._selected_region: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")

        self.table = QTableWidget(self)
        self.table.setColumnCount(1)
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background: white;
                border: none;
                gridline-color: transparent;
            }
            QTableWidget::item {
                padding: 0px;
                border: none;
            }
            """
        )

        layout.addWidget(self.title_label)
        layout.addWidget(self.table, 1)

    def set_title(self, title: str) -> None:
        self._title = title
        self.title_label.setText(title)

    def clear_table(self, placeholder: str = "No contours available.") -> None:
        self._row_regions.clear()
        self._selected_region = None
        scroll_value = self.table.verticalScrollBar().value()
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(1)
        self.table.setRowCount(1)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        item = QTableWidgetItem(placeholder)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(0, 0, item)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.blockSignals(False)
        self.table.verticalScrollBar().setValue(scroll_value)

    def set_blocks(self, blocks: list[dict[str, Any]]) -> None:
        scroll_value = self.table.verticalScrollBar().value()
        self._row_regions.clear()
        self._selected_region = None
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        row = 0
        for block in blocks:
            region = str(block.get("region", "")).strip()
            if not region:
                continue
            alpha_beta = float(block.get("alpha_beta", 3.0))
            metrics = list(block.get("metrics", []))

            self.table.insertRow(row)
            self._row_regions[row] = region
            self.table.setCellWidget(row, 0, self._make_region_widget(row, region, alpha_beta))
            row += 1

            for metric_name, metric_value in metrics:
                self.table.insertRow(row)
                self._row_regions[row] = region
                self.table.setCellWidget(row, 0, self._make_metric_widget(row, str(metric_name), metric_value))
                row += 1

        if row == 0:
            self.table.setRowCount(1)
            item = QTableWidgetItem("No contours available.")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(0, 0, item)
            self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        else:
            self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.resizeRowsToContents()
            self.table.horizontalHeader().setStretchLastSection(True)

        self.table.blockSignals(False)
        self.table.verticalScrollBar().setValue(scroll_value)

    def select_region(self, region: str | None) -> None:
        if region is None:
            return
        target = str(region).strip().lower()
        scroll_value = self.table.verticalScrollBar().value()
        for row, row_region in self._row_regions.items():
            if row_region.strip().lower() == target:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                self.table.verticalScrollBar().setValue(scroll_value)
                self._selected_region = row_region
                return

    def selected_region(self) -> str | None:
        return self._selected_region

    def selected_row_index(self) -> int | None:
        selected = self.table.selectionModel().selectedRows() if self.table.selectionModel() is not None else []
        if not selected:
            return None
        return selected[0].row()

    def _make_region_widget(self, row: int, region: str, alpha_beta: float) -> QWidget:
        container = _SelectableRowWidget(lambda: self.table.selectRow(row), self.table)
        container.setObjectName("doseMetricHeaderRow")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setAutoFillBackground(True)
        container.setStyleSheet(
            """
            QWidget#doseMetricHeaderRow {
                background: white;
                border-top: 1px solid #d8deea;
                border-bottom: 1px solid #d8deea;
            }
            QWidget#doseMetricHeaderRow QPushButton {
                border-radius: 7px;
                padding: 0px 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QWidget#doseMetricHeaderRow QPushButton#addMetricButton {
                min-width: 26px;
                max-width: 26px;
                min-height: 26px;
                max-height: 26px;
            }
            QWidget#doseMetricHeaderRow QPushButton#alphaBetaButton {
                min-height: 26px;
                max-height: 26px;
                min-width: 86px;
            }
            """
        )
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 8, 10, 8)
        container_layout.setSpacing(0)

        center_widget = QWidget(container)
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        name_label = QLabel(region)
        font = name_label.font()
        font.setBold(True)
        name_label.setFont(font)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        name_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        add_button = QPushButton("+")
        add_button.setObjectName("addMetricButton")
        add_button.setToolTip("Add DVH metric")
        add_button.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        add_button.clicked.connect(lambda *_: self.addMetricRequested.emit(region))

        alpha_button = QPushButton(f"α/β = {float(alpha_beta):.1f}")
        alpha_button.setObjectName("alphaBetaButton")
        alpha_button.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        alpha_button.clicked.connect(lambda *_: self.editAlphaBetaRequested.emit(region))

        center_layout.addWidget(name_label, 0)
        center_layout.addWidget(add_button, 0)
        center_layout.addWidget(alpha_button, 0)

        container_layout.addStretch(1)
        container_layout.addWidget(center_widget, 0, Qt.AlignmentFlag.AlignCenter)
        container_layout.addStretch(1)
        container.setMinimumHeight(46)
        return container

    def _make_metric_widget(self, row: int, metric_name: str, metric_value: Any) -> QWidget:
        container = _SelectableRowWidget(lambda: self.table.selectRow(row), self.table)
        container.setObjectName("doseMetricMetricRow")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setAutoFillBackground(True)
        container.setStyleSheet(
            """
            QWidget#doseMetricMetricRow {
                background: #f8fafc;
            }
            """
        )
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(18, 5, 18, 5)
        container_layout.setSpacing(10)

        metric_label = QLabel(metric_name)
        metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metric_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        metric_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        value_label = QLabel(self._format_value(metric_value))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        value_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        value_label.setStyleSheet("color: #374151;")

        container_layout.addStretch(1)
        container_layout.addWidget(metric_label, 1)
        container_layout.addWidget(value_label, 1)
        container_layout.addStretch(1)
        container.setMinimumHeight(34)
        return container

    def _on_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows() if self.table.selectionModel() is not None else []
        if not selected:
            return
        row = selected[0].row()
        region = self._row_regions.get(row)
        if region is None:
            return
        self._selected_region = region
        self.regionSelected.emit(region)

    def _on_cell_clicked(self, row: int, _column: int) -> None:
        if row in self._row_regions:
            self.table.selectRow(row)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return f"{float(value):.4g}"
        return str(value)
