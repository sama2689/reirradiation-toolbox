from __future__ import annotations

from typing import Any
import numbers

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class MetricsTableWidget(QWidget):
    def __init__(self, title: str = "Metrics", parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")

        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.title_label)
        layout.addWidget(self.table, 1)

    def set_title(self, title: str) -> None:
        self._title = title
        self.title_label.setText(title)

    def clear_table(self, placeholder: str = "No metrics available.") -> None:
        self.table.clear()
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Status"])
        item = QTableWidgetItem(placeholder)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(0, 0, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        if dataframe is None or dataframe.empty:
            self.clear_table()
            return

        frame = dataframe.copy().reset_index(drop=True)
        self.table.clear()
        self.table.setRowCount(len(frame))
        self.table.setColumnCount(len(frame.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in frame.columns])

        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row.tolist()):
                item = QTableWidgetItem(self._format_value(value))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row_index, column_index, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def set_records(self, records: list[dict[str, Any]], columns: list[str] | None = None) -> None:
        if not records:
            self.clear_table()
            return
        self.set_dataframe(pd.DataFrame.from_records(records, columns=columns))

    def enable_row_selection(self, enabled: bool = True) -> None:
        if enabled:
            self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        else:
            self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

    def selected_row_index(self) -> int | None:
        indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() is not None else []
        if not indexes:
            return None
        return indexes[0].row()

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return f"{float(value):.4g}"
        return str(value)
