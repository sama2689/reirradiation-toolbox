from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QVBoxLayout, QWidget


class ContourSelectionWidget(QWidget):
    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.fixed_list = self._create_list_box("Fixed contours")
        self.moving_list = self._create_list_box("Moving contours")

        layout.addWidget(self.fixed_list, 1)
        layout.addWidget(self.moving_list, 1)

    def _create_list_box(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label = QLabel("Select one or more contours to overlay.")
        label.setWordWrap(True)
        label.setObjectName("subtleLabel")

        list_widget = QListWidget()
        list_widget.itemChanged.connect(lambda _item: self.selectionChanged.emit())
        list_widget.setMinimumHeight(180)
        list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)

        layout.addWidget(label)
        layout.addWidget(list_widget, 1)

        box.list_widget = list_widget  # type: ignore[attr-defined]
        return box

    def set_names(self, fixed_names: list[str], moving_names: list[str]) -> None:
        self._populate(self.fixed_list.list_widget, fixed_names)  # type: ignore[attr-defined]
        self._populate(self.moving_list.list_widget, moving_names)  # type: ignore[attr-defined]
        self.selectionChanged.emit()

    def selected_fixed_names(self) -> list[str]:
        return self._selected_names(self.fixed_list.list_widget)  # type: ignore[attr-defined]

    def selected_moving_names(self) -> list[str]:
        return self._selected_names(self.moving_list.list_widget)  # type: ignore[attr-defined]

    def _populate(self, list_widget: QListWidget, names: list[str]) -> None:
        list_widget.blockSignals(True)
        list_widget.clear()
        for name in sorted(names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            list_widget.addItem(item)
        list_widget.blockSignals(False)

    def _selected_names(self, list_widget: QListWidget) -> list[str]:
        selected: list[str] = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected
