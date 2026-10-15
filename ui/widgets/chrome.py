
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import QButtonGroup, QComboBox, QFrame, QHBoxLayout, QPushButton, QSizePolicy, QWidget


class ArrowComboBox(QComboBox):
    """Combo box with a reliably visible drop-down arrow."""

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        arrow_rect = QRect(self.width() - 24, 0, 24, self.height())
        if not arrow_rect.isValid():
            return

        color = self.palette().buttonText().color() if self.isEnabled() else self.palette().mid().color()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))

        center = arrow_rect.center()
        half_width = max(4, arrow_rect.width() // 5)
        half_height = max(2, arrow_rect.height() // 8)
        triangle = QPolygon([
            QPoint(center.x() - half_width, center.y() - half_height),
            QPoint(center.x() + half_width, center.y() - half_height),
            QPoint(center.x(), center.y() + half_height + 2),
        ])
        painter.drawPolygon(triangle)


class SegmentedButtonBar(QFrame):
    """Horizontal navigation bar made from checkable buttons."""

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("segmentedNavCard")
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: list[QPushButton] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._layout = layout

        self._button_group.idClicked.connect(self._on_clicked)

    def add_tab(self, text: str) -> int:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setObjectName("segmentedTabButton")
        button.setMinimumHeight(34)
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        index = len(self._buttons)
        self._buttons.append(button)
        self._button_group.addButton(button, index)
        self._layout.addWidget(button)
        if index == 0:
            button.setChecked(True)
        return index

    def set_current_index(self, index: int, emit: bool = True) -> None:
        if index < 0 or index >= len(self._buttons):
            return
        self._buttons[index].setChecked(True)
        if emit:
            self.currentChanged.emit(index)

    def current_index(self) -> int:
        return max(0, self._button_group.checkedId())

    def _on_clicked(self, index: int) -> None:
        self.currentChanged.emit(index)
