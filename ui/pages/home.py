from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class HomePage(QWidget):
    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.main_window = main_window

        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("card")
        card.setMinimumSize(400, 400)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(0)

        title = QLabel("Image Registration")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc_label = QLabel(
            "Perform registration using rigid, affine, and deformable methods. "
            "Registration can be applied to dose images and evaluated using contour "
            "metrics. Select a directory containing the images to begin."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        desc_label.setWordWrap(True)

        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setMinimumHeight(40)
        self.select_dir_button.clicked.connect(self.main_window.on_select_directory_clicked)

        self.directory_label = QLabel("No directory selected.")
        self.directory_label.setObjectName("directoryLabel")
        self.directory_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.directory_label.setWordWrap(False)
        self.directory_label.setMinimumHeight(12)

        self.register_button = QPushButton("Perform Registration")
        self.register_button.setMinimumHeight(40)
        self.register_button.setEnabled(False)
        self.register_button.clicked.connect(self.main_window.show_selection_page)

        self.evaluate_button = QPushButton("Evaluate Registration")
        self.evaluate_button.setMinimumHeight(40)
        self.evaluate_button.setEnabled(False)

        self.apply_button = QPushButton("Apply Registration")
        self.apply_button.setMinimumHeight(40)
        self.apply_button.setEnabled(False)

        widgets_and_spacing = [
            (title, 12),
            (desc_label, 24),
            (self.select_dir_button, 10),
            (self.directory_label, 22),
            (self.register_button, 12),
            (self.evaluate_button, 12),
            (self.apply_button, 0),
        ]

        for widget, spacing in widgets_and_spacing:
            card_layout.addWidget(widget)
            if spacing:
                card_layout.addSpacing(spacing)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)

        outer_layout.addStretch(1)
        outer_layout.addLayout(row)
        outer_layout.addStretch(1)

    def update_directory_label(self, directory: str | None) -> None:
        if not directory:
            self.directory_label.setText("No directory selected.")
            self.directory_label.setToolTip("")
            return

        metrics = QFontMetrics(self.directory_label.font())
        available_width = max(20, self.directory_label.width() - 20)

        self.directory_label.setText(
            metrics.elidedText(
                directory,
                Qt.TextElideMode.ElideMiddle,
                available_width,
            )
        )
        self.directory_label.setToolTip(directory)

    def set_action_states(
        self,
        register_enabled: bool,
        evaluate_enabled: bool,
        apply_enabled: bool,
    ) -> None:
        self.register_button.setEnabled(register_enabled)
        self.evaluate_button.setEnabled(evaluate_enabled)
        self.apply_button.setEnabled(apply_enabled)