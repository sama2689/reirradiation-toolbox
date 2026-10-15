from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from ui.widgets.chrome import ArrowComboBox as QComboBox
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


class SelectionPage(QWidget):
    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.main_window = main_window

        self.image_series_dataset: list[dict[str, Any]] = []
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

        title = QLabel("Select Registration Images")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc_label = QLabel(
            "Choose the fixed and moving image series that will be used during registration."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        desc_label.setWordWrap(True)

        fixed_label = QLabel("Fixed Image")
        fixed_label.setObjectName("sectionTitle")

        fixed_desc = QLabel("The reference image that stays unchanged.")
        fixed_desc.setWordWrap(True)

        self.fixed_combo = QComboBox()

        moving_label = QLabel("Moving Image")
        moving_label.setObjectName("sectionTitle")

        moving_desc = QLabel("The image that will be aligned to the fixed image.")
        moving_desc.setWordWrap(True)

        self.moving_combo = QComboBox()

        button_row = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.main_window.show_home_page)

        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.continue_to_registration)

        button_row.addWidget(back_button)
        button_row.addStretch(1)
        button_row.addWidget(continue_button)

        widgets_and_spacing = [
            (title, 12),
            (desc_label, 22),
            (fixed_label, 4),
            (fixed_desc, 8),
            (self.fixed_combo, 18),
            (moving_label, 4),
            (moving_desc, 8),
            (self.moving_combo, 24),
        ]

        for widget, spacing in widgets_and_spacing:
            card_layout.addWidget(widget)
            if spacing:
                card_layout.addSpacing(spacing)

        card_layout.addLayout(button_row)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)

        outer_layout.addStretch(1)
        outer_layout.addLayout(row)
        outer_layout.addStretch(1)

    def populate_series_dropdowns(self, image_series_dataset: list[dict[str, Any]]) -> None:
        self.image_series_dataset = image_series_dataset

        self.fixed_combo.clear()
        self.moving_combo.clear()

        self.fixed_combo.addItem("Select fixed image...")
        self.fixed_combo.model().item(0).setEnabled(False)

        self.moving_combo.addItem("Select moving image...")
        self.moving_combo.model().item(0).setEnabled(False)

        for series in self.image_series_dataset:
            text = f"{series['name']} ({series['type']})"
            self.fixed_combo.addItem(text, series)
            self.moving_combo.addItem(text, series)

        self.fixed_combo.setCurrentIndex(0)
        self.moving_combo.setCurrentIndex(0)

    def continue_to_registration(self) -> None:
        fixed = self.fixed_combo.currentData()
        moving = self.moving_combo.currentData()

        if (
            not fixed
            or not moving
            or self.fixed_combo.currentIndex() == 0
            or self.moving_combo.currentIndex() == 0
        ):
            self.main_window.show_warning(
                title="Missing Selection",
                message="Missing Selection",
                info_text="Both the fixed and moving selections must be complete.",
            )
            return

        if fixed["path"] == moving["path"]:
            self.main_window.show_warning(
                title="Invalid Selection",
                message="Invalid Selection",
                info_text="The image series for the fixed and moving selection must be different.",
            )
            return

        self.main_window.set_selected_series(fixed, moving)