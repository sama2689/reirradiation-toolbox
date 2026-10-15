from __future__ import annotations

from typing import TYPE_CHECKING

import SimpleITK as sitk
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QPushButton, QVBoxLayout, QWidget

from ui.widgets.image_viewer import SquareCTDisplay
from ui.widgets.registration_pipeline import RegistrationWidget
from ui.widgets.image_viewer_controls import ViewerControlsWidget

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class RegistrationPage(QWidget):
    run_requested = Signal(dict)

    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.main_window = main_window
        self._signals_connected = False
        self.fixed_image: sitk.Image | None = None
        self.moving_image: sitk.Image | None = None

        self._build_ui()
        self.registration_card.run_requested.connect(self.run_requested.emit)

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(16)

        top_card = QFrame()
        top_card.setObjectName("card")

        top_card_layout = QGridLayout(top_card)
        top_card_layout.setContentsMargins(18, 14, 18, 14)
        top_card_layout.setHorizontalSpacing(12)

        back_button = QPushButton("Back")
        back_button.setMinimumHeight(40)
        back_button.clicked.connect(self.main_window.show_home_page)

        page_title = QLabel("Perform Registration")
        page_title.setObjectName("title")
        page_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_spacer = QWidget()
        right_spacer.setFixedWidth(back_button.sizeHint().width())

        top_card_layout.addWidget(back_button, 0, 0)
        top_card_layout.addWidget(page_title, 0, 1)
        top_card_layout.addWidget(right_spacer, 0, 2)
        top_card_layout.setColumnStretch(0, 0)
        top_card_layout.setColumnStretch(1, 1)
        top_card_layout.setColumnStretch(2, 0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(16)

        self.fixed_view = SquareCTDisplay()
        self.moving_view = SquareCTDisplay()
        self.controls = ViewerControlsWidget()

        fixed_card = self.create_scan_card("Fixed CT Scan", self.fixed_view)
        moving_card = self.create_scan_card("Moving CT Scan", self.moving_view)
        controls_card = self.create_controls_card()
        self.registration_card = RegistrationWidget()

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)
        right_layout.addWidget(controls_card, 0)
        right_layout.addWidget(self.registration_card, 1)

        body.addWidget(fixed_card, 3)
        body.addWidget(moving_card, 3)
        body.addWidget(right_column, 2)

        page_layout.addWidget(top_card)
        page_layout.addLayout(body)

    def create_scan_card(self, title_text: str, viewer_widget: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("sectionTitle")

        layout.addWidget(title)
        layout.addWidget(viewer_widget, 1)

        return card

    def create_controls_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Viewer Controls")
        title.setObjectName("sectionTitle")

        layout.addWidget(title)
        layout.addWidget(self.controls)

        return card

    def set_registration_images(self, fixed_image: sitk.Image, moving_image: sitk.Image) -> None:
        self.fixed_image = fixed_image
        self.moving_image = moving_image

        self.fixed_view.set_volume(fixed_image)
        self.moving_view.set_volume(moving_image)
        self.fixed_view.set_render_scale_factor(1.0)
        self.moving_view.set_render_scale_factor(1.0)

        if not self._signals_connected:
            self.controls.sliceChanged.connect(self.fixed_view.set_slice)
            self.controls.sliceChanged.connect(self.moving_view.set_slice)
            self.controls.windowChanged.connect(self.update_window_level)
            self.controls.levelChanged.connect(self.update_window_level)
            self.controls.orientationChanged.connect(self._sync_orientation)
            self._signals_connected = True

        self.controls.set_orientation("axial", emit=False)
        self._sync_orientation("axial")
        self.update_window_level()

    def _slice_count(self, orientation: str) -> int:
        if self.fixed_view.volume is None or self.moving_view.volume is None:
            return 1

        orientation = orientation.strip().lower()
        axis_map = {
            "axial": 0,
            "coronal": 1,
            "sagittal": 2,
        }
        axis = axis_map.get(orientation, 0)
        return max(int(self.fixed_view.volume.shape[axis]), int(self.moving_view.volume.shape[axis]))

    def _sync_orientation(self, orientation: str) -> None:
        orientation = orientation.strip().lower()
        if orientation not in {"axial", "coronal", "sagittal"}:
            orientation = "axial"

        self.fixed_view.set_orientation(orientation)
        self.moving_view.set_orientation(orientation)

        max_slice = max(0, self._slice_count(orientation) - 1)
        current_slice = max_slice // 2
        self.controls.set_slice_range(0, max_slice, current_slice, emit=False)
        self.fixed_view.set_slice(current_slice)
        self.moving_view.set_slice(current_slice)

    def update_window_level(self) -> None:
        window = self.controls.window_spin.value()
        level = self.controls.level_spin.value()
        self.fixed_view.set_window_level(window, level)
        self.moving_view.set_window_level(window, level)

    def set_contour_names(self, moving_names: list[str], fixed_names: list[str]) -> None:
        self.registration_card.set_contour_names(moving_names, fixed_names)

    def set_preview_context(
        self,
        fixed_image: sitk.Image | None,
        moving_image: sitk.Image | None,
        fixed_structures: dict[str, sitk.Image] | None,
        moving_structures: dict[str, sitk.Image] | None,
    ) -> None:
        self.registration_card.set_preview_context(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_structures=fixed_structures,
            moving_structures=moving_structures,
        )
