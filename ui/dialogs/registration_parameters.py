from __future__ import annotations

from typing import Any, Dict

import SimpleITK as sitk

from PySide6.QtCore import Qt
from ui.widgets.chrome import ArrowComboBox as QComboBox
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from registration.methods.pipeline import STEP_TEMPLATES
from registration.methods.masking import Masking
from ui.widgets.evaluation_controls import EvaluationControlsWidget
from ui.widgets.image_viewer import OverlayLayer, SquareCTDisplay


class BaseParamWidget(QFrame):
    def get_values(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_values(self, values: Dict[str, Any]) -> None:
        raise NotImplementedError


class MaskingParamWidget(BaseParamWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramCard")

        FIELD_WIDTH = 240
        LABEL_WIDTH = 140

        self._preview_context: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Masking Parameters")
        title.setObjectName("paramTitle")
        layout.addWidget(title)
        layout.addSpacing(12)

        self.main_form = QFormLayout()
        self.main_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.main_form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.main_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.main_form.setHorizontalSpacing(16)
        self.main_form.setVerticalSpacing(10)

        def add_row(text: str, widget: QWidget) -> None:
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)
            widget.setFixedWidth(FIELD_WIDTH)
            self.main_form.addRow(label, widget)

        self.mode = QComboBox()
        self.mode.addItems(["Contour", "Threshold"])
        self.mode.currentTextChanged.connect(self._update_mode_page)
        add_row("Masking Method", self.mode)

        self.erosion = QSpinBox()
        self.erosion.setRange(0, 1000)
        self.erosion.setSingleStep(1)
        add_row("Erosion", self.erosion)

        self.dilation = QSpinBox()
        self.dilation.setRange(0, 1000)
        self.dilation.setSingleStep(1)
        add_row("Dilation", self.dilation)

        layout.addLayout(self.main_form)

        self.mode_stack = QStackedWidget()

        contour_page = QWidget()
        contour_form = QFormLayout(contour_page)
        contour_form.setContentsMargins(0, 0, 0, 0)
        contour_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        contour_form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        contour_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        contour_form.setHorizontalSpacing(16)
        contour_form.setVerticalSpacing(10)

        self.moving_contour = QComboBox()
        self.moving_contour.setEditable(True)
        self.moving_contour.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.moving_contour.setPlaceholderText("Select moving contour...")
        self.moving_contour.setFixedWidth(FIELD_WIDTH)

        self.fixed_contour = QComboBox()
        self.fixed_contour.setEditable(True)
        self.fixed_contour.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.fixed_contour.setPlaceholderText("Select fixed contour...")
        self.fixed_contour.setFixedWidth(FIELD_WIDTH)

        self._add_row(contour_form, "Moving Contour", self.moving_contour, LABEL_WIDTH, FIELD_WIDTH)
        self._add_row(contour_form, "Fixed Contour", self.fixed_contour, LABEL_WIDTH, FIELD_WIDTH)

        threshold_page = QWidget()
        threshold_form = QFormLayout(threshold_page)
        threshold_form.setContentsMargins(0, 0, 0, 0)
        threshold_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        threshold_form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        threshold_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        threshold_form.setHorizontalSpacing(16)
        threshold_form.setVerticalSpacing(10)

        self.lower_hu_threshold = QSpinBox()
        self.lower_hu_threshold.setRange(-5000, 5000)
        self.lower_hu_threshold.setSingleStep(10)
        self.lower_hu_threshold.setFixedWidth(FIELD_WIDTH)

        self.upper_hu_threshold = QSpinBox()
        self.upper_hu_threshold.setRange(-5000, 5000)
        self.upper_hu_threshold.setSingleStep(10)
        self.upper_hu_threshold.setFixedWidth(FIELD_WIDTH)

        self._add_row(threshold_form, "Lower HU threshold", self.lower_hu_threshold, LABEL_WIDTH, FIELD_WIDTH)
        self._add_row(threshold_form, "Upper HU threshold", self.upper_hu_threshold, LABEL_WIDTH, FIELD_WIDTH)

        self.mode_stack.addWidget(contour_page)
        self.mode_stack.addWidget(threshold_page)

        layout.addWidget(self.mode_stack)

        self.crop_based_on_mask = QCheckBox()
        self.crop_based_on_mask.setFixedWidth(FIELD_WIDTH)

        crop_row = QHBoxLayout()
        crop_row.setContentsMargins(0, 8, 0, 0)
        crop_row.setSpacing(16)

        crop_label = QLabel("Crop based on mask")
        crop_label.setFixedWidth(LABEL_WIDTH)

        crop_row.addWidget(crop_label)
        crop_row.addWidget(self.crop_based_on_mask)
        crop_row.addStretch(1)

        layout.addLayout(crop_row)

        self.preview_btn = QPushButton("Preview Mask")
        self.preview_btn.setObjectName("utilityButton")
        self.preview_btn.setFixedHeight(38)
        self.preview_btn.clicked.connect(self.preview_mask)
        layout.addWidget(self.preview_btn)

        layout.addStretch(1)

        self._update_mode_page()

    def _add_row(self, form: QFormLayout, text: str, widget: QWidget, label_width: int, field_width: int) -> None:
        label = QLabel(text)
        label.setFixedWidth(label_width)
        widget.setFixedWidth(field_width)
        form.addRow(label, widget)

    def set_contour_names(self, moving_names: list[str], fixed_names: list[str]) -> None:
        self.moving_contour.blockSignals(True)
        self.fixed_contour.blockSignals(True)

        self.moving_contour.clear()
        self.fixed_contour.clear()

        self.moving_contour.addItems(moving_names)
        self.fixed_contour.addItems(fixed_names)

        self.moving_contour.blockSignals(False)
        self.fixed_contour.blockSignals(False)

    def set_preview_context(
        self,
        fixed_image: sitk.Image | None,
        moving_image: sitk.Image | None,
        fixed_structures: dict[str, Any] | None,
        moving_structures: dict[str, Any] | None,
    ) -> None:
        self._preview_context = {
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "fixed_structures": dict(fixed_structures or {}),
            "moving_structures": dict(moving_structures or {}),
        }



    def _structure_image_map(self, structures: dict[str, Any]) -> dict[str, sitk.Image]:
        image_map: dict[str, sitk.Image] = {}
        for name, structure in structures.items():
            image = getattr(structure, "image", structure)
            if isinstance(image, sitk.Image):
                image_map[str(name)] = image
        return image_map

    def preview_mask(self) -> None:
        context = self._preview_context
        fixed_image = context.get("fixed_image")
        moving_image = context.get("moving_image")
        fixed_structures = self._structure_image_map(context.get("fixed_structures") or {})
        moving_structures = self._structure_image_map(context.get("moving_structures") or {})

        if not isinstance(fixed_image, sitk.Image) or not isinstance(moving_image, sitk.Image):
            QMessageBox.information(
                self,
                "Preview unavailable",
                """Load fixed and moving images before previewing a mask.

If the images are already loaded, reopen the registration page so the preview context can refresh.""",
            )
            return

        values = self.get_values()
        mode = str(values.get("mode", "Contour")).strip().lower()
        dilation = int(values.get("dilation", 0))
        erosion = int(values.get("erosion", 0))

        try:
            if mode == "threshold":
                fixed_mask = Masking.threshold_mask(
                    fixed_image,
                    lower_hu_threshold=int(values.get("lower_hu_threshold", -200)),
                    upper_hu_threshold=int(values.get("upper_hu_threshold", 3000)),
                    dilation=dilation,
                    erosion=erosion,
                )
                moving_mask = Masking.threshold_mask(
                    moving_image,
                    lower_hu_threshold=int(values.get("lower_hu_threshold", -200)),
                    upper_hu_threshold=int(values.get("upper_hu_threshold", 3000)),
                    dilation=dilation,
                    erosion=erosion,
                )
            else:
                fixed_name = str(values.get("fixed_contour", "")).strip()
                moving_name = str(values.get("moving_contour", "")).strip()

                if fixed_name not in fixed_structures:
                    raise ValueError(f"Fixed contour '{fixed_name}' not found.")
                if moving_name not in moving_structures:
                    raise ValueError(f"Moving contour '{moving_name}' not found.")

                fixed_mask = Masking.contour_mask(fixed_structures[fixed_name], dilation=dilation, erosion=erosion)
                moving_mask = Masking.contour_mask(moving_structures[moving_name], dilation=dilation, erosion=erosion)
        except Exception as exc:
            QMessageBox.warning(self, "Preview error", str(exc))
            return

        dialog = self._MaskPreviewDialog(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            parent=self,
        )
        dialog.exec()

    def _update_mode_page(self) -> None:
        self.mode_stack.setCurrentIndex(0 if self.mode.currentText() == "Contour" else 1)

    def get_values(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.currentText(),
            "erosion": self.erosion.value(),
            "dilation": self.dilation.value(),
            "moving_contour": self.moving_contour.currentText(),
            "fixed_contour": self.fixed_contour.currentText(),
            "lower_hu_threshold": self.lower_hu_threshold.value(),
            "upper_hu_threshold": self.upper_hu_threshold.value(),
            "crop_based_on_mask": self.crop_based_on_mask.isChecked(),
        }

    def set_values(self, values: Dict[str, Any]) -> None:
        self.mode.setCurrentText(values.get("mode", "Contour"))
        self.erosion.setValue(int(values.get("erosion", 0)))
        self.dilation.setValue(int(values.get("dilation", 0)))
        self.moving_contour.setCurrentText(str(values.get("moving_contour", "")))
        self.fixed_contour.setCurrentText(str(values.get("fixed_contour", "")))
        self.lower_hu_threshold.setValue(int(values.get("lower_hu_threshold", -1000)))
        self.upper_hu_threshold.setValue(int(values.get("upper_hu_threshold", 3000)))
        self.crop_based_on_mask.setChecked(bool(values.get("crop_based_on_mask", False)))
        self._update_mode_page()


    class _MaskPreviewDialog(QDialog):
        def __init__(
            self,
            fixed_image: sitk.Image,
            moving_image: sitk.Image,
            fixed_mask: sitk.Image,
            moving_mask: sitk.Image,
            parent=None,
        ):
            super().__init__(parent)
            self.setWindowTitle("Mask Preview")
            self.resize(1150, 760)

            self._reference_image = fixed_image
            self.fixed_view = SquareCTDisplay()
            self.moving_view = SquareCTDisplay()
            self.controls = EvaluationControlsWidget()
            self.controls.orientation_label.hide()
            self.controls.orientation_combo.hide()
            self.controls.sliceChanged.connect(self._sync_slice)
            self.controls.windowChanged.connect(self._sync_window_level)
            self.controls.levelChanged.connect(self._sync_window_level)
            self.controls.orientationChanged.connect(self._sync_orientation)

            self.fixed_view.set_volume(fixed_image)
            self.moving_view.set_volume(moving_image)
            self.fixed_view.set_render_scale_factor(1.0)
            self.moving_view.set_render_scale_factor(1.0)
            self.fixed_view.set_overlays([
                OverlayLayer(
                    name="Preview Mask",
                    image=fixed_mask,
                    kind="mask",
                    color=(37, 99, 235),
                    opacity=0.45,
                    render_mode="fill",
                )
            ])
            self.moving_view.set_overlays([
                OverlayLayer(
                    name="Preview Mask",
                    image=moving_mask,
                    kind="mask",
                    color=(249, 115, 22),
                    opacity=0.45,
                    render_mode="fill",
                )
            ])

            self._setup_layout()
            self._sync_to_image(self._reference_image)

        def _setup_layout(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            header = QLabel("Preview the selected mask on the fixed and moving CT scans.")
            header.setWordWrap(True)
            layout.addWidget(header)

            row = QHBoxLayout()
            row.setSpacing(12)
            row.addWidget(self._make_viewer_card("Fixed CT", self.fixed_view), 1)
            row.addWidget(self._make_viewer_card("Moving CT", self.moving_view), 1)
            layout.addLayout(row, 1)

            controls_card = QFrame()
            controls_card.setObjectName("card")
            controls_layout = QVBoxLayout(controls_card)
            controls_layout.setContentsMargins(16, 16, 16, 16)
            controls_layout.setSpacing(10)
            title = QLabel("Slice Viewer")
            title.setObjectName("sectionTitle")
            controls_layout.addWidget(title)
            controls_layout.addWidget(self.controls)
            layout.addWidget(controls_card)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
            buttons.accepted.connect(self.accept)
            layout.addWidget(buttons)

        def _make_viewer_card(self, title: str, viewer: SquareCTDisplay) -> QFrame:
            card = QFrame()
            card.setObjectName("card")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            title_label = QLabel(title)
            title_label.setObjectName("sectionTitle")
            layout.addWidget(title_label)
            layout.addWidget(viewer, 1)
            return card

        def _sync_to_image(self, image: sitk.Image) -> None:
            orientation = self.controls.orientation_combo.currentText().lower()
            max_slice = max(
                0,
                self._slice_count(self.fixed_view.volume, orientation),
                self._slice_count(self.moving_view.volume, orientation),
            ) - 1
            current_slice = min(self.controls.slice_spin.value(), max_slice)
            self.controls.set_slice_range(0, max_slice, current_slice, emit=False)
            self.controls.set_window_value(800, emit=False)
            self.controls.set_level_value(0, emit=False)
            self._sync_slice(current_slice)

        def _sync_slice(self, value: int) -> None:
            self.fixed_view.set_slice(value)
            self.moving_view.set_slice(value)
            self.controls.set_slice_value(value, emit=False)

        def _sync_window_level(self) -> None:
            window = self.controls.window_spin.value()
            level = self.controls.level_spin.value()
            self.fixed_view.set_window_level(window, level)
            self.moving_view.set_window_level(window, level)
            self.controls.set_window_value(window, emit=False)
            self.controls.set_level_value(level, emit=False)

        def _sync_orientation(self, orientation: str) -> None:
            self.fixed_view.set_orientation(orientation)
            self.moving_view.set_orientation(orientation)
            max_slice = max(
                0,
                self._slice_count(self.fixed_view.volume, orientation),
                self._slice_count(self.moving_view.volume, orientation),
            ) - 1
            current_slice = min(self.controls.slice_spin.value(), max_slice)
            self.controls.set_slice_range(0, max_slice, current_slice, emit=False)
            self._sync_slice(current_slice)

        @staticmethod
        def _slice_count(volume: object, orientation: str) -> int:
            if volume is None:
                return 1
            try:
                shape = volume.shape  # type: ignore[attr-defined]
            except AttributeError:
                return 1
            if len(shape) < 3:
                return 1
            orientation = orientation.strip().lower()
            if orientation == "coronal":
                return int(shape[1])
            if orientation == "sagittal":
                return int(shape[2])
            return int(shape[0])

class _DelimitedListWidget(QWidget):
    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)

    def setFixedWidth(self, width: int) -> None:
        self.edit.setFixedWidth(width)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, text: str) -> None:
        self.edit.setText(text)


class _IntListWidget(_DelimitedListWidget):
    def get_values(self) -> list[int]:
        raw = self.text()
        if not raw:
            return []
        return [int(part.strip()) for part in raw.split(",") if part.strip()]

    def set_values(self, values: list[int]) -> None:
        self.set_text(", ".join(str(v) for v in values))


class _FloatListWidget(_DelimitedListWidget):
    def get_values(self) -> list[float]:
        raw = self.text()
        if not raw:
            return []
        return [float(part.strip()) for part in raw.split(",") if part.strip()]

    def set_values(self, values: list[float]) -> None:
        self.set_text(", ".join(f"{v:g}" for v in values))


class RigidParamWidget(BaseParamWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramCard")

        FIELD_WIDTH = 240
        LABEL_WIDTH = 140
        FIELD_HEIGHT = QSpinBox().sizeHint().height()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel("Rigid Registration Parameters")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addSpacing(12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        row = 0

        def add_row(text: str, widget: QWidget) -> None:
            nonlocal row
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)
            widget.setFixedWidth(FIELD_WIDTH)
            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignLeft)
            row += 1

        def add_check_row(text: str, checkbox: QCheckBox) -> None:
            nonlocal row
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)

            field_widget = QWidget()
            field_widget.setFixedHeight(FIELD_HEIGHT)

            field_layout = QHBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(0)
            field_layout.addWidget(
                checkbox,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            field_layout.addStretch(1)

            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(field_widget, row, 1)
            row += 1

        self.initial_transform = QComboBox()
        self.initial_transform.addItems(["Moments", "Geometry", "Manual"])
        add_row("Initial Transform", self.initial_transform)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Correlation", "Mean Squares", "Mutual Information"])
        add_row("Metric", self.metric_combo)

        self.sampling_strategy = QComboBox()
        self.sampling_strategy.addItems(["None", "Random"])
        add_row("Sampling Strategy", self.sampling_strategy)

        self.sampling_percentage = QDoubleSpinBox()
        self.sampling_percentage.setRange(0.0, 1.0)
        self.sampling_percentage.setDecimals(3)
        self.sampling_percentage.setSingleStep(0.05)
        add_row("Sampling Percentage", self.sampling_percentage)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1000.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.1)
        add_row("Learning Rate", self.learning_rate)

        self.min_step = QDoubleSpinBox()
        self.min_step.setRange(0.0, 1.0)
        self.min_step.setDecimals(6)
        self.min_step.setSingleStep(0.0001)
        add_row("Min Step", self.min_step)

        self.relaxation_factor = QDoubleSpinBox()
        self.relaxation_factor.setRange(0.0, 1.0)
        self.relaxation_factor.setDecimals(2)
        self.relaxation_factor.setSingleStep(0.05)
        add_row("Relaxation Factor", self.relaxation_factor)

        self.gradient_tolerance = QDoubleSpinBox()
        self.gradient_tolerance.setRange(0.0, 1.0)
        self.gradient_tolerance.setDecimals(10)
        self.gradient_tolerance.setSingleStep(1e-6)
        add_row("Gradient Tolerance", self.gradient_tolerance)

        self.use_multiresolution = QCheckBox()
        self.use_multiresolution.setChecked(True)
        self.use_multiresolution.toggled.connect(self._update_mode_page)
        add_check_row("Use multi-resolution", self.use_multiresolution)

        layout.addLayout(grid)

        self.mode_stack = QStackedWidget()

        # -------------------------
        # Single-resolution page
        # -------------------------
        single_page = QWidget()
        single_layout = QGridLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setHorizontalSpacing(16)
        single_layout.setVerticalSpacing(8)
        single_layout.setColumnStretch(0, 0)
        single_layout.setColumnStretch(1, 1)

        single_label = QLabel("Iterations")
        single_label.setFixedWidth(LABEL_WIDTH)

        self.single_iterations = QSpinBox()
        self.single_iterations.setRange(1, 100000)
        self.single_iterations.setSingleStep(10)
        self.single_iterations.setFixedWidth(FIELD_WIDTH)

        single_layout.addWidget(single_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        single_layout.addWidget(self.single_iterations, 0, 1, Qt.AlignmentFlag.AlignLeft)

        # -------------------------
        # Multi-resolution page
        # -------------------------
        multi_page = QWidget()
        multi_layout = QGridLayout(multi_page)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setHorizontalSpacing(16)
        multi_layout.setVerticalSpacing(8)
        multi_layout.setColumnStretch(0, 0)
        multi_layout.setColumnStretch(1, 1)

        multi_row = 0

        def add_multi_row(text: str, widget: QWidget) -> None:
            nonlocal multi_row
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)
            widget.setFixedWidth(FIELD_WIDTH)
            multi_layout.addWidget(label, multi_row, 0, Qt.AlignmentFlag.AlignVCenter)
            multi_layout.addWidget(widget, multi_row, 1, Qt.AlignmentFlag.AlignLeft)
            multi_row += 1

        self.shrink_factors = _IntListWidget("e.g. 4, 2, 1")
        self.smoothing_sigmas = _FloatListWidget("e.g. 2.0, 1.0, 0.0")
        self.multi_iterations = _IntListWidget("e.g. 200, 150, 100")

        add_multi_row("Shrink Factors", self.shrink_factors)
        add_multi_row("Smoothing Sigmas", self.smoothing_sigmas)
        add_multi_row("Iterations", self.multi_iterations)

        self.mode_stack.addWidget(single_page)
        self.mode_stack.addWidget(multi_page)

        layout.addWidget(self.mode_stack)

        self.crop_to_mask_overlap = QCheckBox()
        self.crop_to_mask_overlap.setChecked(False)

        crop_row = QGridLayout()
        crop_row.setHorizontalSpacing(16)
        crop_row.setVerticalSpacing(8)
        crop_row.setColumnStretch(0, 0)
        crop_row.setColumnStretch(1, 1)

        crop_label = QLabel("Crop to Mask Overlap")
        crop_label.setFixedWidth(LABEL_WIDTH)

        crop_field = QWidget()
        crop_field.setFixedHeight(FIELD_HEIGHT)

        crop_field_layout = QHBoxLayout(crop_field)
        crop_field_layout.setContentsMargins(0, 0, 0, 0)
        crop_field_layout.setSpacing(0)
        crop_field_layout.addWidget(
            self.crop_to_mask_overlap,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        crop_field_layout.addStretch(1)

        crop_row.addWidget(crop_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        crop_row.addWidget(crop_field, 0, 1)

        layout.addLayout(crop_row)
        layout.addStretch(1)

        self._update_mode_page()

    def _update_mode_page(self) -> None:
        self.mode_stack.setCurrentIndex(1 if self.use_multiresolution.isChecked() else 0)

    def get_values(self) -> Dict[str, Any]:
        if self.use_multiresolution.isChecked():
            return {
                "initial_transform": self.initial_transform.currentText(),
                "metric": self.metric_combo.currentText(),
                "sampling_strategy": self.sampling_strategy.currentText(),
                "sampling_percentage": self.sampling_percentage.value(),
                "learning_rate": self.learning_rate.value(),
                "min_step": self.min_step.value(),
                "relaxation_factor": self.relaxation_factor.value(),
                "gradient_tolerance": self.gradient_tolerance.value(),
                "use_multiresolution": True,
                "iterations": self.multi_iterations.get_values(),
                "shrink_factors": self.shrink_factors.get_values(),
                "smoothing_sigmas": self.smoothing_sigmas.get_values(),
                "crop_to_mask_overlap": self.crop_to_mask_overlap.isChecked(),
            }

        return {
            "initial_transform": self.initial_transform.currentText(),
            "metric": self.metric_combo.currentText(),
            "sampling_strategy": self.sampling_strategy.currentText(),
            "sampling_percentage": self.sampling_percentage.value(),
            "learning_rate": self.learning_rate.value(),
            "min_step": self.min_step.value(),
            "relaxation_factor": self.relaxation_factor.value(),
            "gradient_tolerance": self.gradient_tolerance.value(),
            "use_multiresolution": False,
            "iterations": self.single_iterations.value(),
            "shrink_factors": [1],
            "smoothing_sigmas": [0.0],
            "crop_to_mask_overlap": self.crop_to_mask_overlap.isChecked(),
        }

    def set_values(self, values: Dict[str, Any]) -> None:
        self.initial_transform.setCurrentText(values.get("initial_transform", "Moments"))
        self.metric_combo.setCurrentText(values.get("metric", "Correlation"))
        self.sampling_strategy.setCurrentText(values.get("sampling_strategy", "None"))
        self.sampling_percentage.setValue(float(values.get("sampling_percentage", 0.25)))
        self.learning_rate.setValue(float(values.get("learning_rate", 1.0)))
        self.min_step.setValue(float(values.get("min_step", 1e-4)))
        self.relaxation_factor.setValue(float(values.get("relaxation_factor", 0.3)))
        self.gradient_tolerance.setValue(float(values.get("gradient_tolerance", 1e-6)))

        self.crop_to_mask_overlap.setChecked(bool(values.get("crop_to_mask_overlap", False)))

        use_mr = bool(values.get("use_multiresolution", True))
        self.use_multiresolution.setChecked(use_mr)

        iterations = values.get("iterations", [200, 150, 100])

        if use_mr:
            if isinstance(iterations, int):
                iterations = [iterations]
            self.multi_iterations.set_values([int(v) for v in iterations])
            self.shrink_factors.set_values([int(v) for v in values.get("shrink_factors", [4, 2, 1])])
            self.smoothing_sigmas.set_values([float(v) for v in values.get("smoothing_sigmas", [2.0, 1.0, 0.0])])
        else:
            if isinstance(iterations, list):
                iterations = iterations[0] if iterations else 200
            self.single_iterations.setValue(int(iterations))

        self._update_mode_page()


class DemonsParamWidget(BaseParamWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paramCard")

        FIELD_WIDTH = 200
        LABEL_WIDTH = 180
        FIELD_HEIGHT = QSpinBox().sizeHint().height()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Demons Parameters")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addSpacing(12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        row = 0

        def add_row(text: str, widget: QWidget) -> None:
            nonlocal row
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)
            widget.setFixedWidth(FIELD_WIDTH)
            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignLeft)
            row += 1

        self.method = QComboBox()
        self.method.addItems(
            [
                "Diffeomorphic",
                "Symmetric forces",
                "Fast symmetric forces",
            ]
        )
        add_row("Method", self.method)

        self.field_smoothing_sigma = QDoubleSpinBox()
        self.field_smoothing_sigma.setRange(0.0, 1000.0)
        self.field_smoothing_sigma.setDecimals(6)
        self.field_smoothing_sigma.setSingleStep(0.1)
        add_row("Field Smoothing Sigma", self.field_smoothing_sigma)

        self.max_rms_error = QDoubleSpinBox()
        self.max_rms_error.setRange(0.0, 1000.0)
        self.max_rms_error.setDecimals(6)
        self.max_rms_error.setSingleStep(0.001)
        add_row("Max RMS Error", self.max_rms_error)

        self.hu_difference_threshold = QDoubleSpinBox()
        self.hu_difference_threshold.setRange(0.0, 5000.0)
        self.hu_difference_threshold.setDecimals(6)
        self.hu_difference_threshold.setSingleStep(1.0)
        add_row("HU Difference Threshold", self.hu_difference_threshold)

        self.stopping_window = QSpinBox()
        self.stopping_window.setRange(2, 100000)
        self.stopping_window.setSingleStep(1)
        add_row("Stopping Window", self.stopping_window)

        self.stopping_tolerance = QDoubleSpinBox()
        self.stopping_tolerance.setRange(0.0, 1.0)
        self.stopping_tolerance.setDecimals(8)
        self.stopping_tolerance.setSingleStep(1e-5)
        add_row("Stopping Tolerance", self.stopping_tolerance)

        self.use_multiresolution = QCheckBox()
        self.use_multiresolution.toggled.connect(self._update_mode_page)

        use_mr_label = QLabel("Use Multi-Resolution")
        use_mr_label.setFixedWidth(LABEL_WIDTH)

        use_mr_field = QWidget()
        use_mr_field.setFixedHeight(FIELD_HEIGHT)

        use_mr_layout = QHBoxLayout(use_mr_field)
        use_mr_layout.setContentsMargins(0, 0, 0, 0)
        use_mr_layout.setSpacing(0)
        use_mr_layout.addWidget(self.use_multiresolution, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        use_mr_layout.addStretch(1)

        grid.addWidget(use_mr_label, row, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(use_mr_field, row, 1)
        row += 1

        layout.addLayout(grid)

        self.mode_stack = QStackedWidget()

        # -------------------------
        # Single-resolution page
        # -------------------------
        single_page = QWidget()
        single_layout = QGridLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setHorizontalSpacing(16)
        single_layout.setColumnStretch(0, 0)
        single_layout.setColumnStretch(1, 1)

        single_label = QLabel("Iterations")
        single_label.setFixedWidth(LABEL_WIDTH)

        self.single_iterations = QSpinBox()
        self.single_iterations.setRange(1, 100000)
        self.single_iterations.setSingleStep(10)
        self.single_iterations.setFixedWidth(FIELD_WIDTH)

        single_layout.addWidget(single_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        single_layout.addWidget(self.single_iterations, 0, 1, Qt.AlignmentFlag.AlignLeft)

        # -------------------------
        # Multi-resolution page
        # -------------------------
        multi_page = QWidget()
        multi_layout = QGridLayout(multi_page)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setHorizontalSpacing(16)
        multi_layout.setVerticalSpacing(10)
        multi_layout.setColumnStretch(0, 0)
        multi_layout.setColumnStretch(1, 1)

        multi_row = 0

        def add_multi_row(text: str, widget: QWidget) -> None:
            nonlocal multi_row
            label = QLabel(text)
            label.setFixedWidth(LABEL_WIDTH)
            widget.setFixedWidth(FIELD_WIDTH)
            multi_layout.addWidget(label, multi_row, 0, Qt.AlignmentFlag.AlignVCenter)
            multi_layout.addWidget(widget, multi_row, 1, Qt.AlignmentFlag.AlignLeft)
            multi_row += 1

        self.shrink_factors = _IntListWidget("e.g. 4, 2, 1")
        self.smoothing_sigmas = _FloatListWidget("e.g. 2.0, 1.0, 0.0")
        self.multi_iterations = _IntListWidget("e.g. 200, 150, 100")

        add_multi_row("Shrink Factors", self.shrink_factors)
        add_multi_row("Smoothing Sigmas", self.smoothing_sigmas)
        add_multi_row("Iterations", self.multi_iterations)

        self.mode_stack.addWidget(single_page)
        self.mode_stack.addWidget(multi_page)

        layout.addWidget(self.mode_stack)
        layout.addStretch(1)

        self._update_mode_page()

    def _update_mode_page(self) -> None:
        self.mode_stack.setCurrentIndex(1 if self.use_multiresolution.isChecked() else 0)

    def get_values(self) -> Dict[str, Any]:
        if self.use_multiresolution.isChecked():
            return {
                "method": self.method.currentText(),
                "use_multiresolution": True,
                "iterations": self.multi_iterations.get_values(),
                "shrink_factors": self.shrink_factors.get_values(),
                "smoothing_sigmas": self.smoothing_sigmas.get_values(),
                "field_smoothing_sigma": self.field_smoothing_sigma.value(),
                "max_rms_error": self.max_rms_error.value(),
                "hu_difference_threshold": self.hu_difference_threshold.value(),
                "stopping_window": self.stopping_window.value(),
                "stopping_tolerance": self.stopping_tolerance.value(),
            }

        return {
            "method": self.method.currentText(),
            "use_multiresolution": False,
            "iterations": self.single_iterations.value(),
            "shrink_factors": [1],
            "smoothing_sigmas": [0.0],
            "field_smoothing_sigma": self.field_smoothing_sigma.value(),
            "max_rms_error": self.max_rms_error.value(),
            "hu_difference_threshold": self.hu_difference_threshold.value(),
            "stopping_window": self.stopping_window.value(),
            "stopping_tolerance": self.stopping_tolerance.value(),
        }

    def set_values(self, values: Dict[str, Any]) -> None:
        self.method.setCurrentText(values.get("method", "Diffeomorphic"))
        self.field_smoothing_sigma.setValue(float(values.get("field_smoothing_sigma", 1.0)))
        self.max_rms_error.setValue(float(values.get("max_rms_error", 0.1)))
        self.hu_difference_threshold.setValue(float(values.get("hu_difference_threshold", 1.5)))
        self.stopping_window.setValue(int(values.get("stopping_window", 10)))
        self.stopping_tolerance.setValue(float(values.get("stopping_tolerance", 1e-5)))

        use_mr = bool(values.get("use_multiresolution", True))
        self.use_multiresolution.setChecked(use_mr)

        iterations = values.get("iterations", [200, 150, 100])

        if use_mr:
            if isinstance(iterations, int):
                iterations = [iterations]
            self.multi_iterations.set_values([int(v) for v in iterations])

            self.shrink_factors.set_values(
                [int(v) for v in values.get("shrink_factors", [4, 2, 1])]
            )
            self.smoothing_sigmas.set_values(
                [float(v) for v in values.get("smoothing_sigmas", [2.0, 1.0, 0.0])]
            )
        else:
            if isinstance(iterations, list):
                iterations = iterations[0] if iterations else 200
            self.single_iterations.setValue(int(iterations))

        self._update_mode_page()


class GenericParamWidget(BaseParamWidget):
    def __init__(self, params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setObjectName("paramCard")
        self._widgets: Dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Parameters")
        title.setObjectName("paramTitle")
        layout.addWidget(title)

        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self.form.setHorizontalSpacing(14)
        self.form.setVerticalSpacing(10)
        layout.addLayout(self.form)
        layout.addStretch(1)

        for key, value in params.items():
            widget = self._make_widget(value)
            self._widgets[key] = widget
            self.form.addRow(key, widget)

    def _make_widget(self, value: Any) -> QWidget:
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            return w
        if isinstance(value, int) and not isinstance(value, bool):
            w = QSpinBox()
            w.setRange(-10_000_000, 10_000_000)
            w.setValue(value)
            return w
        if isinstance(value, float):
            w = QDoubleSpinBox()
            w.setRange(-10_000_000.0, 10_000_000.0)
            w.setDecimals(6)
            w.setSingleStep(0.1)
            w.setValue(value)
            return w

        w = QLineEdit()
        w.setText(str(value))
        return w

    def get_values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text()
        return out

    def set_values(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            widget = self._widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))


PARAM_WIDGETS = {
    "Masking": MaskingParamWidget,
    "Rigid Registration": RigidParamWidget,
    "Demons Registration": DemonsParamWidget,
}


def _params_to_dict(params: Any) -> Dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if is_dataclass(params):
        return asdict(params)
    if hasattr(params, "items"):
        return dict(params)
    if hasattr(params, "__dict__"):
        return {k: v for k, v in vars(params).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported parameter container: {type(params).__name__}")


class ParamDialog(QDialog):
    def __init__(self, step_name: str, params: Dict[str, Any], parent=None, preview_context: dict[str, Any] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit options: {step_name}")

        widget_cls = PARAM_WIDGETS.get(step_name, GenericParamWidget)
        normalized_params = _params_to_dict(params)
        self.editor = widget_cls(normalized_params, self) if widget_cls is GenericParamWidget else widget_cls(self)

        if widget_cls is not GenericParamWidget:
            self.editor.set_values(normalized_params)
            if preview_context is not None and hasattr(self.editor, "set_preview_context"):
                self.editor.set_preview_context(
                    preview_context.get("fixed_image"),
                    preview_context.get("moving_image"),
                    preview_context.get("fixed_structures"),
                    preview_context.get("moving_structures"),
                )

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.adjustSize()
        self.setFixedSize(self.sizeHint())

        self.setStyleSheet(
            """
            QDialog {
                background: #f7f9fc;
            }
            QFrame#paramCard {
                background: white;
                border: 1px solid #d9dee7;
                border-radius: 14px;
            }
            QLabel#paramTitle {
                font-size: 18px;
                font-weight: 700;
                color: #1f2a3d;
            }
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: white;
                border: 1px solid #d9dee7;
                border-radius: 10px;
                padding: 6px 10px;
                color: #1f2a3d;
            }
            QPushButton {
                min-width: 88px;
            }
            QPushButton:hover:!disabled {
                background: #f6f8fb;
                border-color: #cbd5e1;
            }
            QPushButton:pressed:!disabled {
                background: #e8edf4;
                border-color: #b8c2cf;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background: #f8fafc;
                border-color: #e2e8f0;
            }
            QDialogButtonBox QPushButton {
                padding: 8px 14px;
                min-width: 92px;
            }
            QCheckBox {
                spacing: 8px;
            }
            """
        )

    def get_params(self) -> Dict[str, Any]:
        return self.editor.get_values()