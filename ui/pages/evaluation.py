
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import SimpleITK as sitk
from PySide6.QtCore import Qt
from ui.widgets.chrome import ArrowComboBox as QComboBox
from ui.widgets.chrome import SegmentedButtonBar
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from registration.metrics.dose import DVH, combine_dose_images_for_display, convert_dose_image_for_display, metric_result_for_name
from registration.metrics.dvh_defaults import default_dvh_settings_for_regions
from ui.workers.evaluation import EvaluationCase
from ui.widgets.dvh_plot import DvhPlotWidget
from ui.widgets.evaluation_controls import EvaluationControlsWidget
from ui.widgets.image_viewer import OverlayLayer, SquareCTDisplay
from ui.widgets.metrics_table import MetricsTableWidget
from ui.widgets.dose_metrics_table import DoseMetricsTableWidget

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class EvaluationPage(QWidget):
    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.main_window = main_window
        self.case: EvaluationCase | None = None
        self._selected_contour_region: str | None = None
        self._selected_dose_region: str | None = None
        self._dose_region_settings: dict[str, dict[str, object]] = {}
        self._dose_row_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self._dose_visual_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self._last_dose_visual_key: tuple[object, ...] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(16)

        top_card = QFrame()
        top_card.setObjectName("card")
        top_layout = QGridLayout(top_card)
        top_layout.setContentsMargins(18, 14, 18, 14)
        top_layout.setHorizontalSpacing(12)

        back_button = QPushButton("Back")
        back_button.setMinimumHeight(40)
        back_button.clicked.connect(self._on_back)

        title = QLabel("Evaluate Registration")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        spacer = QWidget()
        spacer.setFixedWidth(back_button.sizeHint().width())

        top_layout.addWidget(back_button, 0, 0)
        top_layout.addWidget(title, 0, 1)
        top_layout.addWidget(spacer, 0, 2)
        top_layout.setColumnStretch(0, 0)
        top_layout.setColumnStretch(1, 1)
        top_layout.setColumnStretch(2, 0)


        self.tab_bar = SegmentedButtonBar()
        self.tab_bar.add_tab("Image")
        self.tab_bar.add_tab("Image Overlay")
        self.tab_bar.add_tab("Contours")
        self.tab_bar.add_tab("Dose")
        self.tab_bar.currentChanged.connect(self._refresh_visible_tab)

        self.tab_stack = QStackedWidget()
        self.image_tab = self._build_image_tab()
        self.overlay_tab = self._build_overlay_tab()
        self.contour_tab = self._build_contour_tab()
        self.dose_tab = self._build_dose_tab()
        self.tab_stack.addWidget(self.image_tab)
        self.tab_stack.addWidget(self.overlay_tab)
        self.tab_stack.addWidget(self.contour_tab)
        self.tab_stack.addWidget(self.dose_tab)

        tabs_card = QFrame()
        tabs_card.setObjectName("card")
        tabs_card_layout = QVBoxLayout(tabs_card)
        tabs_card_layout.setContentsMargins(10, 10, 10, 10)
        tabs_card_layout.setSpacing(10)
        tabs_card_layout.addWidget(self.tab_bar)
        page_layout.addWidget(top_card)
        page_layout.addWidget(tabs_card)
        page_layout.addWidget(self.tab_stack, 1)

    def _build_image_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        viewers_row = QHBoxLayout()
        viewers_row.setSpacing(16)

        self.fixed_view = SquareCTDisplay()
        self.rigid_view = SquareCTDisplay()
        self.final_view = SquareCTDisplay()

        viewers_row.addWidget(self._create_viewer_card("Fixed CT", self.fixed_view), 1)
        viewers_row.addWidget(self._create_viewer_card("Rigid Registration", self.rigid_view), 1)
        viewers_row.addWidget(self._create_viewer_card("Final Registration", self.final_view), 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)

        controls_title = QLabel("Viewer Controls")
        controls_title.setObjectName("sectionTitle")

        self.image_controls = EvaluationControlsWidget()
        self.image_controls.sliceChanged.connect(self._sync_slice)
        self.image_controls.windowChanged.connect(self._sync_window_level)
        self.image_controls.levelChanged.connect(self._sync_window_level)
        self.image_controls.orientationChanged.connect(self._sync_orientation)

        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(self.image_controls)

        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(10)

        self.image_metrics_table = MetricsTableWidget("Image Metrics")
        metrics_layout.addWidget(self.image_metrics_table, 1)

        bottom_row.addWidget(controls_card, 1)
        bottom_row.addWidget(metrics_card, 1)

        layout.addLayout(viewers_row, 1)
        layout.addLayout(bottom_row)
        return tab

    def _build_overlay_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        viewers_row = QHBoxLayout()
        viewers_row.setSpacing(16)

        self.overlay_rigid_view = SquareCTDisplay()
        self.overlay_final_view = SquareCTDisplay()

        viewers_row.addWidget(self._create_viewer_card("Rigid over Fixed", self.overlay_rigid_view), 1)
        viewers_row.addWidget(self._create_viewer_card("Final over Fixed", self.overlay_final_view), 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)

        controls_title = QLabel("Viewer Controls")
        controls_title.setObjectName("sectionTitle")

        self.overlay_controls = EvaluationControlsWidget()
        self.overlay_controls.sliceChanged.connect(self._sync_slice)
        self.overlay_controls.windowChanged.connect(self._sync_window_level)
        self.overlay_controls.levelChanged.connect(self._sync_window_level)
        self.overlay_controls.orientationChanged.connect(self._sync_orientation)

        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(self.overlay_controls)

        options_card = QFrame()
        options_card.setObjectName("card")
        options_layout = QVBoxLayout(options_card)
        options_layout.setContentsMargins(16, 16, 16, 16)
        options_layout.setSpacing(10)

        options_title = QLabel("Overlay Controls")
        options_title.setObjectName("sectionTitle")
        options_layout.addWidget(options_title)

        transparency_row = QGridLayout()
        transparency_row.setContentsMargins(0, 0, 0, 0)
        transparency_row.setHorizontalSpacing(10)
        transparency_row.setVerticalSpacing(8)

        transparency_label = QLabel("Transparency")
        transparency_label.setMinimumWidth(110)
        self.overlay_transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_transparency_slider.setRange(0, 100)
        self.overlay_transparency_slider.setValue(55)
        self.overlay_transparency_slider.valueChanged.connect(self._update_overlay_views)
        self.overlay_transparency_value = QLabel("55%")
        self.overlay_transparency_value.setMinimumWidth(50)
        self.overlay_transparency_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.overlay_transparency_slider.valueChanged.connect(
            lambda value: self.overlay_transparency_value.setText(f"{int(value)}%")
        )
        transparency_row.addWidget(transparency_label, 0, 0)
        transparency_row.addWidget(self.overlay_transparency_slider, 0, 1)
        transparency_row.addWidget(self.overlay_transparency_value, 0, 2)

        self.overlay_hue_checkbox = QCheckBox("Hue comparison")
        self.overlay_hue_checkbox.setChecked(True)
        self.overlay_hue_checkbox.toggled.connect(self._on_overlay_render_options_changed)

        self.overlay_difference_checkbox = QCheckBox("Difference map")
        self.overlay_difference_checkbox.toggled.connect(self._on_overlay_difference_toggled)

        options_layout.addLayout(transparency_row)
        options_layout.addWidget(self.overlay_hue_checkbox)
        options_layout.addWidget(self.overlay_difference_checkbox)
        options_layout.addStretch(1)

        bottom_row.addWidget(controls_card, 1)
        bottom_row.addWidget(options_card, 1)

        layout.addLayout(viewers_row, 1)
        layout.addLayout(bottom_row)
        return tab

    def _build_contour_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(16)

        self.contour_view = SquareCTDisplay()
        body.addWidget(self._create_viewer_card("Fixed CT with contour overlays", self.contour_view), 3)

        side_column = QVBoxLayout()
        side_column.setContentsMargins(0, 0, 0, 0)
        side_column.setSpacing(16)

        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)

        controls_title = QLabel("Viewer Controls")
        controls_title.setObjectName("sectionTitle")

        self.contour_controls = EvaluationControlsWidget()
        self.contour_controls.sliceChanged.connect(self._sync_slice)
        self.contour_controls.windowChanged.connect(self._sync_window_level)
        self.contour_controls.levelChanged.connect(self._sync_window_level)
        self.contour_controls.orientationChanged.connect(self._sync_orientation)

        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(self.contour_controls)

        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(10)

        info = QLabel("Select a contour row to update the overlay region.")
        info.setObjectName("placeholder")
        info.setWordWrap(True)
        metrics_layout.addWidget(info)

        self.contour_metrics_table = MetricsTableWidget("Contour Comparison")
        self.contour_metrics_table.enable_row_selection(True)
        self.contour_metrics_table.table.itemSelectionChanged.connect(self._on_contour_selection_changed)
        self.contour_metrics_table.table.cellClicked.connect(self._on_contour_cell_clicked)
        metrics_layout.addWidget(self.contour_metrics_table, 1)

        side_column.addWidget(controls_card, 0)
        side_column.addWidget(metrics_card, 1)

        side_widget = QWidget()
        side_widget.setLayout(side_column)
        body.addWidget(side_widget, 2)

        layout.addLayout(body, 1)
        return tab

    def _build_dose_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        body = QGridLayout()
        body.setHorizontalSpacing(16)
        body.setVerticalSpacing(16)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        self.dose_view = SquareCTDisplay()
        left_layout.addWidget(self._create_viewer_card("Radiotherapy Dose", self.dose_view), 1)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(16)

        # Viewer controls card (slice / window / level / orientation)
        viewer_controls_card = QFrame()
        viewer_controls_card.setObjectName("card")
        viewer_controls_layout = QVBoxLayout(viewer_controls_card)
        viewer_controls_layout.setContentsMargins(16, 16, 16, 16)
        viewer_controls_layout.setSpacing(10)

        viewer_controls_title = QLabel("Viewer Controls")
        viewer_controls_title.setObjectName("sectionTitle")
        viewer_controls_layout.addWidget(viewer_controls_title)

        self.dose_controls = EvaluationControlsWidget()
        self.dose_controls.sliceChanged.connect(self._sync_slice)
        self.dose_controls.windowChanged.connect(self._sync_window_level)
        self.dose_controls.levelChanged.connect(self._sync_window_level)
        self.dose_controls.orientationChanged.connect(self._sync_orientation)
        viewer_controls_layout.addWidget(self.dose_controls)

        # Dose settings card (stage / dose type / fractions)
        dose_controls_card = QFrame()
        dose_controls_card.setObjectName("card")
        dose_controls_layout = QVBoxLayout(dose_controls_card)
        dose_controls_layout.setContentsMargins(16, 16, 16, 16)
        dose_controls_layout.setSpacing(10)

        dose_controls_title = QLabel("Dose Controls")
        dose_controls_title.setObjectName("sectionTitle")
        dose_controls_layout.addWidget(dose_controls_title)

        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(10)

        self.dose_stage_combo = QComboBox()
        self.dose_stage_combo.addItem("Fixed", "fixed")
        self.dose_stage_combo.addItem("Rigid", "rigid")
        self.dose_stage_combo.addItem("Final", "final")
        self.dose_stage_combo.addItem("Rigid Combined", "rigid_combined")
        self.dose_stage_combo.addItem("Final Combined", "final_combined")
        self.dose_stage_combo.setCurrentIndex(self.dose_stage_combo.findData("final_combined"))
        self.dose_stage_combo.currentIndexChanged.connect(lambda *_: self._update_dose_tab())

        self.dose_type_combo = QComboBox()
        self.dose_type_combo.addItem("Physical Dose", "physical")
        self.dose_type_combo.addItem("BED", "bed")
        self.dose_type_combo.addItem("EQD2", "eqd2")
        self.dose_type_combo.currentIndexChanged.connect(lambda *_: self._update_dose_tab())

        self.fixed_fractions_spin = QSpinBox()
        self.fixed_fractions_spin.setRange(1, 1000)
        self.fixed_fractions_spin.setValue(30)
        self.fixed_fractions_spin.valueChanged.connect(lambda *_: self._update_dose_tab())

        self.moving_fractions_spin = QSpinBox()
        self.moving_fractions_spin.setRange(1, 1000)
        self.moving_fractions_spin.setValue(30)
        self.moving_fractions_spin.valueChanged.connect(lambda *_: self._update_dose_tab())

        self._add_form_row(form, 0, "Stage", self.dose_stage_combo)
        self._add_form_row(form, 1, "Dose Type", self.dose_type_combo)
        self._add_form_row(form, 2, "Fixed Fractions", self.fixed_fractions_spin)
        self._add_form_row(form, 3, "Moving Fractions", self.moving_fractions_spin)
        dose_controls_layout.addLayout(form)

        controls_row.addWidget(viewer_controls_card, 1)
        controls_row.addWidget(dose_controls_card, 1)
        left_layout.addLayout(controls_row)

        # Metrics card with table above DVH
        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(10)

        self.dose_metrics_table = DoseMetricsTableWidget("Contours")
        self.dose_metrics_table.regionSelected.connect(self._on_dose_region_selected)
        self.dose_metrics_table.editAlphaBetaRequested.connect(self._on_edit_alpha_beta_requested)
        self.dose_metrics_table.addMetricRequested.connect(self._on_add_metric_requested)

        self.dvh_plot = DvhPlotWidget()

        metrics_layout.addWidget(self.dose_metrics_table, 3)
        dvh_title = QLabel("DVH")
        dvh_title.setObjectName("sectionTitle")
        metrics_layout.addWidget(dvh_title)
        metrics_layout.addWidget(self.dvh_plot, 2)

        body.addWidget(left_panel, 0, 0, 2, 1)
        body.addWidget(metrics_card, 0, 1, 2, 1)
        body.setColumnStretch(0, 2)
        body.setColumnStretch(1, 1)
        body.setRowStretch(0, 4)
        body.setRowStretch(1, 3)

        layout.addLayout(body, 1)
        return tab

    def _create_viewer_card(self, title_text: str, viewer: SquareCTDisplay) -> QFrame:
        card = QFrame()
        card.setObjectName("viewerCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("sectionTitle")

        layout.addWidget(title)
        layout.addWidget(viewer, 1)
        return card

    def _add_form_row(self, layout: QGridLayout, row: int, label_text: str, widget: QWidget) -> None:
        label = QLabel(label_text)
        label.setMinimumWidth(110)
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)


    def set_images(
        self,
        fixed: sitk.Image,
        rigid: sitk.Image,
        final: sitk.Image,
    ) -> None:
        self.set_evaluation_case(
            EvaluationCase(
                fixed_image=fixed,
                rigid_image=rigid,
                final_image=final,
            )
        )

    def set_evaluation_case(self, case: EvaluationCase) -> None:
        self.case = case

        self._set_view_render_scale(1.0)

        self.fixed_view.set_volume(case.fixed_image)
        self.rigid_view.set_volume(case.rigid_image)
        self.final_view.set_volume(case.final_image)
        self.overlay_rigid_view.set_volume(case.fixed_image)
        self.overlay_rigid_view.set_comparison_image(case.rigid_image)
        self.overlay_final_view.set_volume(case.fixed_image)
        self.overlay_final_view.set_comparison_image(case.final_image)
        self.contour_view.set_volume(case.fixed_image)
        self.dose_view.set_volume(case.fixed_image)

        self._selected_contour_region = None
        self._selected_dose_region = None
        self._dose_region_settings = {}
        self._dose_row_cache.clear()
        self._dose_visual_cache.clear()
        self._last_dose_visual_key = None
        self._sync_all_controls_to_case()
        self._sync_orientation(self.image_controls.orientation_combo.currentText().lower())
        self._update_overlay_views()
        self._refresh_all_tabs()
        self.tab_bar.set_current_index(0, emit=False)
        self.tab_stack.setCurrentIndex(0)

    def _set_view_render_scale(self, factor: float) -> None:
        for viewer in (self.fixed_view, self.rigid_view, self.final_view, self.overlay_rigid_view, self.overlay_final_view, self.contour_view, self.dose_view):
            viewer.set_render_scale_factor(factor)

    def _sync_all_controls_to_case(self) -> None:
        if self.case is None:
            return

        orientation = self.image_controls.orientation_combo.currentText().lower()
        max_slice = max(
            0,
            self._slice_count(self.fixed_view.volume, orientation),
            self._slice_count(self.rigid_view.volume, orientation),
            self._slice_count(self.final_view.volume, orientation),
        ) - 1
        current_slice = min(self.image_controls.slice_spin.value(), max_slice)
        self.image_controls.set_slice_range(0, max_slice, current_slice, emit=False)
        self.overlay_controls.set_slice_range(0, max_slice, current_slice, emit=False)
        self.contour_controls.set_slice_range(0, max_slice, current_slice, emit=False)
        self.dose_controls.set_slice_range(0, max_slice, current_slice, emit=False)

    def _refresh_all_tabs(self) -> None:
        self._update_image_metrics()
        self._update_contour_tab()
        self._update_dose_tab()

    def _refresh_visible_tab(self, index: int) -> None:
        self.tab_stack.setCurrentIndex(index)
        if index == 2:
            self._update_contour_tab()
        elif index == 3:
            self._update_dose_tab()

    def _sync_slice(self, value: int) -> None:
        for viewer in (self.fixed_view, self.rigid_view, self.final_view, self.overlay_rigid_view, self.overlay_final_view, self.contour_view, self.dose_view):
            viewer.set_slice(value)
        for controls in (self.image_controls, self.overlay_controls, self.contour_controls, self.dose_controls):
            controls.set_slice_value(value, emit=False)

    def _sync_window_level(self) -> None:
        controls = self._active_controls()
        if controls is None:
            return
        window = controls.window_spin.value()
        level = controls.level_spin.value()
        for viewer in (self.fixed_view, self.rigid_view, self.final_view, self.overlay_rigid_view, self.overlay_final_view, self.contour_view, self.dose_view):
            viewer.set_window_level(window, level)
        for controls_widget in (self.image_controls, self.overlay_controls, self.contour_controls, self.dose_controls):
            controls_widget.set_window_value(window, emit=False)
            controls_widget.set_level_value(level, emit=False)
        self._update_overlay_views()

    def _sync_orientation(self, orientation: str) -> None:
        for viewer in (self.fixed_view, self.rigid_view, self.final_view, self.overlay_rigid_view, self.overlay_final_view, self.contour_view, self.dose_view):
            viewer.set_orientation(orientation)
        for controls_widget in (self.image_controls, self.overlay_controls, self.contour_controls, self.dose_controls):
            controls_widget.set_orientation(orientation, emit=False)
        self._update_overlay_views()
        self._sync_all_controls_to_case()
        controls = self._active_controls()
        if controls is not None:
            self._sync_slice(controls.slice_spin.value())

    def _active_controls(self) -> EvaluationControlsWidget | None:
        current_index = self.tab_stack.currentIndex()
        if current_index == 1:
            return self.overlay_controls
        if current_index == 2:
            return self.contour_controls
        if current_index == 3:
            return self.dose_controls
        return self.image_controls

    def _overlay_transparency(self) -> float:
        return float(self.overlay_transparency_slider.value()) / 100.0

    def _update_overlay_views(self) -> None:
        if self.case is None:
            for viewer in (self.overlay_rigid_view, self.overlay_final_view):
                viewer.clear_comparison_image()
            return

        opacity = self._overlay_transparency()
        hue_enabled = self.overlay_hue_checkbox.isChecked() and not self.overlay_difference_checkbox.isChecked()
        mode = "difference" if self.overlay_difference_checkbox.isChecked() else "overlay"

        fixed_color = (253, 224, 71)
        moving_color = (59, 130, 246)

        self.overlay_rigid_view.set_comparison_properties(
            mode=mode,
            opacity=opacity,
            hue_enabled=hue_enabled,
            base_color=fixed_color,
            color=moving_color,
        )
        self.overlay_final_view.set_comparison_properties(
            mode=mode,
            opacity=opacity,
            hue_enabled=hue_enabled,
            base_color=fixed_color,
            color=moving_color,
        )

    def _on_overlay_render_options_changed(self, *_: object) -> None:
        if self.overlay_difference_checkbox.isChecked():
            self.overlay_hue_checkbox.setEnabled(False)
        else:
            self.overlay_hue_checkbox.setEnabled(True)
        self._update_overlay_views()

    def _on_overlay_difference_toggled(self, checked: bool) -> None:
        self.overlay_hue_checkbox.setEnabled(not checked)
        self._update_overlay_views()

    def _update_image_metrics(self) -> None:
        if self.case is None or self.case.image_metrics is None:
            self.image_metrics_table.clear_table()
            return

        self.image_metrics_table.set_dataframe(self.case.image_metrics)

    def _update_contour_tab(self) -> None:
        if self.case is None or self.case.contour_metrics is None or self.case.contour_metrics.empty:
            self.contour_view.clear_overlays()
            self.contour_metrics_table.clear_table()
            self._selected_contour_region = None
            return

        metrics = self.case.contour_metrics.copy().reset_index(drop=True)
        self.contour_metrics_table.set_dataframe(metrics)
        self.contour_metrics_table.enable_row_selection(True)

        if self._selected_contour_region is None:
            self._selected_contour_region = str(metrics.iloc[0]["Region"])

        selected_row_index = self._row_index_for_region(self._selected_contour_region)
        if selected_row_index is not None:
            self.contour_metrics_table.table.blockSignals(True)
            self.contour_metrics_table.table.selectRow(selected_row_index)
            self.contour_metrics_table.table.blockSignals(False)

        self._render_contour_overlays()

    def _render_contour_overlays(self) -> None:
        if self.case is None:
            self.contour_view.clear_overlays()
            return

        region = self._selected_contour_region or self._first_contour_region()
        if region is None:
            self.contour_view.clear_overlays()
            return

        fixed_contour = self.case.fixed_structures.get(region)
        rigid_contour = self.case.rigid_structures.get(region)
        final_contour = self.case.final_structures.get(region)

        overlay_layers: list[OverlayLayer] = []
        if fixed_contour is not None:
            overlay_layers.append(
                OverlayLayer(
                    name=f"Fixed::{region}",
                    image=fixed_contour,
                    kind="mask",
                    color=(37, 99, 235),
                    opacity=0.8,
                    render_mode="outline",
                )
            )
        if rigid_contour is not None:
            overlay_layers.append(
                OverlayLayer(
                    name=f"Rigid Registration::{region}",
                    image=rigid_contour,
                    kind="mask",
                    color=(249, 115, 22),
                    opacity=0.8,
                    render_mode="outline",
                )
            )
        if final_contour is not None:
            overlay_layers.append(
                OverlayLayer(
                    name=f"Final Registration::{region}",
                    image=final_contour,
                    kind="mask",
                    color=(34, 197, 94),
                    opacity=0.8,
                    render_mode="outline",
                )
            )

        self.contour_view.set_overlays(overlay_layers)

    def _update_dose_tab(self) -> None:
        if self.case is None:
            self.dose_view.clear_overlays()
            self.dose_metrics_table.clear_table()
            self.dvh_plot.set_data(None, None)
            return

        regions = self._dose_regions_for_case()
        self._ensure_dose_region_settings(regions)

        if not regions:
            self.dose_view.clear_overlays()
            self.dose_metrics_table.clear_table("No contours available.")
            self.dvh_plot.set_data(None, None)
            self._selected_dose_region = None
            return

        if self._selected_dose_region not in regions:
            self._selected_dose_region = regions[0]

        dose_type = str(self.dose_type_combo.currentData())
        fixed_fractions = int(self.fixed_fractions_spin.value())
        moving_fractions = int(self.moving_fractions_spin.value())

        blocks: list[dict[str, object]] = []
        for region in regions:
            row = self._dose_metrics_row(region, dose_type, fixed_fractions, moving_fractions)
            setting = self._dose_setting_for_region(region)
            metrics = [str(metric) for metric in setting.get("metrics", ["D50"])]
            metric_rows = [(metric, row.get(metric, "N/A")) for metric in metrics]
            blocks.append(
                {
                    "region": region,
                    "alpha_beta": float(setting.get("alpha_beta", 3.0)),
                    "metrics": metric_rows,
                }
            )

        self.dose_metrics_table.set_blocks(blocks)
        self.dose_metrics_table.select_region(self._selected_dose_region)
        self._update_dose_visuals(self._selected_dose_region, dose_type, fixed_fractions, moving_fractions)

    def _dose_regions_for_case(self) -> list[str]:
        if self.case is None:
            return []

        seen: set[str] = set()
        regions: list[str] = []
        for structure_map in (self.case.fixed_structures, self.case.rigid_structures, self.case.final_structures):
            for region in structure_map.keys():
                key = self._region_key(region)
                if key and key not in seen:
                    seen.add(key)
                    regions.append(region)
        regions.sort(key=str.lower)
        return regions

    def _region_key(self, region: str) -> str:
        return str(region).strip().lower()

    def _ensure_dose_region_settings(self, regions: list[str]) -> None:
        region_keys = {self._region_key(region) for region in regions}
        defaults = {
            self._region_key(entry.get("region", "")): dict(entry)
            for entry in default_dvh_settings_for_regions(regions)
        }

        for region in regions:
            key = self._region_key(region)
            if key not in self._dose_region_settings:
                self._dose_region_settings[key] = defaults.get(
                    key,
                    {"region": region, "alpha_beta": 3.0, "metrics": ["D50"]},
                )
            setting = self._dose_region_settings[key]
            setting["region"] = region
            setting["alpha_beta"] = float(setting.get("alpha_beta", 3.0))
            metrics = setting.get("metrics", ["D50"])
            setting["metrics"] = [str(metric) for metric in metrics] if isinstance(metrics, list) else [str(metrics)]

        for key in list(self._dose_region_settings.keys()):
            if key not in region_keys:
                self._dose_region_settings.pop(key, None)

    def _dose_setting_for_region(self, region: str) -> dict[str, object]:
        key = self._region_key(region)
        if key not in self._dose_region_settings:
            self._dose_region_settings[key] = default_dvh_settings_for_regions([region])[0]
        setting = self._dose_region_settings[key]
        setting.setdefault("region", region)
        setting.setdefault("alpha_beta", 3.0)
        setting.setdefault("metrics", ["D50"])
        return setting

    def _dose_state_key(
        self,
        region: str,
        dose_type: str,
        fixed_fractions: int,
        moving_fractions: int,
    ) -> tuple[object, ...]:
        setting = self._dose_setting_for_region(region)
        metrics = tuple(str(metric) for metric in setting.get("metrics", ["D50"]))
        alpha_beta = round(float(setting.get("alpha_beta", 3.0)), 6)
        stage = str(self.dose_stage_combo.currentData())
        return (
            self._region_key(region),
            stage,
            str(dose_type),
            int(fixed_fractions),
            int(moving_fractions),
            alpha_beta,
            metrics,
        )

    def _alpha_beta_for_region(self, region: str) -> float:
        setting = self._dose_setting_for_region(region)
        return float(setting.get("alpha_beta", 3.0))

    def _set_alpha_beta_for_region(self, region: str, alpha_beta: float) -> None:
        setting = self._dose_setting_for_region(region)
        setting["alpha_beta"] = float(alpha_beta)

    def _sync_dose_controls_to_region(self, region: str | None) -> None:
        return

    def _dose_selected_image(self, region: str, dose_type: str, fixed_fractions: int, moving_fractions: int) -> sitk.Image | None:
        if self.case is None:
            return None
        alpha_beta = self._alpha_beta_for_region(region)
        stage = str(self.dose_stage_combo.currentData())

        if stage == "fixed":
            source = self.case.fixed_dose
            fractions = fixed_fractions
        elif stage == "rigid":
            source = self.case.rigid_moving_dose
            fractions = moving_fractions
        elif stage == "final":
            source = self.case.final_moving_dose
            fractions = moving_fractions
        elif stage == "rigid_combined":
            return combine_dose_images_for_display(
                self.case.fixed_dose,
                self.case.rigid_moving_dose,
                dose_type=dose_type,
                fixed_alpha_beta=alpha_beta,
                moving_alpha_beta=alpha_beta,
                fixed_number_of_fractions=fixed_fractions,
                moving_number_of_fractions=moving_fractions,
            )
        elif stage == "final_combined":
            return combine_dose_images_for_display(
                self.case.fixed_dose,
                self.case.final_moving_dose,
                dose_type=dose_type,
                fixed_alpha_beta=alpha_beta,
                moving_alpha_beta=alpha_beta,
                fixed_number_of_fractions=fixed_fractions,
                moving_number_of_fractions=moving_fractions,
            )
        else:
            return None

        if source is None:
            return None
        return convert_dose_image_for_display(
            source,
            dose_type=dose_type,
            alpha_beta=alpha_beta,
            number_of_fractions=fractions,
        )

    def _format_dose_metric_display(self, metric_name: str, value: object, dose_type: str, unit: str | None = None) -> str:
        if value in {None, "N/A"}:
            return "N/A"
        if isinstance(value, str):
            return value
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        normalized = str(metric_name).strip().lower()
        if normalized.startswith("v"):
            return f"{numeric:.4g} %"
        display_unit = unit or ("Gy" if dose_type == "physical" else "Gy BED" if dose_type == "bed" else "Gy EQD2" if dose_type == "eqd2" else "Gy")
        return f"{numeric:.4g} {display_unit}"

    def _dose_metrics_row(self, region: str, dose_type: str, fixed_fractions: int, moving_fractions: int) -> dict[str, object]:
        setting = self._dose_setting_for_region(region)
        metrics = [str(metric) for metric in setting.get("metrics", ["D50"])]
        alpha_beta = round(float(setting.get("alpha_beta", 3.0)), 6)
        stage = str(self.dose_stage_combo.currentData())
        cache_key = (self._region_key(region), stage, str(dose_type), int(fixed_fractions), int(moving_fractions), alpha_beta)
        cached = self._dose_row_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        row: dict[str, object] = {
            "Region": region,
            "Alpha/Beta": alpha_beta,
            "Fixed Fractions": fixed_fractions,
            "Moving Fractions": moving_fractions,
            "Dose Type": dose_type.title(),
        }

        dose_image = self._dose_selected_image(region, dose_type, fixed_fractions, moving_fractions)
        mask = self._dose_mask_for_region(region)
        if dose_image is None:
            for metric in metrics:
                row[metric] = "N/A"
            self._dose_row_cache[cache_key] = dict(row)
            return row

        for metric in metrics:
            try:
                value, unit = metric_result_for_name(dose_image, metric, mask_image=mask)
                row[metric] = self._format_dose_metric_display(metric, value, dose_type, unit)
            except Exception:
                row[metric] = "N/A"
        self._dose_row_cache[cache_key] = dict(row)
        return row

    def _dose_mask_for_region(self, region: str) -> sitk.Image | None:
        if self.case is None:
            return None
        for structure_map in (self.case.fixed_structures, self.case.rigid_structures, self.case.final_structures):
            mask = structure_map.get(region)
            if mask is not None:
                return mask
        return None

    def _update_dose_visuals(self, region: str, dose_type: str, fixed_fractions: int, moving_fractions: int) -> None:
        if self.case is None:
            self.dose_view.clear_overlays()
            self.dvh_plot.set_data(None, None)
            return

        alpha_beta = round(float(self._dose_setting_for_region(region).get("alpha_beta", 3.0)), 6)
        stage = str(self.dose_stage_combo.currentData())
        cache_key = (self._region_key(region), stage, str(dose_type), int(fixed_fractions), int(moving_fractions), alpha_beta)
        if self._last_dose_visual_key == cache_key:
            return
        self._last_dose_visual_key = cache_key

        cached = self._dose_visual_cache.get(cache_key)
        if cached is None:
            dose_image = self._dose_selected_image(region, dose_type, fixed_fractions, moving_fractions)
            mask = self._dose_mask_for_region(region)
            if dose_image is None:
                self.dose_view.clear_overlays()
                self.dvh_plot.set_data(None, None)
                self._dose_visual_cache[cache_key] = {"dose_image": None, "dvh": None}
                return

            overlay_layers: list[OverlayLayer] = []
            if mask is not None:
                overlay_layers.append(
                    OverlayLayer(
                        name=f"{region} contour",
                        image=mask,
                        kind="mask",
                        color=(15, 23, 42),
                        opacity=0.9,
                        render_mode="outline",
                    )
                )
                dose_image = sitk.Mask(sitk.Cast(dose_image, sitk.sitkFloat32), sitk.Cast(mask > 0, sitk.sitkUInt8), outsideValue=0.0)

            overlay_layers.append(
                OverlayLayer(
                    name=f"{region} dose",
                    image=dose_image,
                    kind="dose",
                    color=(37, 99, 235),
                    opacity=0.35,
                    render_mode="fill",
                )
            )

            dvh_data = None
            if mask is not None:
                dvh_data = DVH.cumulative_dvh(dose_image, mask_image=mask, normalise=True)

            cached = {"dose_image": dose_image, "mask": mask, "overlays": overlay_layers, "dvh": dvh_data}
            self._dose_visual_cache[cache_key] = cached
        else:
            overlay_layers = list(cached.get("overlays", []))
            dose_image = cached.get("dose_image")
            mask = cached.get("mask")
            dvh_data = cached.get("dvh")

        self.dose_view.set_overlays(overlay_layers)
        self.dose_view.set_slice(self.dose_controls.slice_spin.value())
        self.dose_view.set_orientation(self.dose_controls.orientation_combo.currentText().lower())

        if not overlay_layers:
            self.dvh_plot.set_data(None, None)
            return

        if mask is None or dose_image is None or dvh_data is None:
            self.dvh_plot.set_data(None, None)
            return

        dose_axis, volume_axis = dvh_data
        self.dvh_plot.set_data(
            dose_axis,
            volume_axis,
            title=f"{region} - {dose_type.title()} DVH",
            x_label=f"{dose_type.title()} Dose",
            y_label="Volume (%)",
        )

    def _on_dose_region_selected(self, region: str) -> None:
        if self.case is None or region == self._selected_dose_region:
            return
        self._selected_dose_region = region
        self._update_dose_tab()

    def _on_edit_alpha_beta_requested(self, region: str) -> None:
        current = self._alpha_beta_for_region(region)
        value, accepted = QInputDialog.getDouble(
            self,
            "Edit Alpha/Beta",
            f"Alpha/Beta for {region}",
            current,
            0.1,
            100.0,
            2,
        )
        if not accepted:
            return
        self._set_alpha_beta_for_region(region, float(value))
        self._dose_row_cache.clear()
        self._dose_visual_cache.clear()
        self._last_dose_visual_key = None
        if self._selected_dose_region is None:
            self._selected_dose_region = region
        self._update_dose_tab()

    def _on_add_metric_requested(self, region: str) -> None:
        current_metrics = list(self._dose_setting_for_region(region).get("metrics", ["D50"]))
        suggested = "D95"
        value, accepted = QInputDialog.getText(
            self,
            "Add DVH Metric",
            f"Metric for {region}",
            text=suggested,
        )
        if not accepted:
            return
        metric = str(value).strip()
        if not metric:
            return
        if any(str(existing).strip().lower() == metric.lower() for existing in current_metrics):
            return
        current_metrics.append(metric)
        setting = self._dose_setting_for_region(region)
        setting["metrics"] = current_metrics
        self._dose_row_cache.clear()
        self._dose_visual_cache.clear()
        self._last_dose_visual_key = None
        if self._selected_dose_region is None:
            self._selected_dose_region = region
        self._update_dose_tab()

    def _on_contour_selection_changed(self) -> None:
        row_index = self.contour_metrics_table.selected_row_index()
        if row_index is None:
            return
        self._on_contour_cell_clicked(row_index, 0)

    def _on_contour_cell_clicked(self, row: int, _column: int) -> None:
        if self.case is None or self.case.contour_metrics is None or self.case.contour_metrics.empty:
            return
        if row < 0 or row >= len(self.case.contour_metrics):
            return

        region = str(self.case.contour_metrics.iloc[row]["Region"])
        if region != self._selected_contour_region:
            self._selected_contour_region = region
            self._render_contour_overlays()

    def _row_index_for_region(self, region: str | None) -> int | None:
        if self.case is None or self.case.contour_metrics is None or region is None:
            return None

        matches = self.case.contour_metrics.index[self.case.contour_metrics["Region"] == region].tolist()
        return matches[0] if matches else None

    def _first_contour_region(self) -> str | None:
        if self.case is None or self.case.contour_metrics is None or self.case.contour_metrics.empty:
            return None
        return str(self.case.contour_metrics.iloc[0]["Region"])

    def _stage_label(self, stage_key: str) -> str:
        return "Rigid Registration" if stage_key == "rigid" else "Final Registration"

    def _dose_stage_label(self, stage_key: str) -> str:
        return "Rigid Registered Dose" if stage_key == "rigid" else "Final Registered Dose"

    def _dose_stage_suffix(self, stage_key: str) -> str:
        return "Rigid Registration" if stage_key == "rigid" else "Final Registration"

    def _slice_count(self, volume: object, orientation: str) -> int:
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

    def _on_back(self) -> None:
        if self.main_window is not None:
            self.main_window.show_optimisation_page()
