from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from ui.widgets.chrome import ArrowComboBox as QComboBox
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget


class ViewerControlsWidget(QWidget):
    sliceChanged = Signal(int)
    windowChanged = Signal(int)
    levelChanged = Signal(int)
    orientationChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        label_width = 92
        spin_width = 76

        self.orientation_label = QLabel("View")
        self.orientation_label.setFixedWidth(label_width)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.orientation_combo.currentTextChanged.connect(self._on_orientation_changed)
        self.orientation_combo.setFixedWidth(120)

        grid.addWidget(self.orientation_label, 0, 0)
        grid.addWidget(self.orientation_combo, 0, 1, 1, 2)

        self.slice_label = QLabel("Slice")
        self.slice_label.setFixedWidth(label_width)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self.slice_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 100)
        self.slice_spin.setValue(0)
        self.slice_spin.valueChanged.connect(self._on_slice_spin_changed)
        self.slice_spin.setFixedWidth(spin_width)

        grid.addWidget(self.slice_label, 1, 0)
        grid.addWidget(self.slice_slider, 1, 1)
        grid.addWidget(self.slice_spin, 1, 2)

        self.window_label = QLabel("Window")
        self.window_label.setFixedWidth(label_width)

        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setRange(1, 4000)
        self.window_slider.setValue(800)
        self.window_slider.valueChanged.connect(self._on_window_slider_changed)
        self.window_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 4000)
        self.window_spin.setValue(800)
        self.window_spin.valueChanged.connect(self._on_window_spin_changed)
        self.window_spin.setFixedWidth(spin_width)

        grid.addWidget(self.window_label, 2, 0)
        grid.addWidget(self.window_slider, 2, 1)
        grid.addWidget(self.window_spin, 2, 2)

        self.level_label = QLabel("Level")
        self.level_label.setFixedWidth(label_width)

        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setRange(-1000, 3000)
        self.level_slider.setValue(0)
        self.level_slider.valueChanged.connect(self._on_level_slider_changed)
        self.level_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 3000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self._on_level_spin_changed)
        self.level_spin.setFixedWidth(spin_width)

        grid.addWidget(self.level_label, 3, 0)
        grid.addWidget(self.level_slider, 3, 1)
        grid.addWidget(self.level_spin, 3, 2)

        layout.addLayout(grid)

    def set_slice_range(self, min_slice: int, max_slice: int, value: int | None = None, emit: bool = True) -> None:
        self._updating = True
        self.slice_slider.setRange(min_slice, max_slice)
        self.slice_spin.setRange(min_slice, max_slice)

        if value is None:
            value = (min_slice + max_slice) // 2
        value = max(min_slice, min(max_slice, int(value)))

        self.slice_slider.setValue(value)
        self.slice_spin.setValue(value)

        self._updating = False
        if emit:
            self.sliceChanged.emit(value)

    def set_slice_value(self, value: int, emit: bool = True) -> None:
        self._updating = True
        self.slice_slider.setValue(int(value))
        self.slice_spin.setValue(int(value))
        self._updating = False
        if emit:
            self.sliceChanged.emit(int(value))

    def set_window_value(self, value: int, emit: bool = True) -> None:
        self._updating = True
        self.window_slider.setValue(int(value))
        self.window_spin.setValue(int(value))
        self._updating = False
        if emit:
            self.windowChanged.emit(int(value))

    def set_level_value(self, value: int, emit: bool = True) -> None:
        self._updating = True
        self.level_slider.setValue(int(value))
        self.level_spin.setValue(int(value))
        self._updating = False
        if emit:
            self.levelChanged.emit(int(value))

    def set_orientation(self, orientation: str, emit: bool = True) -> None:
        normalized = orientation.strip().lower()
        if normalized not in {"axial", "coronal", "sagittal"}:
            normalized = "axial"

        self._updating = True
        self.orientation_combo.setCurrentText(normalized.title())
        self._updating = False
        if emit:
            self.orientationChanged.emit(normalized)

    def _on_orientation_changed(self, value: str) -> None:
        if self._updating:
            return
        self.orientationChanged.emit(value.lower())

    def _on_slice_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.slice_spin.setValue(value)
        self._updating = False
        self.sliceChanged.emit(value)

    def _on_slice_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.slice_slider.setValue(value)
        self._updating = False
        self.sliceChanged.emit(value)

    def _on_window_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.window_spin.setValue(value)
        self._updating = False
        self.windowChanged.emit(value)

    def _on_window_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.window_slider.setValue(value)
        self._updating = False
        self.windowChanged.emit(value)

    def _on_level_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.level_spin.setValue(value)
        self._updating = False
        self.levelChanged.emit(value)

    def _on_level_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.level_slider.setValue(value)
        self._updating = False
        self.levelChanged.emit(value)
