from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class DvhPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout.addWidget(self.canvas)
        self._draw_placeholder("No DVH available yet.")

    def _draw_placeholder(self, message: str) -> None:
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
        self.canvas.draw_idle()

    def set_data(
        self,
        dose_axis: Iterable[float] | np.ndarray | None,
        volume_axis: Iterable[float] | np.ndarray | None,
        title: str = "Cumulative DVH",
        x_label: str = "Dose (Gy)",
        y_label: str = "Volume (%)",
    ) -> None:
        if dose_axis is None or volume_axis is None:
            self._draw_placeholder("No DVH available for the selected contour.")
            return

        dose = np.asarray(list(dose_axis), dtype=np.float64)
        volume = np.asarray(list(volume_axis), dtype=np.float64)
        if dose.size == 0 or volume.size == 0:
            self._draw_placeholder("No DVH available for the selected contour.")
            return

        self.ax.clear()
        self.ax.plot(dose, volume)
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlim(left=0.0)
        self.ax.set_ylim(bottom=0.0)
        self.canvas.draw_idle()
