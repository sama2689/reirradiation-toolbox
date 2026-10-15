from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget


class OptimizationPlotWidget(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._values: list[float] = []
        self._summary_text = ""

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)

    def clear(self) -> None:
        self._values.clear()
        self._summary_text = ""
        self.update()

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def append_metric(self, value: float) -> None:
        self._values.append(float(value))
        self.update()

    def set_summary_text(self, text: str) -> None:
        self._summary_text = text.strip()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            painter.fillRect(self.rect(), QColor("#f8fafc"))

            outer = QRectF(self.rect()).adjusted(12, 12, -12, -12)
            painter.setPen(QPen(QColor("#d9dee7")))
            painter.setBrush(QColor("white"))
            painter.drawRoundedRect(outer, 14, 14)

            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)

            painter.setFont(title_font)
            painter.setPen(QColor("#1e293b"))
            painter.drawText(
                outer.adjusted(16, 14, -16, -16),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                self._title,
            )

            if self._summary_text:
                painter.setFont(QFont())
                painter.setPen(QColor("#475569"))
                painter.drawText(
                    outer.adjusted(16, 36, -16, -16),
                    Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                    self._summary_text,
                )
                plot_rect = outer.adjusted(18, 62, -18, -18)
            else:
                plot_rect = outer.adjusted(18, 46, -18, -18)

            painter.setPen(QPen(QColor("#e2e8f0")))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(plot_rect, 10, 10)

            values = [float(v) for v in self._values if isinstance(v, (int, float))]
            if len(values) < 2:
                painter.setPen(QColor("#94a3b8"))
                painter.drawText(
                    plot_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    "No optimisation history yet.",
                )
                return

            min_v = min(values)
            max_v = max(values)
            value_range = max(max_v - min_v, 1e-12)

            inner = plot_rect.adjusted(10, 10, -10, -10)
            width = max(1.0, float(inner.width()))
            height = max(1.0, float(inner.height()))

            grid_pen = QPen(QColor("#edf2f7"))
            painter.setPen(grid_pen)
            for i in range(1, 4):
                y = inner.top() + (height * i / 4.0)
                painter.drawLine(inner.left(), y, inner.right(), y)

            line_pen = QPen(QColor("#005EB8"))
            line_pen.setWidth(2)
            painter.setPen(line_pen)

            points: list[QPointF] = []
            count = len(values)
            for i, v in enumerate(values):
                x = inner.left() + (width * i / max(1, count - 1))
                y = inner.bottom() - (((v - min_v) / value_range) * height)
                points.append(QPointF(x, y))

            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)
        finally:
            painter.end()



class StepStatusWidget(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("stepCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")

        self.state_label = QLabel("Pending")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        top.addWidget(self.title_label)
        top.addStretch(1)
        top.addWidget(self.state_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("stepProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        layout.addLayout(top)
        layout.addWidget(self.progress_bar)

        self.set_state("pending", "Pending", 0)

    def set_state(self, state: str, message: str | None = None, progress: int = 0, duration_text: str | None = None) -> None:
        self.progress_bar.setProperty("state", state)
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)

        self.progress_bar.setValue(max(0, min(100, int(progress))))

        label = message or state.title()
        if duration_text:
            label = f"{label} ({duration_text})"
        self.state_label.setText(label)
