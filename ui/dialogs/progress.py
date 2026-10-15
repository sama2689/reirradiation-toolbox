from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class BusyDialog(QDialog):
    def __init__(self, message: str, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Progress")
        self.setModal(True)
        self.setMinimumWidth(450)

        self.label = QLabel(message)
        self.label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                min-height: 10px;
                border: none;
                border-radius: 6px;
                background: #d1d5db;
                text-align: center;
            }

            QProgressBar::chunk {
                border-radius: 6px;
                background: #005EB8;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

    def set_progress(self, current, total, message=None):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        if message:
            self.label.setText(message)