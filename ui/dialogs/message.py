from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class AppMessageDialog(QDialog):
    def __init__(
        self,
        title: str,
        message: str,
        info_text: str = "",
        icon: QStyle.StandardPixmap | None = QStyle.StandardPixmap.SP_MessageBoxWarning,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(500)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-weight: 600;
            }
            """
        )

        info_label = None
        if info_text:
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setStyleSheet(
                """
                QLabel {
                    font-size: 13px;
                    font-weight: 400;
                    color: #444;
                }
                """
            )

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(8)

        text_layout.addWidget(message_label)
        if info_label is not None:
            text_layout.addWidget(info_label)

        content_layout = QGridLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(16)
        content_layout.setVerticalSpacing(0)

        if icon is not None:
            icon_label = QLabel()
            pixmap = self.style().standardIcon(icon).pixmap(64, 64)
            icon_label.setPixmap(pixmap)
            icon_label.setFixedSize(64, 64)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_layout.addWidget(
                icon_label,
                0,
                0,
                2,
                1,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )

        content_layout.addLayout(text_layout, 0, 1)

        ok_button = QPushButton("Confirm")
        ok_button.setDefault(False)
        ok_button.setAutoDefault(False)
        ok_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ok_button.setStyleSheet(
            """
            QPushButton {
                min-width: 90px;
                padding: 8px 18px;
                border: none;
                border-radius: 10px;
                background: #d1d5db;
                color: #111827;
            }
            QPushButton:hover {
                background: #bbc1ca;
            }
            """
        )
        ok_button.clicked.connect(self.accept)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(ok_button)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)
        main_layout.addLayout(content_layout)
        main_layout.addLayout(button_layout)