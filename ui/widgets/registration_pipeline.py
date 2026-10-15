from __future__ import annotations

import copy
from typing import Any, Dict, List, TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from registration.methods.pipeline import DEFAULT_PIPELINE, RegistrationStep

from ui.dialogs.registration_parameters import ParamDialog

if TYPE_CHECKING:
    from PySide6.QtWidgets import QDialog


class StepRow(QFrame):
    remove_clicked = Signal()
    edit_clicked = Signal()
    move_up_clicked = Signal()
    move_down_clicked = Signal()

    def __init__(self, step: RegistrationStep, parent: QWidget | None = None):
        super().__init__(parent)
        self.step = step
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self.title = QLabel(step.name)
        self.title.setObjectName("stepTitle")
        self.title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.title, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setObjectName("utilityButton")
        self.edit_btn.setFixedSize(60, 32)
        self.edit_btn.clicked.connect(lambda: self.edit_clicked.emit())
        controls.addWidget(self.edit_btn)

        self.up_btn = QToolButton()
        self.up_btn.setObjectName("iconButton")
        self.up_btn.setToolTip("Move up")
        self.up_btn.setText("↑")
        self.up_btn.setFixedSize(32, 32)
        self.up_btn.clicked.connect(lambda: self.move_up_clicked.emit())
        controls.addWidget(self.up_btn)

        self.down_btn = QToolButton()
        self.down_btn.setObjectName("iconButton")
        self.down_btn.setToolTip("Move down")
        self.down_btn.setText("↓")
        self.down_btn.setFixedSize(32, 32)
        self.down_btn.clicked.connect(lambda: self.move_down_clicked.emit())
        controls.addWidget(self.down_btn)

        self.remove_btn = QToolButton()
        self.remove_btn.setObjectName("iconButton")
        self.remove_btn.setToolTip("Remove step")
        self.remove_btn.setText("−")
        self.remove_btn.setFixedSize(32, 32)
        self.remove_btn.clicked.connect(lambda: self.remove_clicked.emit())
        controls.addWidget(self.remove_btn)

        layout.addLayout(controls)


class RegistrationWidget(QFrame):
    
    run_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.steps: List[RegistrationStep] = []
        self.moving_contour_names: list[str] = []
        self.fixed_contour_names: list[str] = []
        self.preview_context: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("Registration")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)
        layout.addSpacing(10)

        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(10)
        self.steps_layout.addStretch(1)
        layout.addWidget(self.steps_container, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        self.reset_btn = QPushButton("Reset defaults")
        self.reset_btn.setObjectName("utilityButton")
        self.reset_btn.setFixedHeight(50)
        self.reset_btn.clicked.connect(self.reset_defaults)
        bottom_row.addWidget(self.reset_btn, 1)

        self.add_btn = QToolButton()
        self.add_btn.setObjectName("iconButton")
        self.add_btn.setToolTip("Add step")
        self.add_btn.setText("+")
        self.add_btn.setFixedSize(50, 50)
        self.add_btn.clicked.connect(self.add_step_menu)
        bottom_row.addWidget(self.add_btn)

        layout.addLayout(bottom_row)

        self.run_btn = QPushButton("Run registration")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setFixedHeight(50)
        self.run_btn.clicked.connect(self.run_registration)
        layout.addWidget(self.run_btn)

        self.reset_defaults()

    def add_step_menu(self) -> None:
        menu = QMenu(self)

        for name in ["Masking", "Rigid Registration", "Demons Registration"]:
            action = menu.addAction(name)
            action.triggered.connect(lambda checked=False, step_name=name: self._add_step(step_name))

        menu.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft()))

    def _add_step(self, step_name: str) -> None:
        self.steps.append(RegistrationStep.from_template(step_name))
        self.rebuild()

    def remove_step(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
            self.rebuild()

    def move_step_up(self, index: int) -> None:
        if index > 0:
            self.steps[index - 1], self.steps[index] = self.steps[index], self.steps[index - 1]
            self.rebuild()

    def move_step_down(self, index: int) -> None:
        if 0 <= index < len(self.steps) - 1:
            self.steps[index + 1], self.steps[index] = self.steps[index], self.steps[index + 1]
            self.rebuild()

    def edit_step(self, index: int) -> None:
        if not (0 <= index < len(self.steps)):
            return

        step = self.steps[index]
        dlg = ParamDialog(step.name, step.params, self, preview_context=self.preview_context)

        if step.name == "Masking" and hasattr(dlg.editor, "set_contour_names"):
            dlg.editor.set_contour_names(self.moving_contour_names, self.fixed_contour_names)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            step.params = dlg.get_params()
            self.rebuild()

    def reset_defaults(self) -> None:
        self.steps = [RegistrationStep.from_template(name) for name in DEFAULT_PIPELINE]
        self.rebuild()

    def run_registration(self) -> None:
        config = [
            {"step": step.name, "params": copy.deepcopy(step.params)}
            for step in self.steps
        ]
        self.run_requested.emit({"pipeline": config})

    def rebuild(self) -> None:
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for idx, step in enumerate(self.steps):
            row = StepRow(step)
            row.remove_clicked.connect(lambda i=idx: self.remove_step(i))
            row.edit_clicked.connect(lambda i=idx: self.edit_step(i))
            row.move_up_clicked.connect(lambda i=idx: self.move_step_up(i))
            row.move_down_clicked.connect(lambda i=idx: self.move_step_down(i))
            self.steps_layout.addWidget(row)

        self.steps_layout.addStretch(1)

    def set_contour_names(self, moving_names: list[str], fixed_names: list[str]) -> None:
        self.moving_contour_names = list(moving_names)
        self.fixed_contour_names = list(fixed_names)

    def set_preview_context(
        self,
        *,
        fixed_image: Any | None,
        moving_image: Any | None,
        fixed_structures: dict[str, Any] | None,
        moving_structures: dict[str, Any] | None,
    ) -> None:
        self.preview_context = {
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "fixed_structures": dict(fixed_structures or {}),
            "moving_structures": dict(moving_structures or {}),
        }
