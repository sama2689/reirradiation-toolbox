from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from registration.methods.executor import RegistrationExecutor, RegistrationExecutionResult


class RegistrationCancelled(Exception):
    pass


class RegistrationCancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise RegistrationCancelled()


class RegistrationWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        fixed_image,
        moving_image,
        pipeline: list[dict[str, Any]],
        fixed_structures: dict[str, Any] | None,
        moving_structures: dict[str, Any] | None,
        cancel_token: RegistrationCancelToken,
        parent=None,
    ):
        super().__init__(parent)
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.pipeline = pipeline
        self.fixed_structures = fixed_structures
        self.moving_structures = moving_structures
        self.cancel_token = cancel_token

    def run(self) -> None:
        try:
            executor = RegistrationExecutor(progress_callback=self.progress.emit)
            result: RegistrationExecutionResult = executor.execute(
                fixed_image=self.fixed_image,
                moving_image=self.moving_image,
                pipeline=self.pipeline,
                fixed_structures=self.fixed_structures,
                moving_structures=self.moving_structures,
                cancel_check=self.cancel_token.check,
            )
            self.finished.emit(result)
        except RegistrationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))