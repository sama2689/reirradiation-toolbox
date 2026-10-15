from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ui.dialogs.progress import BusyDialog
from ui.dialogs.message import AppMessageDialog
from ui.pages.home import HomePage
from ui.pages.registration import RegistrationPage
from ui.pages.selection import SelectionPage
from ui.pages.optimisation import OptimisationPage
from ui.pages.evaluation import EvaluationPage
from ui.workers.evaluation import EvaluationWorker

from registration.dataset.dicom_reader import DICOMReader
from registration.dataset.discovery import DatasetDiscovery
from ui.workers.registration import RegistrationCancelToken, RegistrationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        icon_path = Path(__file__).parent / "app_logo.png"
        self.setWindowIcon(QIcon(str(icon_path)))

        self.setWindowTitle("Image Registration")
        self.resize(1000, 600)

        # Shared state across pages
        self.current_directory: str | None = None
        self.image_series_dataset: list[dict[str, Any]] = []
        self.fixed_series: dict[str, Any] | None = None
        self.moving_series: dict[str, Any] | None = None
        self.fixed_dataset = None
        self.moving_dataset = None

        self._signals_connected = False

        self.worker_thread: QThread | None = None
        self.worker: RegistrationWorker | None = None
        self.cancel_token: RegistrationCancelToken | None = None
        self.latest_result = None
        self.evaluation_worker_thread: QThread | None = None
        self.evaluation_worker: EvaluationWorker | None = None
        self.evaluation_progress_dialog: BusyDialog | None = None

        self.setup_ui()

    def setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(20, 20, 20, 20)

        self.stack = QStackedWidget()
        outer_layout.addWidget(self.stack)

        # Pages
        self.home_page = HomePage(main_window=self)
        self.selection_page = SelectionPage(main_window=self)
        self.registration_page = RegistrationPage(main_window=self)
        self.optimisation_page = OptimisationPage(main_window=self)
        self.evaluation_page = EvaluationPage(main_window=self)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.selection_page)
        self.stack.addWidget(self.registration_page)
        self.stack.addWidget(self.optimisation_page)
        self.stack.addWidget(self.evaluation_page)

        self.registration_page.run_requested.connect(self.start_registration)

        self.stack.setCurrentWidget(self.home_page)
    
    def return_to_registration(self) -> None:
        """
        Return to the registration configuration page without
        reloading any datasets.
        """

        if self.cancel_token is not None:
            self.cancel_token.cancel()

        self.stack.setCurrentWidget(self.registration_page)

    # -------------------------------------------------
    # Navigation
    # -------------------------------------------------

    def show_home_page(self) -> None:
        self.stack.setCurrentWidget(self.home_page)

    def show_selection_page(self) -> None:
        self.selection_page.populate_series_dropdowns(self.image_series_dataset)
        self.stack.setCurrentWidget(self.selection_page)

    def show_registration_page(self) -> None:
        progress = BusyDialog("Importing image series...", self)
        progress.show()
        QApplication.processEvents()

        def update_progress(current: int, message: str | None = None) -> None:
            progress.set_progress(current, 100, message)
            QApplication.processEvents()

        try:
            if self.fixed_series is None or self.moving_series is None:
                raise ValueError("Fixed and moving series must be selected before registration.")

            reader = DICOMReader()

            self.fixed_dataset = reader.read(
                directory=self.fixed_series["path"],
                progress_callback=update_progress,
            )
            self.moving_dataset = reader.read(
                directory=self.moving_series["path"],
                progress_callback=update_progress,
            )

            fixed_image = self.fixed_dataset.ct_image
            moving_image = self.moving_dataset.ct_image

            fixed_names = sorted(self.fixed_dataset.structures.keys())
            moving_names = sorted(self.moving_dataset.structures.keys())
            preview_context = {
                "fixed_image": fixed_image,
                "moving_image": moving_image,
                "fixed_structures": getattr(self.fixed_dataset, "structures", None),
                "moving_structures": getattr(self.moving_dataset, "structures", None),
            }

        except Exception:
            progress.close()
            raise
        else:
            progress.close()
            QApplication.processEvents()

        self.stack.setCurrentWidget(self.registration_page)
        QApplication.processEvents()

        QTimer.singleShot(
            0,
            lambda: self._populate_registration_page(
                fixed_names=fixed_names,
                moving_names=moving_names,
                fixed_image=fixed_image,
                moving_image=moving_image,
                preview_context=preview_context,
            ),
        )

    def show_optimisation_page(self) -> None:
        self.stack.setCurrentWidget(self.optimisation_page)

    def _populate_registration_page(
        self,
        *,
        fixed_names: list[str],
        moving_names: list[str],
        fixed_image,
        moving_image,
        preview_context: dict[str, Any],
    ) -> None:
        try:
            self.registration_page.set_contour_names(moving_names, fixed_names)
            self.registration_page.set_registration_images(fixed_image, moving_image)
            self.registration_page.set_preview_context(
                fixed_image=preview_context.get("fixed_image"),
                moving_image=preview_context.get("moving_image"),
                fixed_structures=preview_context.get("fixed_structures"),
                moving_structures=preview_context.get("moving_structures"),
            )
        except Exception as exc:
            self.show_warning("Registration unavailable", f"The registration page could not be prepared: {exc}")

    def show_evaluation_page(self) -> None:
        self.stack.setCurrentWidget(self.evaluation_page)

    def cancel_registration(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        self.optimisation_page.set_status("Cancelling registration.")
        self.optimisation_page.set_detail("")
        self.return_to_registration()

    def start_registration(self, payload: dict) -> None:
        pipeline = payload.get("pipeline", [])
        if not pipeline:
            return

        if self.fixed_dataset is None or self.moving_dataset is None:
            self.show_warning("Missing data", "No fixed/moving dataset has been loaded.")
            return

        fixed_image = getattr(self.fixed_dataset, "ct_image", None)
        moving_image = getattr(self.moving_dataset, "ct_image", None)

        if fixed_image is None or moving_image is None:
            self.show_warning("Missing image", "CT images are not available for registration.")
            return

        step_names = [
            str(step.get("step", step.get("name", "Step")))
            for step in pipeline
        ]
        self.optimisation_page.reset_for_pipeline(step_names)
        self.optimisation_page.set_status("Starting registration...")
        self.optimisation_page.set_detail("")
        self.optimisation_page.set_overall_progress(0)
        self.optimisation_page.set_evaluate_enabled(False)

        self.latest_result = None
        self.cancel_token = RegistrationCancelToken()

        self.worker_thread = QThread(self)
        self.worker = RegistrationWorker(
            fixed_image=fixed_image,
            moving_image=moving_image,
            pipeline=pipeline,
            fixed_structures=getattr(self.fixed_dataset, "structures", None),
            moving_structures=getattr(self.moving_dataset, "structures", None),
            cancel_token=self.cancel_token,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker.progress.connect(self.optimisation_page.update_from_progress)
        self.worker.finished.connect(self._on_registration_finished)
        self.worker.failed.connect(self._on_registration_failed)
        self.worker.cancelled.connect(self._on_registration_cancelled)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.cancelled.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.show_optimisation_page()
        self.worker_thread.start()

    def _on_registration_finished(self, result) -> None:
        self.latest_result = result
        self.optimisation_page.set_status("Registration complete.")
        self.optimisation_page.set_detail("Done.")
        self.optimisation_page.set_overall_progress(100)
        self.optimisation_page.set_evaluate_enabled(True)

        self.worker = None
        self.worker_thread = None
        self.cancel_token = None

    def _on_registration_failed(self, message: str) -> None:
        self.optimisation_page.set_status("Registration failed.")
        self.optimisation_page.set_detail(message)
        self.optimisation_page.set_evaluate_enabled(False)

        self.show_warning("Registration failed", message)

        self.worker = None
        self.worker_thread = None
        self.cancel_token = None

    def _on_registration_cancelled(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.cancel_token = None


    def open_evaluation_page(self) -> None:
        if self.latest_result is None:
            return

        if self.fixed_dataset is None or self.moving_dataset is None:
            self.show_warning("Evaluation unavailable", "Registration datasets are not available.")
            return

        fixed_image = getattr(self.fixed_dataset, "ct_image", None)
        moving_image = getattr(self.moving_dataset, "ct_image", None)
        if fixed_image is None or moving_image is None:
            self.show_warning("Evaluation unavailable", "CT images are not available for evaluation.")
            return

        if self.evaluation_worker_thread is not None:
            return

        rigid_transform = getattr(self.latest_result, "rigid_transform", None)
        final_transform = getattr(self.latest_result, "final_transform", None) or rigid_transform

        fixed_structures = dict(getattr(self.fixed_dataset, "structures", None) or {})
        moving_structures = dict(getattr(self.moving_dataset, "structures", None) or {})

        fixed_dose = self._physical_dose_image(self.fixed_dataset)
        moving_dose = self._physical_dose_image(self.moving_dataset)
        fixed_dose_scaling = getattr(self.fixed_dataset, "dose_scaling", None)
        moving_dose_scaling = getattr(self.moving_dataset, "dose_scaling", None)

        contour_count = len(set(fixed_structures.keys()) & set(moving_structures.keys()))
        evaluation_total = 9 + contour_count

        self.evaluation_progress_dialog = BusyDialog("Applying transforms and calculating metrics...", self)
        self.evaluation_progress_dialog.set_progress(0, evaluation_total, "Preparing evaluation...")
        self.evaluation_progress_dialog.show()

        self.evaluation_worker_thread = QThread(self)
        self.evaluation_worker = EvaluationWorker(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_structures=fixed_structures,
            moving_structures=moving_structures,
            fixed_dose=fixed_dose,
            moving_dose=moving_dose,
            rigid_transform=rigid_transform,
            final_transform=final_transform,
            fixed_dose_scaling=fixed_dose_scaling,
            moving_dose_scaling=moving_dose_scaling,
            pipeline_steps=list(getattr(self.latest_result, "steps", []) or []),
        )
        self.evaluation_worker.moveToThread(self.evaluation_worker_thread)

        self.evaluation_worker.progress.connect(self._on_evaluation_progress)
        self.evaluation_worker.finished.connect(self._on_evaluation_finished)
        self.evaluation_worker.failed.connect(self._on_evaluation_failed)

        self.evaluation_worker_thread.started.connect(self.evaluation_worker.run)
        self.evaluation_worker.finished.connect(self.evaluation_worker_thread.quit)
        self.evaluation_worker.failed.connect(self.evaluation_worker_thread.quit)
        self.evaluation_worker_thread.finished.connect(self.evaluation_worker.deleteLater)
        self.evaluation_worker_thread.finished.connect(self.evaluation_worker_thread.deleteLater)

        self.evaluation_worker_thread.start()

    def _on_evaluation_progress(self, payload: dict) -> None:
        dialog = self.evaluation_progress_dialog
        if dialog is None:
            return

        current = int(payload.get("current", 0))
        total = int(payload.get("total", 100))
        message = payload.get("message")
        dialog.set_progress(current, total, str(message) if message is not None else None)

    def _on_evaluation_finished(self, evaluation_case) -> None:
        self.evaluation_page.set_evaluation_case(evaluation_case)
        self._cleanup_evaluation_worker()
        self.show_evaluation_page()

    def _on_evaluation_failed(self, message: str) -> None:
        self._cleanup_evaluation_worker()
        self.show_warning("Evaluation unavailable", message)

    def _cleanup_evaluation_worker(self) -> None:
        if self.evaluation_progress_dialog is not None:
            self.evaluation_progress_dialog.close()
            self.evaluation_progress_dialog.deleteLater()
            self.evaluation_progress_dialog = None

        self.evaluation_worker = None
        self.evaluation_worker_thread = None

    def _physical_dose_image(self, dataset):
        dose_image = getattr(dataset, "dose_image", None)
        if dose_image is None:
            return None

        import SimpleITK as sitk

        # Keep the raw RTDOSE values here; the evaluation worker applies the
        # scaling exactly once before masking / resampling so we do not double-scale.
        return sitk.Cast(dose_image, sitk.sitkFloat32)

    # -------------------------------------------------
    # Home page actions
    # -------------------------------------------------

    def on_select_directory_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        valid_files, valid_transforms = self.check_directory(directory)

        if directory and valid_files:
            self.current_directory = directory
            self.home_page.update_directory_label(directory)
            self.home_page.set_action_states(
                register_enabled=True,
                evaluate_enabled=valid_transforms,
                apply_enabled=valid_transforms,
            )
            return

        self.current_directory = None
        self.home_page.update_directory_label(None)
        self.home_page.set_action_states(
            register_enabled=False,
            evaluate_enabled=False,
            apply_enabled=False,
        )

        if not directory:
            self.show_warning(
                title="Invalid Directory",
                message="No directory selected.",
                info_text="Please select a directory containing at least two valid image series for registration.",
            )
        else:
            self.show_warning(
                title="Invalid Directory",
                message="Unable to find valid CT series.",
                info_text="Please select a directory containing at least two valid image series for registration.",
            )

    def check_directory(self, directory: str | None) -> tuple[bool, bool]:
        if not directory:
            return False, False

        progress = BusyDialog("Searching for valid image series...", self)
        progress.show()
        QApplication.processEvents()

        def update_progress(current: int, total: int) -> None:
            progress.set_progress(current, total)
            QApplication.processEvents()

        try:
            result = DatasetDiscovery.count_datasets(directory, progress_callback=update_progress)
        finally:
            progress.close()

        valid_files = False
        if result is not None:
            self.image_series_dataset = result["datasets"]
            valid_files = result["total_count"] >= 2

        transforms_directory = os.path.join(directory, "transforms")
        valid_transforms = os.path.isdir(transforms_directory) and bool(
            DatasetDiscovery.find_hdf5_files(transforms_directory)
        )

        return valid_files, valid_transforms

    def show_warning(self, title: str, message: str, info_text: str = "") -> None:
        dlg = AppMessageDialog(
            title=title,
            message=message,
            info_text=info_text,
            parent=self,
            icon=QStyle.SP_MessageBoxWarning,
        )
        dlg.exec()

    def set_selected_series(self, fixed: dict[str, Any], moving: dict[str, Any]) -> None:
        self.fixed_series = fixed
        self.moving_series = moving
        self.show_registration_page()
