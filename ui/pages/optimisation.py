from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.optimisation import OptimizationPlotWidget, StepStatusWidget

if TYPE_CHECKING:
    from ui.main_window import MainWindow


class OptimisationPage(QWidget):
    def __init__(self, main_window: "MainWindow", parent: QWidget | None = None):
        super().__init__(parent)
        self.main_window = main_window

        self.step_widgets: dict[str, StepStatusWidget] = {}
        self.plot_widgets: dict[str, OptimizationPlotWidget] = {}
        self._step_started_at: dict[str, float] = {}

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

        self.back_button = QPushButton("Back")
        self.back_button.setMinimumHeight(40)
        self.back_button.clicked.connect(self._on_back)

        title = QLabel("Registration Optimisation")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.evaluate_button = QPushButton("Evaluate Registration")
        self.evaluate_button.setMinimumHeight(40)
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.clicked.connect(self._on_evaluate)

        top_layout.addWidget(self.back_button, 0, 0)
        top_layout.addWidget(title, 0, 1)
        top_layout.addWidget(self.evaluate_button, 0, 2)
        top_layout.setColumnStretch(0, 0)
        top_layout.setColumnStretch(1, 1)
        top_layout.setColumnStretch(2, 0)

        body = QHBoxLayout()
        body.setSpacing(16)

        plot_card = QFrame()
        plot_card.setObjectName("plotCard")
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(16, 16, 16, 16)
        plot_layout.setSpacing(10)

        plot_title = QLabel("Optimisation Plot")
        plot_title.setObjectName("sectionTitle")

        self.plot_stack = QStackedWidget()
        self.rigid_plot = OptimizationPlotWidget("Rigid Registration")
        self.demons_plot = OptimizationPlotWidget("Demons Registration")
        self.plot_widgets["Rigid Registration"] = self.rigid_plot
        self.plot_widgets["Demons Registration"] = self.demons_plot
        self.plot_stack.addWidget(self.rigid_plot)
        self.plot_stack.addWidget(self.demons_plot)

        plot_layout.addWidget(plot_title)
        plot_layout.addWidget(self.plot_stack, 1)

        step_card = QFrame()
        step_card.setObjectName("stepCard")
        step_layout = QVBoxLayout(step_card)
        step_layout.setContentsMargins(16, 16, 16, 16)
        step_layout.setSpacing(10)

        step_title = QLabel("Pipeline Steps")
        step_title.setObjectName("sectionTitle")
        step_layout.addWidget(step_title)

        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(10)
        self.steps_layout.addStretch(1)

        step_layout.addWidget(self.steps_container, 1)

        self.status_label = QLabel("Waiting to start...")
        self.status_label.setWordWrap(True)
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)

        self.overall_progress = QProgressBar()
        self.overall_progress.setObjectName("overallProgress")
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("")
        self.overall_progress.setTextVisible(False)

        # The optimisation page no longer shows bottom-right status text or
        # progress controls; the pipeline step cards and plot summaries provide
        # the necessary feedback instead.
        self.status_label.hide()
        self.detail_label.hide()
        self.overall_progress.hide()

        body.addWidget(plot_card, 3)
        body.addWidget(step_card, 2)

        page_layout.addWidget(top_card)
        page_layout.addLayout(body)

        self.reset_for_pipeline([])

    def reset_for_pipeline(self, step_names: list[str]) -> None:
        self._clear_layout(self.steps_layout)
        self.step_widgets.clear()
        self._step_started_at.clear()

        canonical_steps: list[str] = [self._canonical_step_name(name) for name in step_names]
        if not canonical_steps:
            canonical_steps = ["Masking", "Rigid Registration", "Demons Registration"]

        for step in canonical_steps:
            widget = StepStatusWidget(step)
            self.step_widgets[step] = widget
            self.steps_layout.addWidget(widget)

        self.steps_layout.addStretch(1)

        self.rigid_plot.clear()
        self.demons_plot.clear()
        self._show_plot("Rigid Registration")

        self.status_label.setText("Ready.")
        self.detail_label.setText("")
        self.overall_progress.setValue(0)
        self.evaluate_button.setEnabled(False)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_detail(self, text: str) -> None:
        self.detail_label.setText(text)

    def set_overall_progress(self, value: int) -> None:
        self.overall_progress.setValue(max(0, min(100, int(value))))

    def set_evaluate_enabled(self, enabled: bool) -> None:
        self.evaluate_button.setEnabled(enabled)

    def update_from_progress(self, info: dict[str, Any]) -> None:
        stage = str(info.get("stage") or "")
        pipeline_step = self._canonical_step_name(str(info.get("pipeline_step") or info.get("method") or ""))
        metric = info.get("metric")
        iteration = info.get("iteration")
        level = info.get("level")

        if not stage:
            if metric is not None and iteration not in (None, 0):
                stage = "iteration"
            elif metric is None and iteration in (0, None) and level in (0, None):
                stage = "start"
            elif metric is None and level not in (None, 0):
                stage = "level_start"
            elif metric is not None:
                stage = "iteration"

        message = str(info.get("message") or stage.replace("_", " ").title())

        if stage == "step_start" and pipeline_step:
            self._step_started_at[pipeline_step] = time.perf_counter()

        if pipeline_step in self.step_widgets:
            duration_text = None
            if stage in {"step_complete", "step_skipped"} and pipeline_step in self._step_started_at:
                elapsed = time.perf_counter() - self._step_started_at[pipeline_step]
                duration_text = self._format_duration(elapsed)
            self._update_step_widget(pipeline_step, stage, info, message, duration_text)

        if pipeline_step in self.plot_widgets:
            self._show_plot(pipeline_step)
            plot_widget = self.plot_widgets[pipeline_step]
            if metric is not None and stage in {"iteration", "level_complete", "step_complete"}:
                try:
                    plot_widget.append_metric(float(metric))
                except Exception:
                    pass
            plot_widget.set_summary_text(self._build_plot_summary(info, message, metric))

        overall = self._compute_overall_progress(info)
        self.set_overall_progress(overall)
        self.set_status(message)

        detail_parts = []
        step_index = info.get("pipeline_step_index")
        total_steps = info.get("pipeline_total_steps")
        if step_index is not None and total_steps is not None:
            detail_parts.append(f"Step {int(step_index)}/{int(total_steps)}")

        level = info.get("level")
        total_levels = info.get("total_levels")
        if level is not None and total_levels is not None:
            detail_parts.append(f"Level {int(level)}/{int(total_levels)}")

        iteration = info.get("iteration")
        if iteration is not None:
            detail_parts.append(f"Iteration {int(iteration)}")

        if metric is not None:
            try:
                detail_parts.append(f"Metric {float(metric):.6f}")
            except Exception:
                pass

        self.set_detail(" | ".join(detail_parts))

        if stage == "complete":
            self.evaluate_button.setEnabled(True)
            self.status_label.setText("Final registration complete.")
            self.detail_label.setText("Done.")
        else:
            self.evaluate_button.setEnabled(False)

    def _on_back(self) -> None:
        if self.main_window is not None:
            self.main_window.cancel_registration()

    def _on_evaluate(self) -> None:
        if self.main_window is not None:
            self.main_window.open_evaluation_page()

    def _show_plot(self, step_name: str) -> None:
        widget = self.plot_widgets.get(step_name)
        if widget is None:
            return
        index = self.plot_stack.indexOf(widget)
        if index >= 0:
            self.plot_stack.setCurrentIndex(index)

    def _update_step_widget(
        self,
        step_name: str,
        stage: str,
        info: dict[str, Any],
        message: str,
        duration_text: str | None = None,
    ) -> None:
        widget = self.step_widgets[step_name]
        progress = self._estimate_step_progress(stage, info)

        if stage in {"start", "step_start", "level_start", "iteration"}:
            widget.set_state("running", "Running", progress)
        elif stage in {"step_complete", "complete"}:
            widget.set_state("complete", "Complete", 100, duration_text=duration_text)
        elif stage == "step_skipped":
            widget.set_state("complete", "Skipped", 100, duration_text=duration_text)
        elif stage == "step_failed":
            widget.set_state("failed", message, 100)
        else:
            widget.set_state("running", message, progress)

    def _build_plot_summary(self, info: dict[str, Any], message: str, metric: Any) -> str:
        parts: list[str] = []

        step_index = info.get("pipeline_step_index")
        total_steps = info.get("pipeline_total_steps")
        if step_index is not None and total_steps is not None:
            parts.append(f"Step {int(step_index)}/{int(total_steps)}")

        level = info.get("level")
        total_levels = info.get("total_levels")
        if level is not None and total_levels is not None:
            parts.append(f"Level {int(level)}/{int(total_levels)}")

        iteration = info.get("iteration")
        if iteration is not None:
            parts.append(f"Iteration {int(iteration)}")

        if metric is not None:
            try:
                parts.append(f"Metric {float(metric):.6f}")
            except Exception:
                pass

        if not parts:
            return message
        return " | ".join(parts)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def _estimate_step_progress(self, stage: str, info: dict[str, Any]) -> int:
        if stage in {"step_complete", "complete", "step_skipped"}:
            return 100
        if stage == "step_start":
            return 10
        if stage == "level_start":
            return 15
        if stage == "iteration":
            level = max(1, int(info.get("level", 1)))
            total_levels = max(1, int(info.get("total_levels", 1)))
            return min(95, max(15, int((level / total_levels) * 95)))
        if stage == "level_complete":
            level = max(1, int(info.get("level", 1)))
            total_levels = max(1, int(info.get("total_levels", 1)))
            return min(95, int((level / total_levels) * 95))
        return 0

    def _compute_overall_progress(self, info: dict[str, Any]) -> int:
        stage = str(info.get("stage", ""))
        step_index = int(info.get("pipeline_step_index", 0) or 0)
        total_steps = max(1, int(info.get("pipeline_total_steps", 1) or 1))

        if stage == "complete":
            return 100
        if step_index <= 0:
            return 0

        step_progress = self._estimate_step_progress(stage, info)
        overall = ((step_index - 1) + (step_progress / 100.0)) / total_steps * 100.0
        return int(max(0, min(100, round(overall))))

    def _canonical_step_name(self, name: str) -> str:
        value = name.strip().lower()
        if "mask" in value:
            return "Masking"
        if "rigid" in value:
            return "Rigid Registration"
        if "demons" in value:
            return "Demons Registration"
        return name.strip() or "Step"

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
