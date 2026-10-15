
from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import pandas as pd
import SimpleITK as sitk
from PySide6.QtCore import QObject, Signal

from registration.dataset.dataset import Structure
from registration.methods.masking import Masking
from registration.methods.resample import Resample, sum_images
from registration.metrics.metrics import RegistrationMetrics

@dataclass(slots=True)
class EvaluationCase:
    fixed_image: sitk.Image
    rigid_image: sitk.Image
    final_image: sitk.Image
    fixed_dose: sitk.Image | None = None
    rigid_moving_dose: sitk.Image | None = None
    final_moving_dose: sitk.Image | None = None
    rigid_dose_sum: sitk.Image | None = None
    final_dose_sum: sitk.Image | None = None
    fixed_structures: dict[str, sitk.Image] = field(default_factory=dict)
    rigid_structures: dict[str, sitk.Image] = field(default_factory=dict)
    final_structures: dict[str, sitk.Image] = field(default_factory=dict)
    image_metrics: pd.DataFrame | None = None
    contour_metrics: pd.DataFrame | None = None
    dose_summary: pd.DataFrame | None = None


class EvaluationWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        fixed_structures: dict[str, Structure | sitk.Image],
        moving_structures: dict[str, Structure | sitk.Image],
        fixed_dose: sitk.Image | None,
        moving_dose: sitk.Image | None,
        rigid_transform: sitk.Transform | None,
        final_transform: sitk.Transform | None,
        fixed_dose_scaling: float | None = None,
        moving_dose_scaling: float | None = None,
        pipeline_steps: list[dict[str, Any]] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.fixed_structures = self._structure_image_map(fixed_structures)
        self.moving_structures = self._structure_image_map(moving_structures)
        self.fixed_dose = fixed_dose
        self.moving_dose = moving_dose
        self.rigid_transform = rigid_transform
        self.final_transform = final_transform
        self.fixed_dose_scaling = fixed_dose_scaling
        self.moving_dose_scaling = moving_dose_scaling
        self.pipeline_steps = list(pipeline_steps or [])

    def run(self) -> None:
        try:
            evaluation_case = self._build_case()
            self.finished.emit(evaluation_case)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _build_case(self) -> EvaluationCase:
        shared_contours = self._shared_contour_names()
        total_steps = 9 + len(shared_contours)
        step_index = 0

        rigid_transform = self.rigid_transform or sitk.Transform()
        final_transform = self.final_transform or rigid_transform

        fixed_mask, moving_mask = self._build_evaluation_masks()

        def emit_progress(message: str) -> None:
            self.progress.emit(
                {
                    "current": step_index,
                    "total": total_steps,
                    "message": message,
                }
            )

        step_index = 1
        if fixed_mask is not None or moving_mask is not None:
            emit_progress("Applying evaluation mask to fixed and moving inputs...")
        else:
            emit_progress("Preparing unmasked evaluation inputs...")

        fixed_image = self._mask_ct(self.fixed_image, fixed_mask)
        moving_image = self._mask_ct(self.moving_image, moving_mask)
        fixed_structures = self._mask_contours(self.fixed_structures, fixed_mask)
        moving_structures = self._mask_contours(self.moving_structures, moving_mask)
        fixed_dose = self._mask_dose(self.fixed_dose, fixed_mask, self.fixed_dose_scaling)
        moving_dose = self._mask_dose(self.moving_dose, moving_mask, self.moving_dose_scaling)

        step_index = 2
        emit_progress("Applying rigid transform to moving CT...")
        rigid_image = Resample.ct(moving_image, fixed_image, rigid_transform)
        rigid_image = self._mask_ct(rigid_image, fixed_mask)

        step_index = 3
        emit_progress("Applying final transform to moving CT...")
        final_image = Resample.ct(moving_image, fixed_image, final_transform)
        final_image = self._mask_ct(final_image, fixed_mask)

        step_index = 4
        emit_progress("Transforming rigid contour overlays...")
        rigid_structures = Resample.structure_set(moving_structures, fixed_image, rigid_transform)
        rigid_structures = self._mask_contours(self._structure_image_map(rigid_structures), fixed_mask)

        step_index = 5
        emit_progress("Transforming final contour overlays...")
        final_structures = Resample.structure_set(moving_structures, fixed_image, final_transform)
        final_structures = self._mask_contours(self._structure_image_map(final_structures), fixed_mask)

        step_index = 6
        emit_progress("Applying dose transforms...")
        rigid_moving_dose = None
        final_moving_dose = None
        if moving_dose is not None:
            rigid_moving_dose = Resample.dose(moving_dose, fixed_image, rigid_transform)
            final_moving_dose = Resample.dose(moving_dose, fixed_image, final_transform)
            rigid_moving_dose = self._mask_dose_image(rigid_moving_dose, fixed_mask) if rigid_moving_dose is not None else None
            final_moving_dose = self._mask_dose_image(final_moving_dose, fixed_mask) if final_moving_dose is not None else None

        step_index = 7
        emit_progress("Summing dose volumes...")
        rigid_dose_sum = sum_images(fixed_dose, rigid_moving_dose)
        final_dose_sum = sum_images(fixed_dose, final_moving_dose)
        if fixed_mask is not None:
            rigid_dose_sum = self._mask_dose_image(rigid_dose_sum, fixed_mask) if rigid_dose_sum is not None else None
            final_dose_sum = self._mask_dose_image(final_dose_sum, fixed_mask) if final_dose_sum is not None else None

        step_index = 8
        emit_progress("Calculating image metrics...")
        image_metrics = RegistrationMetrics.calculate_image_metrics_table(fixed_image, rigid_image, final_image)

        contour_rows: list[dict[str, object]] = []
        if shared_contours:
            for contour_offset, contour_name in enumerate(shared_contours, start=1):
                step_index = 8 + contour_offset
                emit_progress(f"Calculating contour metrics ({contour_offset}/{len(shared_contours)}): {contour_name}")

                rigid_metrics = RegistrationMetrics.calculate_contour_metrics(
                    fixed_structures[contour_name],
                    rigid_structures[contour_name],
                )
                final_metrics = RegistrationMetrics.calculate_contour_metrics(
                    fixed_structures[contour_name],
                    final_structures[contour_name],
                )

                for metric_name, rigid_value, final_value in (
                    ("Dice", rigid_metrics["dsc"], final_metrics["dsc"]),
                    ("Jaccard", rigid_metrics["jaccard"], final_metrics["jaccard"]),
                    ("Hausdorff Distance 95", rigid_metrics["hd95"], final_metrics["hd95"]),
                    ("Mean Distance To Agreement", rigid_metrics["mda"], final_metrics["mda"]),
                ):
                    contour_rows.append(
                        {
                            "Region": contour_name,
                            "Metric": metric_name,
                            "Rigid Registration": rigid_value,
                            "Final Registration": final_value,
                        }
                    )

        contour_metrics = pd.DataFrame(
            contour_rows,
            columns=["Region", "Metric", "Rigid Registration", "Final Registration"],
        )

        step_index = total_steps
        emit_progress("Preparing evaluation data...")

        dose_summary = self._dose_summary_frame(
            fixed_dose,
            rigid_dose_sum,
            final_dose_sum,
            rigid_moving_dose,
            final_moving_dose,
        )

        return EvaluationCase(
            fixed_image=fixed_image,
            rigid_image=rigid_image,
            final_image=final_image,
            fixed_dose=fixed_dose,
            rigid_moving_dose=rigid_moving_dose,
            final_moving_dose=final_moving_dose,
            rigid_dose_sum=rigid_dose_sum,
            final_dose_sum=final_dose_sum,
            fixed_structures=fixed_structures,
            rigid_structures=rigid_structures,
            final_structures=final_structures,
            image_metrics=image_metrics,
            contour_metrics=contour_metrics,
            dose_summary=dose_summary,
        )

    def _build_evaluation_masks(self) -> tuple[sitk.Image | None, sitk.Image | None]:
        mask_params = self._mask_step_params()
        if mask_params is None:
            return None, None

        mode = str(mask_params.get("mode", "Contour")).strip().lower()
        dilation = int(mask_params.get("dilation", 0))
        erosion = int(mask_params.get("erosion", 0))

        if mode == "threshold":
            return (
                Masking.threshold_mask(
                    self.fixed_image,
                    lower_hu_threshold=int(mask_params.get("lower_hu_threshold", -200)),
                    upper_hu_threshold=int(mask_params.get("upper_hu_threshold", 3000)),
                    dilation=dilation,
                    erosion=erosion,
                ),
                Masking.threshold_mask(
                    self.moving_image,
                    lower_hu_threshold=int(mask_params.get("lower_hu_threshold", -200)),
                    upper_hu_threshold=int(mask_params.get("upper_hu_threshold", 3000)),
                    dilation=dilation,
                    erosion=erosion,
                ),
            )

        fixed_name = str(mask_params.get("fixed_contour", "")).strip()
        moving_name = str(mask_params.get("moving_contour", "")).strip()
        if fixed_name not in self.fixed_structures:
            raise ValueError(f"Fixed contour '{fixed_name}' not found for evaluation masking.")
        if moving_name not in self.moving_structures:
            raise ValueError(f"Moving contour '{moving_name}' not found for evaluation masking.")

        return (
            Masking.contour_mask(self.fixed_structures[fixed_name], dilation=dilation, erosion=erosion),
            Masking.contour_mask(self.moving_structures[moving_name], dilation=dilation, erosion=erosion),
        )

    def _structure_image_map(self, structures: dict[str, Structure | sitk.Image]) -> dict[str, sitk.Image]:
        image_map: dict[str, sitk.Image] = {}
        for name, structure in structures.items():
            image = self._structure_image(structure)
            if image is not None:
                image_map[str(name)] = image
        return image_map

    def _structure_image(self, structure: Structure | sitk.Image | Any) -> sitk.Image | None:
        if isinstance(structure, sitk.Image):
            return structure
        image = getattr(structure, "image", None)
        if isinstance(image, sitk.Image):
            return image
        return None

    def _mask_step_params(self) -> dict[str, Any] | None:
        for step in self.pipeline_steps:
            name = str(step.get("step", step.get("name", ""))).strip().lower()
            if "mask" in name:
                params = step.get("params", {})
                if isinstance(params, dict):
                    return dict(params)
        return None

    def _mask_ct(self, image: sitk.Image, mask: sitk.Image | None) -> sitk.Image:
        if mask is None:
            return sitk.Cast(image, sitk.sitkFloat32)
        return Masking.apply_mask_to_image(image, mask, background_value=-1000.0)

    def _mask_dose_image(self, image: sitk.Image, mask: sitk.Image | None) -> sitk.Image:
        if mask is None:
            return sitk.Cast(image, sitk.sitkFloat32)
        return Masking.apply_mask_to_image(image, mask, background_value=0.0)

    def _mask_contours(
        self,
        contours: dict[str, sitk.Image],
        mask: sitk.Image | None,
    ) -> dict[str, sitk.Image]:
        if mask is None:
            return dict(contours)

        masked_contours: dict[str, sitk.Image] = {}
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        for name, contour in contours.items():
            contour_mask = sitk.Cast(contour > 0, sitk.sitkUInt8)
            masked = sitk.Mask(contour_mask, binary_mask, outsideValue=0)
            masked_contours[name] = sitk.Cast(masked > 0, sitk.sitkUInt8)
        return masked_contours

    def _mask_dose(
        self,
        dose_image: sitk.Image | None,
        mask: sitk.Image | None,
        dose_scaling: float | None,
    ) -> sitk.Image | None:
        if dose_image is None:
            return None

        dose = sitk.Cast(dose_image, sitk.sitkFloat32)
        if dose_scaling is not None:
            dose = sitk.Multiply(dose, float(dose_scaling))
        if mask is not None:
            dose = Masking.apply_mask_to_image(dose, mask, background_value=0.0)
        return dose

    def _shared_contour_names(self) -> list[str]:
        return sorted(set(self.fixed_structures.keys()) & set(self.moving_structures.keys()))

    def _dose_summary_frame(
        self,
        fixed_dose: sitk.Image | None,
        rigid_dose_sum: sitk.Image | None,
        final_dose_sum: sitk.Image | None,
        rigid_moving_dose: sitk.Image | None,
        final_moving_dose: sitk.Image | None,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        if fixed_dose is not None:
            records.append({"Component": "Fixed Dose", "Mean": self._image_mean(fixed_dose), "Max": self._image_max(fixed_dose)})
        if rigid_moving_dose is not None:
            records.append({"Component": "Rigid Moving Dose", "Mean": self._image_mean(rigid_moving_dose), "Max": self._image_max(rigid_moving_dose)})
        if final_moving_dose is not None:
            records.append({"Component": "Final Moving Dose", "Mean": self._image_mean(final_moving_dose), "Max": self._image_max(final_moving_dose)})
        if rigid_dose_sum is not None:
            records.append({"Component": "Rigid Summed Dose", "Mean": self._image_mean(rigid_dose_sum), "Max": self._image_max(rigid_dose_sum)})
        if final_dose_sum is not None:
            records.append({"Component": "Final Summed Dose", "Mean": self._image_mean(final_dose_sum), "Max": self._image_max(final_dose_sum)})
        return pd.DataFrame(records, columns=["Component", "Mean", "Max"])

    def _image_mean(self, image: sitk.Image) -> float:
        arr = sitk.GetArrayFromImage(image).astype("float32")
        return float(arr.mean())

    def _image_max(self, image: sitk.Image) -> float:
        arr = sitk.GetArrayFromImage(image).astype("float32")
        return float(arr.max())
