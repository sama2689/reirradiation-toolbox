from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Mapping, Sequence

import SimpleITK as sitk

from registration.dataset.dataset import Structure
from registration.methods.demons_registration import DemonsMethod, DemonsParameters, DemonsRegistration
from registration.methods.masking import Masking
from registration.methods.cropping import Cropping
from registration.methods.rigid_registration import RigidParameters, RigidRegistration


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], None]


@dataclass
class RegistrationExecutionResult:
    final_transform: sitk.Transform = field(default_factory=sitk.Transform)
    rigid_transform: sitk.Transform | None = None
    demons_transform: sitk.Transform | None = None
    fixed_image: sitk.Image | None = None
    moving_image: sitk.Image | None = None
    rigid_registered_image: sitk.Image | None = None
    final_registered_image: sitk.Image | None = None
    fixed_mask: sitk.Image | None = None
    moving_mask: sitk.Image | None = None
    rigid_fixed_mask: sitk.Image | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)


class RegistrationExecutor:
    """Execute a multi-step image registration pipeline.

    The executor resolves pipeline entries into named steps and runs them in order,
    supporting masking, rigid registration, and demons registration stages. It
    tracks the current transform through the pipeline, applies cropping based on
    masks when requested, and emits progress updates to an optional callback.

    Args:
        progress_callback: Optional callback invoked for pipeline progress events.

    Returns:
        RegistrationExecutionResult: Aggregated output of the executed pipeline,
        including transforms, images, masks, and metadata for each step.
    """

    def __init__(self, progress_callback: ProgressCallback | None = None):
        self.progress_callback = progress_callback

    def execute(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        pipeline: Sequence[Any],
        fixed_structures: Mapping[str, Any] | None = None,
        moving_structures: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        initial_transform: sitk.Transform | None = None,
    ) -> RegistrationExecutionResult:
        """Execute the configured registration pipeline and return the results.

        The pipeline is resolved into concrete step definitions, then each step is
        applied in order to the current fixed and moving images. Masking, rigid
        alignment, and demons-based registration are supported, with the current
        transform tracked through the full process. Optional structure masks may be
        used during masking stages, and progress notifications can be emitted via
        the callback or the instance-level callback.

        Args:
            fixed_image: Fixed image used as the reference during registration.
            moving_image: Moving image transformed to align with the fixed image.
            pipeline: Sequence of pipeline entries describing the registration steps.
            fixed_structures: Optional mapping of fixed structure names to structure
                definitions or image masks used for masking operations.
            moving_structures: Optional mapping of moving structure names to structure
                definitions or image masks used for masking operations.
            progress_callback: Optional callback invoked with progress events.
            cancel_check: Optional callback used to interrupt execution when
                cancellation is requested.

        Returns:
            RegistrationExecutionResult: Aggregated outputs from the executed
            pipeline, including intermediate and final transforms, images, masks,
            and per-step metadata.
        """
        callback = progress_callback or self.progress_callback
        cancel_check = cancel_check or (lambda: None)

        cancel_check()
        steps = [self._resolve_step(step) for step in pipeline]
        total_steps = len(steps)

        working_fixed = fixed_image
        working_moving = moving_image
        fixed_mask: sitk.Image | None = None
        moving_mask: sitk.Image | None = None
        # Optional fixed metric mask used only by automatic rigid registration.
        # The ordinary fixed_mask remains the body/working mask used for image
        # masking, cropping and overlap preparation for DIR.
        rigid_fixed_mask: sitk.Image | None = None

        current_transform: sitk.Transform | None = initial_transform
        overlap_prepared = False

        fixed_structures_images = self._structure_image_map(fixed_structures)
        moving_structures_images = self._structure_image_map(moving_structures)

        result = RegistrationExecutionResult(
            final_transform=sitk.Transform(),
            rigid_transform=initial_transform,
            steps=[
                {"step": step_name, "params": copy.deepcopy(step_params)}
                for step_name, step_params in steps
            ],
        )

        self._emit_progress(
            callback,
            {
                "stage": "start",
                "pipeline_step": None,
                "pipeline_step_index": 0,
                "pipeline_total_steps": total_steps,
                "message": "Starting registration pipeline",
            },
        )

        for index, (step_name, params) in enumerate(steps, start=1):
            cancel_check()
            step_type = step_name.strip().lower()

            self._emit_progress(
                callback,
                {
                    "stage": "step_start",
                    "pipeline_step": step_name,
                    "pipeline_step_index": index,
                    "pipeline_total_steps": total_steps,
                    "message": f"Starting step {index}/{total_steps}: {step_name}",
                },
            )

            if "mask" in step_type:
                (
                    working_fixed,
                    working_moving,
                    fixed_mask,
                    moving_mask,
                    rigid_fixed_mask,
                ) = self._run_masking_step(
                    fixed_image=working_fixed,
                    moving_image=working_moving,
                    params=params,
                    fixed_structures=fixed_structures_images,
                    moving_structures=moving_structures_images,
                )
                result.fixed_mask = fixed_mask
                result.moving_mask = moving_mask
                result.rigid_fixed_mask = rigid_fixed_mask

                self._emit_progress(
                    callback,
                    {
                        "stage": "step_complete",
                        "pipeline_step": step_name,
                        "pipeline_step_index": index,
                        "pipeline_total_steps": total_steps,
                        "message": f"Completed step {index}/{total_steps}: {step_name}",
                    },
                )
                continue

            if "rigid" in step_type:
                rigid = RigidRegistration(parameters=self._build_rigid_parameters(params))

                step_callback = self._wrap_progress_callback(
                    callback,
                    pipeline_step=step_name,
                    pipeline_step_index=index,
                    pipeline_total_steps=total_steps,
                )

                rigid_transform = rigid.execute(
                    working_fixed,
                    working_moving,
                    fixed_mask=rigid_fixed_mask if rigid_fixed_mask is not None else fixed_mask,
                    moving_mask=moving_mask,
                    progress_callback=step_callback,
                    cancel_check=cancel_check,
                )

                current_transform = rigid_transform
                result.rigid_transform = rigid_transform
                result.rigid_registered_image = self._apply_transform(
                    moving_image=working_moving,
                    fixed_image=working_fixed,
                    transform=rigid_transform,
                )
                
                fixed_overlay, moving_overlay = Cropping.crop_by_axis_overlap(working_fixed, fixed_mask, working_moving, moving_mask, transform=current_transform)
                working_fixed, fixed_mask = fixed_overlay
                working_moving, moving_mask = moving_overlay
                overlap_prepared = True

                self._emit_progress(
                    callback,
                    {
                        "stage": "step_complete",
                        "pipeline_step": step_name,
                        "pipeline_step_index": index,
                        "pipeline_total_steps": total_steps,
                        "message": f"Completed step {index}/{total_steps}: {step_name}",
                    },
                )
                continue

            if "demons" in step_type:
                # When the workstation reuses a cached rigid transform, the rigid
                # step is intentionally omitted. Prepare the same overlapping ROI
                # here before DIR so the fast path is equivalent to a full run.
                if current_transform is not None and not overlap_prepared:
                    fixed_overlay, moving_overlay = Cropping.crop_by_axis_overlap(
                        working_fixed, fixed_mask, working_moving, moving_mask, transform=current_transform
                    )
                    working_fixed, fixed_mask = fixed_overlay
                    working_moving, moving_mask = moving_overlay
                    overlap_prepared = True

                demons = DemonsRegistration(parameters=self._build_demons_parameters(params))

                step_callback = self._wrap_progress_callback(
                    callback,
                    pipeline_step=step_name,
                    pipeline_step_index=index,
                    pipeline_total_steps=total_steps,
                )

                demons_transform = demons.execute(
                    working_fixed,
                    working_moving,
                    initial_transform=current_transform,
                    progress_callback=step_callback,
                    cancel_check=cancel_check,
                )

                current_transform = demons_transform
                result.demons_transform = demons_transform

                self._emit_progress(
                    callback,
                    {
                        "stage": "step_complete",
                        "pipeline_step": step_name,
                        "pipeline_step_index": index,
                        "pipeline_total_steps": total_steps,
                        "message": f"Completed step {index}/{total_steps}: {step_name}",
                    },
                )
                continue

            self._emit_progress(
                callback,
                {
                    "stage": "step_skipped",
                    "pipeline_step": step_name,
                    "pipeline_step_index": index,
                    "pipeline_total_steps": total_steps,
                    "message": f"Skipping unknown step: {step_name}",
                },
            )

        if current_transform is None:
            current_transform = sitk.Transform()

        result.final_transform = current_transform
        result.fixed_image = working_fixed
        result.moving_image = working_moving
        result.final_registered_image = self._apply_transform(
            moving_image=working_moving,
            fixed_image=working_fixed,
            transform=current_transform,
        )

        self._emit_progress(
            callback,
            {
                "stage": "complete",
                "pipeline_step": None,
                "pipeline_step_index": total_steps,
                "pipeline_total_steps": total_steps,
                "message": "Registration pipeline complete",
            },
        )

        return result

    def _resolve_step(self, step: Any) -> tuple[str, dict[str, Any]]:
        if hasattr(step, "name") and hasattr(step, "params"):
            return str(step.name), self._params_to_dict(step.params)

        if isinstance(step, dict):
            name = step.get("step", step.get("name"))
            params = step.get("params", {})
            if name is None:
                raise ValueError("Pipeline step dictionary must contain 'step' or 'name'.")
            return str(name), self._params_to_dict(params)

        raise TypeError(
            "Pipeline steps must be dicts with 'step'/'name' and 'params', or objects with .name and .params."
        )

    def _run_masking_step(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        params: dict[str, Any],
        fixed_structures: Mapping[str, sitk.Image],
        moving_structures: Mapping[str, sitk.Image],
    ) -> tuple[
        sitk.Image,
        sitk.Image,
        sitk.Image | None,
        sitk.Image | None,
        sitk.Image | None,
    ]:
        """Build fast registration masks and optionally crop the working CTs.

        The returned fifth value is the fixed metric mask for automatic rigid
        registration. Normally it is the same as ``fixed_mask``. If a fixed ROI
        contour is requested, it is a physically cropped body mask limited to
        that contour's superior/inferior extent. The ordinary ``fixed_mask`` is
        deliberately retained for working-image masking and later DIR overlap
        preparation.

        ``mode=Auto`` uses a requested contour (normally ``body``) when it is
        available for that dataset and independently falls back to thresholding
        when it is not. This deliberately supports the mixed workflow shown in
        example.py (e.g. fixed BODY contour, moving HU threshold).
        """
        mode = str(params.get("mode", "Auto")).strip().lower()
        # The workstation can choose mask sources independently for fixed and
        # moving. Fall back to the legacy shared mode for older pipelines.
        fixed_mode = str(params.get("fixed_mask_mode", mode)).strip().lower()
        moving_mode = str(params.get("moving_mask_mode", mode)).strip().lower()
        dilation = int(params.get("dilation", 0))
        erosion = int(params.get("erosion", 0))
        lower = float(params.get("lower_hu_threshold", -200))
        upper = float(params.get("upper_hu_threshold", 3000))

        fixed_name = str(params.get("fixed_contour", "body")).strip()
        moving_name = str(params.get("moving_contour", "body")).strip()

        def normalise_name(value: object) -> str:
            return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

        def resolve_contour(
            structures: Mapping[str, sitk.Image],
            requested_name: str,
            *,
            automatic: bool,
        ) -> tuple[str, sitk.Image] | None:
            by_normalised_name = {normalise_name(key): (key, value) for key, value in structures.items()}
            requested_normalised = normalise_name(requested_name)

            # Automatic BODY selection deliberately prefers BODYWITHOUTBOARD.
            # Manual contour selection is exact: explicitly choosing BODY must
            # use BODY even when BODYWITHOUTBOARD is also present.
            if automatic and requested_normalised == "body":
                preferred = by_normalised_name.get("bodywithoutboard")
                if preferred is not None:
                    return preferred

            return by_normalised_name.get(requested_normalised)

        def make_mask(
            image: sitk.Image,
            structures: Mapping[str, sitk.Image],
            contour_name: str,
            side: str,
            side_mode: str,
        ) -> sitk.Image:
            side_mode = str(side_mode).strip().lower()
            if side_mode == "contour":
                resolved = resolve_contour(structures, contour_name, automatic=False)
                if resolved is None:
                    raise ValueError(f"{side} contour '{contour_name}' not found.")
                _, contour_image = resolved
                return Masking.contour_mask(
                    contour_image,
                    dilation=dilation,
                    erosion=erosion,
                )

            if side_mode == "auto":
                resolved = resolve_contour(structures, contour_name, automatic=True)
                if resolved is not None:
                    _, contour_image = resolved
                    return Masking.contour_mask(
                        contour_image,
                        dilation=dilation,
                        erosion=erosion,
                    )
                # No suitable contour: fall through to threshold-derived body.

            if side_mode not in {"auto", "threshold"}:
                raise ValueError(f"Unsupported {side} masking mode: {side_mode}")

            return Masking.threshold_mask(
                image,
                lower_threshold=lower,
                upper_threshold=upper,
                dilation=dilation,
                erosion=erosion,
            )

        fixed_mask = make_mask(fixed_image, fixed_structures, fixed_name, "Fixed", fixed_mode)
        moving_mask = make_mask(moving_image, moving_structures, moving_name, "Moving", moving_mode)

        # Match the fast demonstration workflow: replace irrelevant voxels before
        # registration, then crop to a small padded ROI. Physical coordinates are
        # preserved by SimpleITK, so the resulting transform remains valid when
        # applied to the original full-resolution datasets for display/export.
        fixed_image = Masking.apply_mask_to_image(fixed_image, fixed_mask)
        moving_image = Masking.apply_mask_to_image(moving_image, moving_mask)

        if params.get("crop_based_on_mask", False):
            padding = max(0, int(params.get("crop_padding", 5)))
            fixed_image, fixed_mask = Cropping.crop_image_and_mask(
                fixed_image, fixed_mask, padding=padding
            )
            moving_image, moving_mask = Cropping.crop_image_and_mask(
                moving_image, moving_mask, padding=padding
            )

        # Optional SI restriction of the fixed metric mask for rigid registration.
        # The selected contour is first resampled onto the *current* fixed-mask
        # grid so its voxel indices correspond after any earlier body crop.
        # crop_body_to_contour_extent then returns a smaller mask which is valid
        # in the same physical space and can be passed directly to
        # RigidRegistration.execute(fixed_mask=...).
        rigid_fixed_mask = fixed_mask
        fixed_roi_name = params.get("fixed_roi_contour")
        if fixed_roi_name not in {None, ""}:
            resolved_roi = resolve_contour(
                fixed_structures,
                str(fixed_roi_name),
                automatic=False,
            )
            if resolved_roi is None:
                raise ValueError(f"Fixed ROI contour '{fixed_roi_name}' not found.")

            _, roi_contour_image = resolved_roi
            roi_on_fixed_mask_grid = sitk.Resample(
                sitk.Cast(roi_contour_image > 0, sitk.sitkUInt8),
                fixed_mask,
                sitk.Transform(fixed_mask.GetDimension(), sitk.sitkIdentity),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
            rigid_fixed_mask = Cropping.crop_body_to_contour_extent(
                body_mask=fixed_mask,
                contour_mask=roi_on_fixed_mask_grid,
                inferior_margin_mm=float(params.get("fixed_roi_inferior_margin_mm", 20.0)),
                superior_margin_mm=float(params.get("fixed_roi_superior_margin_mm", 20.0)),
            )

        return fixed_image, moving_image, fixed_mask, moving_mask, rigid_fixed_mask

    def _crop_image_and_mask(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> tuple[sitk.Image, sitk.Image]:
        bbox = self._mask_bounding_box(mask)
        if bbox is None:
            return image, mask

        index, size = bbox
        cropped_image = sitk.RegionOfInterest(image, size, index)
        cropped_mask = sitk.RegionOfInterest(mask, size, index)
        return cropped_image, cropped_mask

    def _mask_bounding_box(self, mask: sitk.Image) -> tuple[list[int], list[int]] | None:
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(binary_mask)

        if not stats.HasLabel(1):
            return None

        bbox = stats.GetBoundingBox(1)
        dim = mask.GetDimension()

        index = list(bbox[:dim])
        size = list(bbox[dim:])

        return index, size

    def _build_rigid_parameters(self, params: Mapping[str, Any]) -> RigidParameters:
        metric = self._map_rigid_metric(params.get("metric", "Correlation"))
        use_multiresolution = bool(params.get("use_multiresolution", True))
        iterations = params.get("iterations", 200)
        if use_multiresolution:
            if isinstance(iterations, int):
                iterations = [iterations]
        else:
            if isinstance(iterations, list):
                iterations = iterations[0] if iterations else 200
            if isinstance(iterations, tuple):
                iterations = iterations[0] if iterations else 200
            if not isinstance(iterations, list):
                iterations = [int(iterations)]

        return RigidParameters(
            initialisation=str(params.get("initial_transform", "Moments")),
            metric=metric,
            sampling_strategy=str(params.get("sampling_strategy", "None")),
            sampling_percentage=float(params.get("sampling_percentage", 0.25)),
            optimizer="RegularStepGradientDescent",
            optimizer_kwargs={
                "learningRate": float(params.get("learning_rate", 1.0)),
                "minStep": float(params.get("min_step", 1e-4)),
                "relaxationFactor": float(params.get("relaxation_factor", 0.3)),
                "gradientMagnitudeTolerance": float(params.get("gradient_tolerance", 1e-6)),
            },
            use_multiresolution=use_multiresolution,
            iterations=list(iterations),
            shrink_factors=self._as_int_list(params.get("shrink_factors", [4, 2, 1])),
            smoothing_sigmas=self._as_float_list(params.get("smoothing_sigmas", [2.0, 1.0, 0.0])),
            window_centre=self._optional_int(params.get("window_centre")),
            window_width=self._optional_int(params.get("window_width")),
        )

    def _build_demons_parameters(self, params: Mapping[str, Any]) -> DemonsParameters:
        use_multiresolution = bool(params.get("use_multiresolution", True))
        iterations = params.get("iterations", [200, 150, 100])
        if use_multiresolution:
            if isinstance(iterations, int):
                iterations = [iterations]
        else:
            if isinstance(iterations, list):
                iterations = iterations[0] if iterations else 200
            if isinstance(iterations, tuple):
                iterations = iterations[0] if iterations else 200
            if not isinstance(iterations, list):
                iterations = [int(iterations)]

        return DemonsParameters(
            method=self._map_demons_method(params.get("method", "Diffeomorphic")),
            use_multiresolution=use_multiresolution,
            iterations=list(iterations),
            shrink_factors=self._as_int_list(params.get("shrink_factors", [4, 2, 1])),
            smoothing_sigmas=self._as_float_list(params.get("smoothing_sigmas", [2.0, 1.0, 0.0])),
            field_smoothing_sigma=float(params.get("field_smoothing_sigma", 1.0)),
            max_rms_error=float(params.get("max_rms_error", 0.1)),
            hu_difference_threshold=float(params.get("hu_difference_threshold", 1.5)),
            stopping_window=int(params.get("stopping_window", 10)),
            stopping_tolerance=float(params.get("stopping_tolerance", 1e-5)),
            window_centre=self._optional_int(params.get("window_centre")),
            window_width=self._optional_int(params.get("window_width")),
        )

    def _as_int_list(self, value: Any) -> list[int]:
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, tuple):
            return [int(v) for v in value]
        if isinstance(value, list):
            return [int(v) for v in value]
        return [int(value)]

    def _as_float_list(self, value: Any) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, tuple):
            return [float(v) for v in value]
        if isinstance(value, list):
            return [float(v) for v in value]
        return [float(value)]

    def _map_rigid_metric(self, metric: Any) -> str:
        value = str(metric).strip()
        metric_map = {
            "Mean Squares": "MeanSquares",
            "MeanSquares": "MeanSquares",
            "Mutual Information": "MattesMutualInformation",
            "Mattes Mutual Information": "MattesMutualInformation",
            "MattesMutualInformation": "MattesMutualInformation",
        }
        return metric_map.get(value, value.replace(" ", ""))

    def _map_demons_method(self, method: Any) -> DemonsMethod:
        value = str(method).strip().lower()
        if value in {"diffeomorphic", "diffeomorphic demons"}:
            return DemonsMethod.DIFFEOMORPHIC
        if value in {"demons", "symetric forces", "symmetric forces"}:
            return DemonsMethod.DEMONS
        if value in {"fast symmetric demons", "fast symmetric forces"}:
            return DemonsMethod.FAST_SYMMETRIC
        return DemonsMethod.DIFFEOMORPHIC

    def _optional_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    def _structure_image_map(self, structures: Mapping[str, Any] | None) -> dict[str, sitk.Image]:
        if not structures:
            return {}

        normalized: dict[str, sitk.Image] = {}
        for key, value in structures.items():
            image = self._structure_image(value)
            if image is not None:
                normalized[str(key)] = image
        return normalized

    def _structure_image(self, value: Any) -> sitk.Image | None:
        if isinstance(value, sitk.Image):
            return value
        image = getattr(value, "image", None)
        if isinstance(image, sitk.Image):
            return image
        return None

    def _params_to_dict(self, params: Any) -> dict[str, Any]:
        if params is None:
            return {}
        if isinstance(params, Mapping):
            return dict(params)
        if is_dataclass(params):
            return asdict(params)
        if hasattr(params, "__dict__"):
            return {k: v for k, v in vars(params).items() if not k.startswith("_")}
        raise TypeError(
            f"Step parameters must be a mapping or dataclass-like object, got {type(params).__name__}."
        )

    def _apply_transform(
        self,
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        transform: sitk.Transform,
    ) -> sitk.Image:
        return sitk.Resample(
            moving_image,
            fixed_image,
            transform,
            sitk.sitkLinear,
            -1000.0,
            moving_image.GetPixelID(),
        )

    def _wrap_progress_callback(
        self,
        callback: ProgressCallback | None,
        *,
        pipeline_step: str,
        pipeline_step_index: int,
        pipeline_total_steps: int,
    ) -> ProgressCallback | None:
        if callback is None:
            return None

        def wrapped(info: dict[str, Any]) -> None:
            payload = dict(info)
            payload.setdefault("pipeline_step", pipeline_step)
            payload.setdefault("pipeline_step_index", pipeline_step_index)
            payload.setdefault("pipeline_total_steps", pipeline_total_steps)
            try:
                callback(payload)
            except Exception:
                pass

        return wrapped

    def _emit_progress(
        self,
        callback: ProgressCallback | None,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            pass
