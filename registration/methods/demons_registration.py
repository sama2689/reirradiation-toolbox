from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

import logging
logger = logging.getLogger(__name__)

from registration.methods.resample import Resample


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], None]


class DemonsMethod(str, Enum):
    DIFFEOMORPHIC  = "diffeomorphic demons"
    DEMONS         = "demons"
    FAST_SYMMETRIC = "fast symmetric demons"


@dataclass(slots=True)
class DemonsParameters:
    method                  : DemonsMethod = DemonsMethod.DIFFEOMORPHIC
    use_multiresolution     : bool         = True
    iterations              : list[int]    = field(default_factory = lambda: [200, 150, 100])
    shrink_factors          : list[int]    = field(default_factory = lambda: [4, 2, 1])
    smoothing_sigmas        : list[float]  = field(default_factory = lambda: [2.0, 1.0, 0.0])

    field_smoothing_sigma   : float = 1.0
    max_rms_error           : float = 0.1
    hu_difference_threshold : float = 1.5

    stopping_window         : int   = 10
    stopping_tolerance      : float = 1e-5

    window_centre           : int | None = None
    window_width            : int | None = None


class DemonsRegistration:
    """
    Configure and run demons image registration using SimpleITK.

    Use ``configure(params)`` to pass either a dictionary of settings or a
    ``DemonsParameters`` instance.
    """

    def __init__(self, parameters: Mapping[str, Any] | DemonsParameters | None = None) -> None:
        self.reset_defaults()
        if parameters:
            self.configure(parameters)

    def reset_defaults(self) -> "DemonsRegistration":
        self.parameters = DemonsParameters()
        return self

    def configure(self, parameters: Mapping[str, Any] | DemonsParameters) -> "DemonsRegistration":
        """
        Update settings from a dict or a ``DemonsParameters`` instance.

        Example:
        ``configure({"method": DemonsMethod.DIFFEOMORPHIC, "use_multiresolution": 
        True, "iterations": [200, 150, 100]})``
        """
        if isinstance(parameters, DemonsParameters):
            self.parameters = parameters
        else:
            for key, value in parameters.items():
                if not hasattr(self.parameters, key):
                    logger.warning("Ignoring unknown DemonsRegistration parameter '%s'.", key)
                    continue
                setattr(self.parameters, key, value)

        self._normalize_and_validate()
        return self

    def execute(
        self,
        fixed_image       : sitk.Image,
        moving_image      : sitk.Image,
        initial_transform : sitk.Transform | None = None,
        progress_callback : ProgressCallback | None = None,
        cancel_check      : CancelCheck | None = None,
    ) -> sitk.Transform:
        cancel_check = cancel_check or (lambda: None)
        p = self.parameters

        fixed_image  = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
        
        resampled_moving_image = Resample.ct(moving_image, fixed_image, transform=initial_transform)

        if p.window_centre is not None and p.window_width is not None:
            fixed_image  = self._windowing(fixed_image)
            resampled_moving_image = self._windowing(resampled_moving_image)

        levels       = self._build_levels()
        total_levels = len(levels)

        cancel_check()
        self._emit_progress(
            progress_callback,
            level        = 0,
            total_levels = total_levels,
            iteration    = 0,
            metric       = None,
        )
            
        field = sitk.Image(
            fixed_image.GetSize(),
            sitk.sitkVectorFloat64,
            fixed_image.GetDimension(),
        )
        field.CopyInformation(fixed_image)

        for level_index, (shrink_factor, smoothing_sigma, iterations) in enumerate(levels, start = 1):
            field = self._run_level(
                fixed_image       = fixed_image,
                moving_image      = resampled_moving_image,
                field             = field,
                level_index       = level_index,
                total_levels      = total_levels,
                shrink_factor     = shrink_factor,
                smoothing_sigma   = smoothing_sigma,
                iterations        = iterations,
                progress_callback = progress_callback,
                cancel_check      = cancel_check,
            )

        if initial_transform is not None:
            return sitk.CompositeTransform([initial_transform, sitk.DisplacementFieldTransform(field)])
        
        return sitk.DisplacementFieldTransform(field)

    def _normalize_and_validate(self) -> None:
        p = self.parameters

        p.method                  = self._normalize_method(p.method)
        p.use_multiresolution     = bool(p.use_multiresolution)
        p.field_smoothing_sigma   = float(p.field_smoothing_sigma)
        p.max_rms_error           = float(p.max_rms_error)
        p.hu_difference_threshold = float(p.hu_difference_threshold)
        p.stopping_window         = int(p.stopping_window)
        p.stopping_tolerance      = float(p.stopping_tolerance)

        if p.window_centre is not None:
            p.window_centre = int(p.window_centre)
        if p.window_width is not None:
            p.window_width  = int(p.window_width)

        p.iterations       = self._coerce_int_list(p.iterations, "iterations")
        p.shrink_factors   = self._coerce_int_list(p.shrink_factors, "shrink_factors")
        p.smoothing_sigmas = self._coerce_float_list(p.smoothing_sigmas, "smoothing_sigmas")

        if p.stopping_window < 2:
            raise ValueError("stopping_window must be at least 2.")
        if p.stopping_tolerance < 0.0:
            raise ValueError("stopping_tolerance must be greater than or equal to 0.")
        if not p.iterations:
            raise ValueError("iterations must not be empty.")

        if p.use_multiresolution:
            if not p.shrink_factors or not p.smoothing_sigmas:
                raise ValueError("Multi-resolution mode requires shrink_factors and smoothing_sigmas.")
            if not len(p.shrink_factors) == len(p.smoothing_sigmas) == len(p.iterations):
                raise ValueError("shrink_factors, smoothing_sigmas, and iterations must have the same length when multi-resolution is enabled.")

    def _coerce_int_list(self, value: Any, name: str) -> list[int]:
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        raise TypeError(f"{name} must be an int or a list of ints.")

    def _coerce_float_list(self, value: Any, name: str) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        raise TypeError(f"{name} must be a float or a list of floats.")

    def _normalize_method(self, method: Any) -> DemonsMethod:
        if isinstance(method, DemonsMethod):
            return method

        value = str(method).strip().lower()

        if value in {"diffeomorphic", "diffeomorphic demons"}:
            return DemonsMethod.DIFFEOMORPHIC
        if value in {"demons", "symmetric forces", "symetric forces"}:
            return DemonsMethod.DEMONS
        if value in {"fast symmetric forces", "fast symmetric demons"}:
            return DemonsMethod.FAST_SYMMETRIC

        raise ValueError(f"Unknown demons method: {method}")

    def _build_levels(self) -> list[tuple[int, float, int]]:
        p = self.parameters

        if p.use_multiresolution:
            return [
                (int(sf), float(sig), int(it))
                for sf, sig, it 
                in zip(p.shrink_factors, p.smoothing_sigmas, p.iterations)
            ]

        return [(1, 0.0, int(p.iterations[0]))]

    def _run_level(
        self,
        *,
        fixed_image       : sitk.Image,
        moving_image      : sitk.Image,
        field             : sitk.Image,
        level_index       : int,
        total_levels      : int,
        shrink_factor     : int,
        smoothing_sigma   : float,
        iterations        : int,
        progress_callback : ProgressCallback | None,
        cancel_check      : CancelCheck,
    ) -> sitk.Image:
        p = self.parameters

        cancel_check()
        self._emit_progress(
            progress_callback,
            level        = level_index,
            total_levels = total_levels,
            iteration    = 0,
            metric       = None,
        )

        if p.use_multiresolution:
            fixed_level  = self._multiresolution_resample(fixed_image, shrink_factor, smoothing_sigma)
            moving_level = self._multiresolution_resample_to_reference(moving_image, fixed_level, smoothing_sigma)
            field        = sitk.Resample(field, fixed_level)
        else:
            fixed_level = fixed_image
            moving_level = sitk.Resample(
                moving_image,
                fixed_level,
                sitk.Transform(),
                sitk.sitkLinear,
                -1000.0,
                moving_image.GetPixelID(),
            )

        registration_method = self._create_filter(number_of_iterations = iterations)
        
        iteration      = 0
        metric_history = []

        def _iteration_event() -> None:
            nonlocal iteration
            
            cancel_check()
            iteration += 1

            metric_value = self._safe_metric_value(registration_method)
            if metric_value is not None:
                metric_history.append(metric_value)

            self._emit_progress(
            progress_callback,
            level        = level_index,
            total_levels = total_levels,
            iteration    = iteration,
            metric       = metric_value,
            )

            if self._should_stop_early(metric_history):
                registration_method.StopRegistration()

        registration_method.AddCommand(sitk.sitkIterationEvent, _iteration_event)
        field = registration_method.Execute(fixed_level, moving_level, field)

        cancel_check()
        last_metric = self._safe_metric_value(registration_method)

        self._emit_progress(
            progress_callback,
            level        = level_index,
            total_levels = total_levels,
            iteration    = iteration,
            metric       = last_metric,
        )

        return field

    def _build_filter(self) -> Any:
        p = self.parameters

        if p.method == DemonsMethod.DEMONS:
            return sitk.DemonsRegistrationFilter()
        if p.method == DemonsMethod.FAST_SYMMETRIC:
            return sitk.FastSymmetricForcesDemonsRegistrationFilter()
        if p.method == DemonsMethod.DIFFEOMORPHIC:
            return sitk.DiffeomorphicDemonsRegistrationFilter()

        raise ValueError(f"Unknown demons method: {p.method}")

    def _create_filter(self, number_of_iterations: int) -> Any:
        p = self.parameters
        registration_method = self._build_filter()

        registration_method.SetNumberOfIterations(int(number_of_iterations))
        registration_method.SetMaximumRMSError(float(p.max_rms_error))
        registration_method.SetStandardDeviations(float(p.field_smoothing_sigma))
        registration_method.SetSmoothDisplacementField(True)
        registration_method.SetIntensityDifferenceThreshold(float(p.hu_difference_threshold))

        return registration_method

    def _resampled_spacing(
        self,
        original_size: tuple[int, ...] | list[int],
        original_spacing: tuple[float, ...] | list[float],
        new_size: list[int],
    ) -> list[float]:
        spacing: list[float] = []
        for osz, ospc, nsz in zip(original_size, original_spacing, new_size):
            if nsz <= 1 or osz <= 1:
                spacing.append(float(ospc))
            else:
                spacing.append(((osz - 1) * ospc) / (nsz - 1))
        return spacing

    def _multiresolution_resample(
        self,
        image: sitk.Image,
        shrink_factor: int | float | list[int],
        smoothing_sigma: float | list[float],
    ) -> sitk.Image:
        if shrink_factor is None:
            shrink_factor = 1
        if smoothing_sigma is None:
            smoothing_sigma = 0.0

        if isinstance(shrink_factor, (int, float)):
            shrink_factor = [int(shrink_factor)] * image.GetDimension()
        else:
            shrink_factor = [int(v) for v in shrink_factor]

        if isinstance(smoothing_sigma, (int, float)):
            smoothing_values = [float(smoothing_sigma)] * image.GetDimension()
        else:
            smoothing_values = [float(v) for v in smoothing_sigma]

        if any(v > 0 for v in smoothing_values):
            smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_values)
        else:
            smoothed_image = image

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_size = [
            max(1, int(size_value / float(shrink_value) + 0.5))
            for shrink_value, size_value in zip(shrink_factor, original_size)
        ]
        new_spacing = self._resampled_spacing(original_size, original_spacing, new_size)

        return sitk.Resample(
            smoothed_image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0.0,
            image.GetPixelID(),
        )

    def _multiresolution_resample_to_reference(
        self,
        image: sitk.Image,
        reference: sitk.Image,
        smoothing_sigma: float | list[float],
    ) -> sitk.Image:
        if isinstance(smoothing_sigma, (int, float)):
            smoothing_values = [float(smoothing_sigma)] * image.GetDimension()
        else:
            smoothing_values = [float(value) for value in smoothing_sigma]

        if any(value > 0 for value in smoothing_values):
            image = sitk.SmoothingRecursiveGaussian(image, smoothing_values)

        return sitk.Resample(
            image,
            reference,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            image.GetPixelID(),
        )

    def _safe_metric_value(self, method: Any) -> float | None:
        try:
            return float(method.GetMetric())
        except Exception:
            try:
                return float(method.GetMetricValue())
            except Exception:
                return None

    def _should_stop_early(self, metric_history: list[float]) -> bool:
        p = self.parameters
        if len(metric_history) < p.stopping_window:
            return False

        start = metric_history[-p.stopping_window]
        end = metric_history[-1]

        denominator = max(abs(start), 1e-12)
        relative_change = abs(start - end) / denominator
        return relative_change < p.stopping_tolerance

    def _windowing(self, image: sitk.Image) -> sitk.Image:
        p = self.parameters
        lower = p.window_centre - p.window_width / 2.0
        upper = p.window_centre + p.window_width / 2.0

        return sitk.IntensityWindowing(
            image,
            windowMinimum=lower,
            windowMaximum=upper,
            outputMinimum=lower,
            outputMaximum=upper,
        )

    def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        *,
        level: int,
        total_levels: int,
        iteration: int,
        metric: float | None,
    ) -> None:
        if progress_callback is None:
            return

        progress = {
            "level": level,
            "total_levels": total_levels,
            "iteration": iteration,
            "metric": metric,
        }

        try:
            progress_callback(progress)
        except Exception:
            pass