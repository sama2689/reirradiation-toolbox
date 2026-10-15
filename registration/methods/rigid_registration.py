from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
import logging
import math

import SimpleITK as sitk

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], None]


OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    "RegularStepGradientDescent": {
        "learningRate"               : 1.0,
        "minStep"                    : 1e-4,
        "relaxationFactor"           : 0.3,
        "gradientMagnitudeTolerance" : 1e-6,
    },
    "GradientDescent": {
        "learningRate"               : 1.0,
        "convergenceMinimumValue"    : 1e-6,
        "convergenceWindowSize"      : 10,
    },
    "GradientDescentLineSearch": {
        "learningRate"               : 1.0,
        "convergenceMinimumValue"    : 1e-6,
        "convergenceWindowSize"      : 10,
    },
    "ConjugateGradientLineSearch": {
        "learningRate"               : 1.0,
        "convergenceMinimumValue"    : 1e-6,
        "convergenceWindowSize"      : 10,
    },
}


@dataclass(slots=True)
class RigidParameters:
    initialisation      : str = "Moments"
    metric              : str = "Correlation"
    sampling_strategy   : str = "None"
    sampling_percentage : float = 0.25
    interpolator        : int = sitk.sitkLinear

    optimizer           : str = "RegularStepGradientDescent"
    optimizer_kwargs    : dict[str, Any] = field(
        default_factory = lambda: deepcopy(OPTIMIZER_DEFAULTS["RegularStepGradientDescent"])
    )

    use_multiresolution : bool = True
    iterations          : list[int]  = field(default_factory = lambda: [200, 150, 100])
    shrink_factors      : list[int]  = field(default_factory = lambda: [4, 2, 1])
    smoothing_sigmas    : list[float] = field(default_factory = lambda: [2.0, 1.0, 0.0])

    window_centre       : int | None = None
    window_width        : int | None = None

    manual_translation  : list[float] = field(default_factory = lambda: [0.0, 0.0, 0.0])
    manual_rotation     : list[float] = field(default_factory = lambda: [0.0, 0.0, 0.0])


class RigidRegistration:
    """Configure and execute rigid image registration using SimpleITK."""

    def __init__(self, parameters: Mapping[str, Any] | RigidParameters | None = None) -> None:
        self.reset_defaults()
        if parameters:
            self.configure(parameters)

    def reset_defaults(self) -> "RigidRegistration":
        """Restore all registration parameters to their default values."""
        self.parameters = RigidParameters()
        return self

    def configure(self, parameters: Mapping[str, Any] | RigidParameters) -> "RigidRegistration":
        """
        Update registration parameters. Unknown parameter names are ignored with a warning.

        Args:
            parameters: Mapping of parameter names to values, or a RigidParameters instance.
        """
        if isinstance(parameters, RigidParameters):
            self.parameters = parameters
        else:
            for key, value in parameters.items():
                if not hasattr(self.parameters, key):
                    logger.warning("Ignoring unknown RigidRegistration parameter '%s'.", key)
                    continue
                setattr(self.parameters, key, value)

        self._normalize_and_validate()
        return self

    def execute(
        self,
        fixed_image       : sitk.Image,
        moving_image      : sitk.Image,
        fixed_mask        : sitk.Image | None = None,
        moving_mask       : sitk.Image | None = None,
        progress_callback : ProgressCallback | None = None,
        cancel_check      : CancelCheck | None = None,
    ) -> sitk.Transform:
        """
        Perform rigid image registration.

        Args:
            fixed_image:
                Fixed reference image.
            moving_image:
                Moving image to be registered.
            fixed_mask:
                Optional mask defining the region of interest in the fixed image.
            moving_mask:
                Optional mask defining the region of interest in the moving image.
            progress_callback:
                Optional callback invoked during optimisation.
            cancel_check:
                Optional callback used to cancel registration.

        Returns:
            sitk.Transform:
                The SimpleITK resampling transform used to sample the moving image
                on the fixed image grid (fixed/output -> moving/input physical space).
        """
        cancel_check = cancel_check or (lambda: None)
        p = self.parameters

        fixed_image  = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        if p.window_centre is not None and p.window_width is not None:
            fixed_image  = self._apply_window(fixed_image, p.window_centre, p.window_width)
            moving_image = self._apply_window(moving_image, p.window_centre, p.window_width)

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

        current_transform : sitk.Transform | None = None

        for level_index, (shrink_factor, smoothing_sigma, iterations) in enumerate(levels, start=1):
            current_transform = self._run_level(
                fixed_image       = fixed_image,
                moving_image      = moving_image,
                fixed_mask        = fixed_mask,
                moving_mask       = moving_mask,
                current_transform = current_transform,
                level_index       = level_index,
                total_levels      = total_levels,
                shrink_factor     = shrink_factor,
                smoothing_sigma   = smoothing_sigma,
                iterations        = iterations,
                progress_callback = progress_callback,
                cancel_check      = cancel_check,
            )

        if current_transform is None:
            current_transform = self._initial_transform(fixed_image, moving_image)

        return current_transform

    def _normalize_and_validate(self) -> None:
        """Normalise parameter values and validate the current configuration."""
        p = self.parameters

        p.initialisation = self._normalize_choice(
            p.initialisation,
            {"Moments", "Geometry", "Manual"},
            "initialisation",
        )
        p.metric = self._normalize_choice(
            p.metric,
            {
                "Correlation",
                "MeanSquares",
                "Demons",
                "ANTSNeighborhoodCorrelation",
                "MattesMutualInformation",
                "JointHistogramMutualInformation",
            },
            "metric",
        )
        p.sampling_strategy = self._normalize_choice(
            p.sampling_strategy,
            {"None", "Random", "Regular"},
            "sampling_strategy",
        )
        p.optimizer = self._normalize_choice(
            p.optimizer,
            set(OPTIMIZER_DEFAULTS.keys()),
            "optimizer",
        )

        p.sampling_percentage = float(p.sampling_percentage)
        p.use_multiresolution = bool(p.use_multiresolution)

        if p.window_centre is not None:
            p.window_centre = int(p.window_centre)
        if p.window_width is not None:
            p.window_width = int(p.window_width)

        p.iterations         = self._coerce_int_list(p.iterations, "iterations")
        p.shrink_factors     = self._coerce_int_list(p.shrink_factors, "shrink_factors")
        p.smoothing_sigmas   = self._coerce_float_list(p.smoothing_sigmas, "smoothing_sigmas")
        p.manual_translation = self._coerce_float_list(p.manual_translation, "manual_translation")
        p.manual_rotation    = self._coerce_float_list(p.manual_rotation, "manual_rotation")

        if not p.iterations:
            raise ValueError("iterations must not be empty.")
        if p.sampling_percentage < 0.0:
            raise ValueError("sampling_percentage must be greater than or equal to 0.")

        if not isinstance(p.optimizer_kwargs, dict):
            p.optimizer_kwargs = dict(p.optimizer_kwargs)

        if "numberOfIterations" in p.optimizer_kwargs:
            logger.warning(
                "Ignoring redundant optimizer_kwargs['numberOfIterations']; use parameters.iterations instead."
            )
            p.optimizer_kwargs.pop("numberOfIterations", None)

        defaults = deepcopy(OPTIMIZER_DEFAULTS[p.optimizer])
        defaults.update(p.optimizer_kwargs)
        p.optimizer_kwargs = defaults

        if p.use_multiresolution:
            if not p.shrink_factors or not p.smoothing_sigmas:
                raise ValueError(
                    "Multi-resolution mode requires shrink_factors and smoothing_sigmas."
                )
            if len(p.iterations) != len(p.shrink_factors) or len(p.iterations) != len(p.smoothing_sigmas):
                raise ValueError(
                    "iterations, shrink_factors, and smoothing_sigmas must have the same length "
                    "when multi-resolution is enabled."
                )
        else:
            p.iterations = [int(p.iterations[0])]

    def _run_level(
        self,
        *,
        fixed_image       : sitk.Image,
        moving_image      : sitk.Image,
        fixed_mask        : sitk.Image | None,
        moving_mask       : sitk.Image | None,
        current_transform : sitk.Transform | None,
        level_index       : int,
        total_levels      : int,
        shrink_factor     : int,
        smoothing_sigma   : float,
        iterations        : int,
        progress_callback : ProgressCallback | None,
        cancel_check      : CancelCheck,
    ) -> sitk.Transform:
        """Execute registration for a single multi-resolution level."""
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
            fixed_level = self._multiresolution_resample(
                image           = fixed_image,
                shrink_factor   = shrink_factor,
                smoothing_sigma = smoothing_sigma,
            )
            moving_level = self._multiresolution_resample(
                image           = moving_image,
                shrink_factor   = shrink_factor,
                smoothing_sigma = smoothing_sigma,
            )
        else:
            fixed_level  = fixed_image
            moving_level = moving_image

        fixed_mask_level  = (self._resample_mask(fixed_mask, fixed_level) if fixed_mask is not None else None)
        moving_mask_level = (self._resample_mask(moving_mask, moving_level) if moving_mask is not None else None)

        registration_method = self._create_method(iterations)

        if fixed_mask_level is not None:
            registration_method.SetMetricFixedMask(sitk.Cast(fixed_mask_level > 0, sitk.sitkUInt8))
        if moving_mask_level is not None:
            registration_method.SetMetricMovingMask(sitk.Cast(moving_mask_level > 0, sitk.sitkUInt8))

        if current_transform is None:
            current_transform = self._initial_transform(fixed_level, moving_level)

        registration_method.SetInitialTransform(current_transform, inPlace=False)

        iteration = 0
        metric_history: list[float] = []

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

        registration_method.AddCommand(sitk.sitkIterationEvent, _iteration_event)
        current_transform = registration_method.Execute(fixed_level, moving_level)

        return current_transform

    def _create_method(self, number_of_iterations: int) -> sitk.ImageRegistrationMethod:
        """Create and configure a SimpleITK image registration method."""
        p = self.parameters
        registration_method = sitk.ImageRegistrationMethod()

        if p.metric == "Correlation":
            registration_method.SetMetricAsCorrelation()
        elif p.metric == "MeanSquares":
            registration_method.SetMetricAsMeanSquares()
        elif p.metric == "Demons":
            registration_method.SetMetricAsDemons()
        elif p.metric == "ANTSNeighborhoodCorrelation":
            registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=4)
        elif p.metric == "MattesMutualInformation":
            registration_method.SetMetricAsMattesMutualInformation()
        elif p.metric == "JointHistogramMutualInformation":
            registration_method.SetMetricAsJointHistogramMutualInformation()
        else:
            raise ValueError(f"Unknown metric: {p.metric}")

        if p.sampling_strategy == "None":
            registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        elif p.sampling_strategy == "Random":
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(p.sampling_percentage)
        elif p.sampling_strategy == "Regular":
            registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
            registration_method.SetMetricSamplingPercentage(p.sampling_percentage)
        else:
            raise ValueError(f"Unknown sampling strategy: {p.sampling_strategy}")

        registration_method.SetInterpolator(p.interpolator)
        self._configure_optimizer(registration_method, number_of_iterations)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        return registration_method

    def _configure_optimizer(self, registration_method: sitk.ImageRegistrationMethod, number_of_iterations: int) -> None:
        """Configure the selected optimiser for the registration method."""
        p = self.parameters
        okw = p.optimizer_kwargs

        if p.optimizer == "RegularStepGradientDescent":
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate               = okw["learningRate"],
                minStep                    = okw["minStep"],
                numberOfIterations         = number_of_iterations,
                relaxationFactor           = okw["relaxationFactor"],
                gradientMagnitudeTolerance = okw["gradientMagnitudeTolerance"],
            )
        elif p.optimizer == "GradientDescent":
            registration_method.SetOptimizerAsGradientDescent(
                learningRate              = okw["learningRate"],
                numberOfIterations        = number_of_iterations,
                convergenceMinimumValue   = okw["convergenceMinimumValue"],
                convergenceWindowSize     = okw["convergenceWindowSize"],
            )
        elif p.optimizer == "GradientDescentLineSearch":
            registration_method.SetOptimizerAsGradientDescentLineSearch(
                learningRate              = okw["learningRate"],
                numberOfIterations        = number_of_iterations,
                convergenceMinimumValue   = okw["convergenceMinimumValue"],
                convergenceWindowSize     = okw["convergenceWindowSize"],
            )
        elif p.optimizer == "ConjugateGradientLineSearch":
            registration_method.SetOptimizerAsConjugateGradientLineSearch(
                learningRate              = okw["learningRate"],
                numberOfIterations        = number_of_iterations,
                convergenceMinimumValue   = okw["convergenceMinimumValue"],
                convergenceWindowSize     = okw["convergenceWindowSize"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {p.optimizer}")

    def _initial_transform(
        self, 
        fixed_image  : sitk.Image, 
        moving_image : sitk.Image
    ) -> sitk.Transform:
        """Create the initial transform for registration."""
        p = self.parameters
        dim = fixed_image.GetDimension()

        if p.initialisation == "Manual":
            if dim == 2:
                tx, ty = p.manual_translation[:2]
                angle_deg = p.manual_rotation[0]

                transform = sitk.Euler2DTransform()
                transform.SetAngle(math.radians(angle_deg))
                transform.SetTranslation((tx, ty))
                return transform

            if dim == 3:
                tx, ty, tz = p.manual_translation[:3]
                rx, ry, rz = p.manual_rotation[:3]

                transform = sitk.Euler3DTransform()
                transform.SetRotation(
                    math.radians(rx),
                    math.radians(ry),
                    math.radians(rz),
                )
                transform.SetTranslation((tx, ty, tz))
                return transform

            raise ValueError(f"Unsupported dimension for manual initialisation: {dim}")

        if dim == 2:
            transform = sitk.Euler2DTransform()
        elif dim == 3:
            transform = sitk.VersorRigid3DTransform()
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        if p.initialisation == "Moments":
            init_mode = sitk.CenteredTransformInitializerFilter.MOMENTS
        elif p.initialisation == "Geometry":
            init_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY
        else:
            raise ValueError(f"Unknown initialisation: {p.initialisation}")

        return sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            transform,
            init_mode,
        )

    def _build_levels(self) -> list[tuple[int, float, int]]:
        """Construct the list of multi-resolution pyramid levels."""
        p = self.parameters

        if p.use_multiresolution:
            return [
                (int(sf), float(sig), int(it))
                for sf, sig, it in zip(p.shrink_factors, p.smoothing_sigmas, p.iterations)
            ]

        return [(1, 0.0, int(p.iterations[0]))]

    def _multiresolution_resample(
        self,
        image           : sitk.Image,
        shrink_factor   : int | float,
        smoothing_sigma : float | list[float],
    ) -> sitk.Image:
        """Generate a smoothed and downsampled image for one pyramid level."""
        if isinstance(shrink_factor, (int, float)):
            shrink_vector = [int(shrink_factor)] * image.GetDimension()
        else:
            shrink_vector = [int(v) for v in shrink_factor]

        if isinstance(smoothing_sigma, (int, float)):
            smoothing_values = [float(smoothing_sigma)] * image.GetDimension()
        else:
            smoothing_values = [float(v) for v in smoothing_sigma]

        if any(v > 0 for v in smoothing_values):
            image = sitk.SmoothingRecursiveGaussian(image, smoothing_values)

        original_size    = image.GetSize()
        original_spacing = image.GetSpacing()

        new_size = [
            max(1, int(sz / float(sf) + 0.5))
            for sz, sf in zip(original_size, shrink_vector)
        ]
        new_spacing = self._resampled_spacing(original_size, original_spacing, new_size)

        return sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0.0,
            image.GetPixelID(),
        )

    def _resample_mask(self, mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
        """Resample a mask onto the reference image grid."""
        return sitk.Resample(
            sitk.Cast(mask > 0, sitk.sitkUInt8),
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

    def _resampled_spacing(
        self,
        original_size: tuple[int, ...] | list[int],
        original_spacing: tuple[float, ...] | list[float],
        new_size: list[int],
    ) -> list[float]:
        """Compute voxel spacing for a resampled image."""
        spacing: list[float] = []
        for osz, ospc, nsz in zip(original_size, original_spacing, new_size):
            if nsz <= 1 or osz <= 1:
                spacing.append(float(ospc))
            else:
                spacing.append(((osz - 1) * ospc) / (nsz - 1))
        return spacing

    def _apply_window(self, image: sitk.Image, level: int, width: int) -> sitk.Image:
        """Apply intensity windowing to an image."""
        window_min = level - (width / 2.0)
        window_max = level + (width / 2.0)

        return sitk.IntensityWindowing(
            image,
            windowMinimum=window_min,
            windowMaximum=window_max,
            outputMinimum=window_min,
            outputMaximum=window_max,
        )

    def _safe_metric_value(self, method: Any) -> float | None:
        """Return the current metric value if available."""
        try:
            return float(method.GetMetricValue())
        except Exception:
            try:
                return float(method.GetMetric())
            except Exception:
                return None

    def _normalize_choice(self, value: Any, allowed: set[str], label: str) -> str:
        """Validate a string option and return its used value."""
        text = str(value).strip()
        for option in allowed:
            if option.lower() == text.lower():
                return option
        raise ValueError(f"Unknown {label}: {value}")

    def _coerce_int_list(self, value: Any, name: str) -> list[int]:
        """Convert an integer or sequence of integers into a list."""
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        raise TypeError(f"{name} must be an int or a list of ints.")

    def _coerce_float_list(self, value: Any, name: str) -> list[float]:
        """Convert a numeric value or sequence into a list of floats."""
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        raise TypeError(f"{name} must be a float or a list of floats.")

    def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        *,
        level: int,
        total_levels: int,
        iteration: int,
        metric: float | None,
    ) -> None:
        """Invoke the progress callback with the current registration state."""
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


def _xyz(values: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    """Return exactly three finite float values."""
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result  # type: ignore[return-value]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _transpose3(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _matvec3(a: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(a[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _xyz_rotation_matrix(
    rotation_degrees: tuple[float, float, float],
) -> list[list[float]]:
    """Return a right-handed XYZ Euler matrix, applied X then Y then Z.

    The resulting matrix is ``Rz @ Ry @ Rx``.  This is used only to define
    the workstation's DICOM-LPS XYZ input convention and to reproduce the
    Rotation/Orientation relationship displayed by Eclipse's Registration
    Technical tab.
    """
    rx_deg, ry_deg, rz_deg = _xyz(rotation_degrees, "rotation_degrees")
    rx, ry, rz = (math.radians(v) for v in (rx_deg, ry_deg, rz_deg))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry_m = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rz_m = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return _matmul3(rz_m, _matmul3(ry_m, rx_m))


def _sitk_rotation_matrix(
    rotation_degrees: tuple[float, float, float],
) -> list[list[float]]:
    """Return the matrix produced by SimpleITK Euler3DTransform by default.

    ``Euler3DTransform`` with ``ComputeZYX=False`` uses ``Rz @ Rx @ Ry``.
    ``ManualRigidTransform`` explicitly fixes ComputeZYX to False so the
    conversion functions and the transform constructor cannot drift apart.
    """
    rx_deg, ry_deg, rz_deg = _xyz(rotation_degrees, "rotation_degrees")
    rx, ry, rz = (math.radians(v) for v in (rx_deg, ry_deg, rz_deg))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry_m = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rz_m = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return _matmul3(rz_m, _matmul3(rx_m, ry_m))


def _matrix_to_sitk_euler(
    matrix: list[list[float]],
) -> tuple[float, float, float]:
    """Decompose a rotation matrix into SimpleITK's default Euler order.

    For ``R = Rz @ Rx @ Ry``:
        R[2,1] = sin(x)
        z = atan2(-R[0,1], R[1,1])
        y = atan2(-R[2,0], R[2,2])

    The gimbal-lock branch chooses y=0 and absorbs the remaining in-plane
    rotation into z.  The resulting matrix is still identical.
    """
    sx = max(-1.0, min(1.0, float(matrix[2][1])))
    rx = math.asin(sx)
    cx = math.cos(rx)

    if abs(cx) > 1e-10:
        rz = math.atan2(-matrix[0][1], matrix[1][1])
        ry = math.atan2(-matrix[2][0], matrix[2][2])
    else:
        # At +/-90 degrees about X, Y and Z are not independently unique.
        # Choose Y=0 and solve the remaining Z rotation from the top-left 2x2.
        ry = 0.0
        rz = math.atan2(matrix[1][0], matrix[0][0])

    return tuple(math.degrees(v) for v in (rx, ry, rz))  # type: ignore[return-value]


def _forward_to_sitk_resampling(
    translation: tuple[float, float, float],
    rotation: list[list[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a forward moving->fixed rigid matrix to SITK resampling parameters.

    Image resampling asks where an output/fixed-space point should be sampled in
    the input/moving image, therefore the transform passed to ``sitk.Resample``
    is the inverse of the intuitive moving/source -> fixed/registered transform.
    """
    t = _xyz(translation, "translation")
    inverse_rotation = _transpose3(rotation)
    rotated_t = _matvec3(inverse_rotation, t)
    inverse_translation = tuple(-v for v in rotated_t)
    sitk_rotation = _matrix_to_sitk_euler(inverse_rotation)
    return inverse_translation, sitk_rotation  # type: ignore[return-value]


def dicom_to_sitk(
    translation: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a DICOM-LPS manual registration to SimpleITK parameters.

    Inputs describe the *forward* moving/source -> fixed/registered rigid
    registration in DICOM LPS coordinates.  Rotations are right-handed XYZ
    Euler angles, applied X then Y then Z (``Rz @ Ry @ Rx``).

    Returns ``(translation_mm, rotation_degrees)`` suitable for passing directly
    to :meth:`ManualRigidTransform.build`.  The returned transform is the inverse
    fixed/output -> moving/input mapping required by SimpleITK resampling.
    """
    rotation = _xyz_rotation_matrix(rotation_degrees)
    return _forward_to_sitk_resampling(translation, rotation)


def eclipse_to_sitk(
    translation: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert Eclipse Registration Technical-tab values to SimpleITK parameters.

    ``translation`` and ``rotation_degrees`` are copied exactly as displayed in
    Eclipse Registration Properties -> Technical and describe the forward
    Source FOR -> Registered FOR registration.

    Eclipse's displayed Orientation matrix is reconstructed as the transpose of
    ``Rz @ Ry @ Rx`` from its displayed Rotation X/Y/Z values.  The complete
    forward rigid transform is then inverted and decomposed into the Euler order
    expected by SimpleITK's ``Euler3DTransform``.

    Returns ``(translation_mm, rotation_degrees)`` suitable for passing directly
    to :meth:`ManualRigidTransform.build`.
    """
    eclipse_orientation = _transpose3(_xyz_rotation_matrix(rotation_degrees))
    return _forward_to_sitk_resampling(translation, eclipse_orientation)


class ManualRigidTransform:
    """Create a 3-D SimpleITK rigid transform from already-converted values.

    Coordinate-system conversion deliberately lives in :func:`dicom_to_sitk`
    and :func:`eclipse_to_sitk`.  This class only constructs the transform.
    """

    @staticmethod
    def build(
        *,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
        center: tuple[float, float, float] | None = None,
    ) -> sitk.Transform:
        """Create a 3-D Euler rigid transform from SimpleITK parameters."""
        tx, ty, tz = _xyz(translation, "translation")
        rx, ry, rz = _xyz(rotation_degrees, "rotation_degrees")

        transform = sitk.Euler3DTransform()
        # Be explicit: ITK/SimpleITK's default Euler3D convention is
        # ComputeZYX=False, corresponding to Rz @ Rx @ Ry.
        transform.SetComputeZYX(False)

        if center is not None:
            transform.SetCenter(_xyz(center, "center"))

        transform.SetRotation(
            math.radians(rx),
            math.radians(ry),
            math.radians(rz),
        )
        transform.SetTranslation((tx, ty, tz))
        return transform

    @staticmethod
    def homogeneous_matrix(
        translation: tuple[float, float, float],
        rotation_degrees: tuple[float, float, float],
    ) -> list[list[float]]:
        """Return the 4x4 SimpleITK resampling matrix for UI preview/tests."""
        tx, ty, tz = _xyz(translation, "translation")
        rotation = _sitk_rotation_matrix(rotation_degrees)
        return [
            [rotation[0][0], rotation[0][1], rotation[0][2], tx],
            [rotation[1][0], rotation[1][1], rotation[1][2], ty],
            [rotation[2][0], rotation[2][1], rotation[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @staticmethod
    def format_matrix(matrix: list[list[float]], decimals: int = 6) -> str:
        return "\n".join(
            "  ".join(f"{float(value): .{decimals}f}" for value in row)
            for row in matrix
        )

