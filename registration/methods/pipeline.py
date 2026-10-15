from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict


STEP_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Masking": {
        "mode": "Auto",  # Auto: use body contour when available, else HU threshold
        "erosion": 0,
        "dilation": 1,
        "moving_contour": "body",
        "fixed_contour": "body",
        "lower_hu_threshold": -200,
        "upper_hu_threshold": 3000,
        "crop_based_on_mask": True,
        "crop_padding": 5,
        # Optional fixed-image SI ROI for rigid registration only. When set,
        # the fixed body mask is limited to this contour's axial extent.
        "fixed_roi_contour": None,
        "fixed_roi_inferior_margin_mm": 20.0,
        "fixed_roi_superior_margin_mm": 20.0,
    },
    "Rigid Registration": {
        "initial_transform": "Moments",
        "metric": "Correlation",
        "sampling_strategy": "None",
        "sampling_percentage": 1.0,
        "learning_rate": 1.0,
        "min_step": 1e-4,
        "relaxation_factor": 0.3,
        "gradient_tolerance": 1e-6,
        "use_multiresolution": True,
        "iterations": [200, 150, 100],
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas": [2.0, 1.0, 0.0],
        "crop_to_mask_overlap": False,
    },
    "Demons Registration": {
        "method": "Diffeomorphic",

        "use_multiresolution": True,

        "single_iterations": 250,

        "iterations": [200, 150, 100],
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas": [2.0, 1.0, 0.0],

        "field_smoothing_sigma": 1.0,
        "max_rms_error": 0.1,
        "hu_difference_threshold": 1.5,
        "stopping_window": 10,
        "stopping_tolerance": 1e-5,
    },
}

DEFAULT_PIPELINE = ["Masking", "Rigid Registration", "Demons Registration"]


@dataclass
class RegistrationStep:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_template(cls, name: str) -> "RegistrationStep":
        return cls(name=name, params=copy.deepcopy(STEP_TEMPLATES.get(name, {})))