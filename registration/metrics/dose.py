from __future__ import annotations

from typing import Any
import re

import pandas as pd
import SimpleITK as sitk
import numpy as np

from registration.methods.resample import sum_images


def _check_same_geometry(a: sitk.Image, b: sitk.Image) -> None:
    if (
        a.GetSize() != b.GetSize()
        or a.GetSpacing() != b.GetSpacing()
        or a.GetOrigin() != b.GetOrigin()
        or a.GetDirection() != b.GetDirection()
    ):
        raise ValueError("Dose image and mask must have matching geometry.")


class DoseConversion:

    @staticmethod
    def physical_dose_to_bed(
        physical_dose: float | np.ndarray,
        alpha_beta: float | int,
        number_of_fractions: float | int,
    ) -> float | np.ndarray:
        """
        Convert physical dose to biological effective dose (BED).

        BED = D × (1 + d / (α/β))

        where:
            D   = total physical dose (Gy)
            d   = dose per fraction = D / n
            n   = number of fractions
            α/β = alpta-beta ration (Gy)

        Args:
            physical_dose (float): The physical dose (Gy) delivered to the region.
            alpha_beta (float): The alpha-beta ratio (Gy) of the region the dose is delivered to.
            number_of_fractions (float | int): The total number of fractions the physical dose is delivered over.
        """

        # validate inputs
        if alpha_beta <= 0:
            raise ValueError("The alpha-beta ratio (var: alpha_beta) must be positive.")
        if number_of_fractions <= 0:
            raise ValueError("The total number of frations (var: number_of_fractions) must be positive.")

        # calculate bed
        fraction_dose = physical_dose / number_of_fractions
        return physical_dose * (1.0 + fraction_dose / alpha_beta)

    @staticmethod
    def physical_dose_to_eqd2(
        physical_dose: float | np.ndarray,
        alpha_beta: float | int,
        number_of_fractions: float | int,
    ) -> float | np.ndarray:
        """
        Convert physical dose to equivalent dose in 2 Gy fractions (EQD2).

        EQD2 = D × (d + α/β) / (2 + α/β)

        where:
            D = total physical dose (Gy)
            d = D / n (dose per fraction) (Gy)
            n = number of fractions
            α/β = alpha-beta ratio (Gy)

        Args:
            physical_dose: The total physical dose (Gy) delivered to the region.
            alpha_beta: The alpha-beta ratio (Gy) of the region the dose is delivered to.
            number_of_fractions: The total number of fractions over which the physical dose is delivered.
        """

        # validate inputs
        if alpha_beta <= 0:
            raise ValueError("The alpha-beta ratio (var: alpha_beta) must be positive.")
        if number_of_fractions <= 0:
            raise ValueError("The total number of frations (var: number_of_fractions) must be positive.")
        
        # calculate eqd2
        fraction_dose = np.asarray(physical_dose) / number_of_fractions
        return np.asarray(physical_dose) * (fraction_dose + alpha_beta) / (2.0 + alpha_beta)
    
    @staticmethod
    def dose_image_to_bed(
        dose_image: sitk.Image,
        alpha_beta: float | int,
        number_of_fractions: float | int,
        dose_scaling_factor: float = 1.0,
    ) -> sitk.Image:
        """
        Convert a SimpleITK image from physical dose to biological effective dose (BED).

        Args:
            dose_image (sitk.Image): The distribution of the delivered dose.
            alpha_beta (float | int): The alpha-beta ratio (Gy) of the region the dose is delivered to.
            number_of_fractions (float | int): The total number of fractions over which the physical dose is delivered.
            dose_scaling_factor (float, optional): Factor to scale the dose distribution by. Defaults to 1.0.

        Returns:
            sitk.Image: SimpleITK image of the BED.
        """
        
        # convert to NumPy array and calculate BED
        physical_dose_array = DoseConversion.dose_image_to_array(dose_image, dose_scaling_factor)
        bed_array = DoseConversion.physical_dose_to_bed(
            physical_dose_array, alpha_beta, number_of_fractions
        )
        
        # convert back to SimpleITK.Image and return
        return DoseConversion.array_to_dose_image(bed_array, dose_image)

    @staticmethod
    def dose_image_to_eqd2(
        dose_image: sitk.Image,
        alpha_beta: float | int,
        number_of_fractions: float | int,
        dose_scaling_factor: float = 1.0,
    ) -> sitk.Image:
        """
        Convert a SimpleITK image from physical dose to equivalent dose in 2 Gy fractions (EQD2).

        Args:
            dose_image (sitk.Image): The distribution of the delivered dose.
            alpha_beta (float | int): The alpha-beta ratio (Gy) of the region the dose is delivered to.
            number_of_fractions (float | int): The total number of fractions over which the physical dose is delivered.
            dose_scaling_factor (float, optional): Factor to scale the dose distribution by. Defaults to 1.0.

        Returns:
            sitk.Image: SimpleITK image of the EQD2.
        """
        
        # convert to NumPy array and calculate BED
        physical_dose_array = DoseConversion.dose_image_to_array(dose_image, dose_scaling_factor)
        eqd2_array = DoseConversion.physical_dose_to_eqd2(
            physical_dose_array, alpha_beta, number_of_fractions
        )
        
        # convert back to SimpleITK.Image and return
        return DoseConversion.array_to_dose_image(eqd2_array, dose_image)

    @staticmethod
    def dose_image_to_array(
        dose_image: sitk.Image,
        dose_scaling_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Convert a SimpleITK dose image to a NumPy array (optionally) scaled by the dose scaling factor.
        """
        
        # convert to NumPy array with float type
        return sitk.GetArrayFromImage(dose_image).astype(np.float32) * float(dose_scaling_factor)

    @staticmethod
    def array_to_dose_image(
        dose_array: np.ndarray,
        reference_image: sitk.Image | None = None,
    ) -> sitk.Image:
        """
        Convert a NumPy dose array back to a SimpleITK image and (optionally) copy spatial metadata.
        """
        
        # convert to sitk.Image
        dose_image = sitk.GetImageFromArray(dose_array)

        # copy metadata (origin, spacing, direction, etc.) if reference image passed
        if reference_image is not None:
            dose_image.CopyInformation(reference_image)
        
        return dose_image






class DVH:
    """Class to store functions for DVH metric calculations and plot data generation."""
    
    # dose metrics
    
    @staticmethod
    def dose_cc(
        dose_image: sitk.Image,
        volume_cc: float,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Dcc.

        Returns the minimum dose (Gy) received by the hottest `volume_cc` of the
        dose distribution or masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            volume_cc (float): The volume (cc) for which to calculate Dcc.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Dcc value (Gy).
        """

        if volume_cc <= 0:
            raise ValueError("volume_cc must be positive.")

        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)

        # determine how many voxels correspond to the requested volume
        voxel_volume_cc = np.prod(dose_image.GetSpacing()) / 1000.0
        n_voxels = int(np.ceil(volume_cc / voxel_volume_cc))

        # if the requested volume exceeds the roi volume, Dcc is the minimum dose
        if n_voxels >= dose_values.size:
            return float(dose_values.min())

        # find the minimum dose within the hottest volume_cc
        k = dose_values.size - n_voxels
        return float(np.partition(dose_values, k)[k])

    @staticmethod
    def dose_percent(
        dose_image: sitk.Image,
        volume_percent: float,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Dx%.

        Returns the dose (Gy) received by the hottest `volume_percent` of the dose 
        distribution or masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            volume_percent (float): The volume percentage for which to calculate Dx%.
                For example, 95 returns D95%.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Dx% value (Gy).
        """

        if not (0.0 <= volume_percent <= 100.0):
            raise ValueError("volume_percent must be between 0 and 100.")

        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)

        # Dx% is the (100 - x)th percentile of the dose distribution
        return float(np.percentile(dose_values, 100.0 - volume_percent))

    @staticmethod
    def dose_mean(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Dmean.

        Returns the mean dose (Gy) received by the dose distribution or masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Dmean value (Gy).
        """
        
        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)
        return float(np.mean(dose_values))

    @staticmethod
    def dose_median(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate D50%.

        Returns the dose (Gy) received by the median of the dose distribution or 
        masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The D50% value (Gy).
        """
        
        return DVH.dose_percent(
            dose_image,
            volume_percent=50.0,
            mask_image=mask_image,
            dose_scaling_factor=dose_scaling_factor,
        )

    @staticmethod
    def dose_min(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Dmin.

        Returns the minimum dose (Gy) received by the dose distribution or masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Dmin value (Gy).
        """
        
        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)
        return float(np.min(dose_values))

    @staticmethod
    def dose_max(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Dmax.

        Returns the maximum dose (Gy) received by the dose distribution or masked region.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Dmax value (Gy).
        """
        
        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)
        return float(np.max(dose_values))

    # volume metrics

    @staticmethod
    def volume_cc(
        dose_image: sitk.Image,
        dose_threshold: float,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Vx (cc).

        Returns the volume (cc) receiving at least `dose_threshold` Gy.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            dose_threshold (float): The dose threshold (Gy) for which to calculate Vx.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Vx value (cc).
        """

        if dose_threshold < 0:
            raise ValueError("dose_threshold must be non-negative.")

        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)

        # calculate the volume receiving at least the threshold dose
        voxel_volume_cc = np.prod(dose_image.GetSpacing()) / 1000.0
        n_voxels = np.count_nonzero(dose_values >= float(dose_threshold))
        return float(n_voxels * voxel_volume_cc)

    @staticmethod
    def volume_percent(
        dose_image: sitk.Image,
        dose_threshold: float,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> float:
        """
        Calculate Vx (%).

        Returns the volume (%) receiving at least `dose_threshold` Gy.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            dose_threshold (float): The dose threshold (Gy) for which to calculate Vx.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.

        Returns:
            float: The Vx value (%).
        """

        if dose_threshold < 0:
            raise ValueError("dose_threshold must be non-negative.")

        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)

        # calculate the volume receiving at least the threshold dose
        fraction = np.mean(dose_values >= dose_threshold)
        return float(100.0 * fraction)

    # data for plotting

    @staticmethod
    def cumulative_dvh(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
        num_bins: int = 200,
        normalise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate cumulative DVH data.

        Returns dose thresholds (Gy) and the cumulative volume receiving at least
        each threshold. By default the volume is normalized to percent of ROI volume.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.
            num_bins (int, optional): Number of dose bins to use. Defaults to 200.
            normalise (bool, optional): If True, return volume in percent of ROI.
                If False, return volume in cc.

        Returns:
            tuple[np.ndarray, np.ndarray]: (dose_axis, cumulative_volume)
        """

        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")

        dose_values = DVH._get_masked_dose_values(dose_image, mask_image, dose_scaling_factor)
        voxel_volume_cc = np.prod(dose_image.GetSpacing()) / 1000.0

        max_dose = float(np.max(dose_values))

        # handle if every voxel has zero dose
        if np.isclose(max_dose, 0.0):
            dose_axis = np.array([0.0], dtype=np.float64)
            if normalise:
                volume_axis = np.array([100.0], dtype=np.float64)
            else:
                volume_axis = np.array([dose_values.size * voxel_volume_cc], dtype=np.float64)
            return dose_axis, volume_axis

        # bin dose values over requested dose range
        dose_edges = np.linspace(0.0, max_dose, num_bins + 1, dtype=np.float64)
        hist, _ = np.histogram(dose_values, bins=dose_edges)

        # convert histogram into cumulative dvh
        cumulative_counts = np.cumsum(hist[::-1])[::-1].astype(np.float64)
        dose_axis = dose_edges[:-1]

        if normalise:
            volume_axis = 100.0 * cumulative_counts / dose_values.size
        else:
            volume_axis = cumulative_counts * voxel_volume_cc

        return dose_axis, volume_axis

    @staticmethod
    def differential_dvh(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
        num_bins: int = 200,
        normalise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate differential DVH data.

        Returns dose bins (Gy) and the volume contained within each dose bin.
        By default the volume is normalised to the percentage of the ROI.

        Args:
            dose_image (sitk.Image): The delivered dose distribution.
            mask_image (sitk.Image | None, optional): Optional binary mask defining
                the region of interest. If omitted, the entire dose image is used.
            dose_scaling_factor (float, optional): Factor used to convert the stored
                dose values to Gy. Defaults to 1.0.
            num_bins (int, optional): Number of dose bins. Defaults to 200.
            normalise (bool, optional): If True, return volume in percent of the ROI.
                If False, return volume in cc.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Dose bin centres (Gy) and differential volume.
        """

        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")

        # Extract the dose values within the ROI.
        dose_values = DVH._get_masked_dose_values(
            dose_image,
            mask_image,
            dose_scaling_factor,
        )

        voxel_volume_cc = np.prod(dose_image.GetSpacing()) / 1000.0

        max_dose = float(np.max(dose_values))

        # bin dose values
        dose_edges = np.linspace(0.0, max_dose, num_bins + 1)
        counts, _ = np.histogram(dose_values, bins=dose_edges)

        # get the centre of each dose bin
        dose_axis = 0.5 * (dose_edges[:-1] + dose_edges[1:])

        if normalise:
            volume_axis = 100.0 * counts / dose_values.size
        else:
            volume_axis = counts * voxel_volume_cc

        return dose_axis, volume_axis

    # helpers

    @staticmethod
    def _get_masked_dose_values(
        dose_image: sitk.Image,
        mask_image: sitk.Image | None = None,
        dose_scaling_factor: float = 1.0,
    ) -> np.ndarray:
        """Return the dose values within the ROI as a 1D NumPy array."""

        dose_array = DoseConversion.dose_image_to_array(dose_image, dose_scaling_factor,)

        # restrict the dose values to the roi if a mask is provided
        if mask_image is not None:
            _check_same_geometry(dose_image, mask_image)
            mask = sitk.GetArrayFromImage(mask_image) > 0
            dose_values = dose_array[mask]
        else:
            dose_values = dose_array.ravel()

        if dose_values.size == 0:
            raise ValueError("The region of interest contains no voxels.")

        return dose_values



def convert_dose_image_for_display(
    dose_image: sitk.Image,
    dose_type: str,
    alpha_beta: float,
    number_of_fractions: float,
    dose_scaling_factor: float = 1.0,
) -> sitk.Image:
    """Return a dose image converted to the requested display scale."""

    normalized = str(dose_type).strip().lower()
    if normalized in {"physical", "dose", "raw"}:
        dose = DoseConversion.dose_image_to_array(dose_image, dose_scaling_factor)
        return DoseConversion.array_to_dose_image(dose, dose_image)
    if normalized == "bed":
        return DoseConversion.dose_image_to_bed(
            dose_image,
            alpha_beta=alpha_beta,
            number_of_fractions=number_of_fractions,
            dose_scaling_factor=dose_scaling_factor,
        )
    if normalized == "eqd2":
        return DoseConversion.dose_image_to_eqd2(
            dose_image,
            alpha_beta=alpha_beta,
            number_of_fractions=number_of_fractions,
            dose_scaling_factor=dose_scaling_factor,
        )
    raise ValueError(f"Unsupported dose type: {dose_type!r}")


def _metric_result_for_name(
    dose_image: sitk.Image,
    metric_name: str,
    mask_image: sitk.Image | None = None,
    dose_scaling_factor: float = 1.0,
) -> tuple[float, str]:
    normalized = str(metric_name).strip().lower()
    if normalized == "dmax":
        return (
            DVH.dose_max(dose_image, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "Gy",
        )
    if normalized == "d50":
        return (
            DVH.dose_median(dose_image, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "Gy",
        )
    if normalized == "d1cc":
        return (
            DVH.dose_cc(dose_image, volume_cc=1.0, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "Gy",
        )
    if normalized.startswith("d") and normalized.endswith("cc"):
        try:
            volume_cc = float(normalized[1:-2])
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported DVH metric: {metric_name!r}") from exc
        return (
            DVH.dose_cc(dose_image, volume_cc=volume_cc, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "Gy",
        )
    if normalized.startswith("d") and normalized.endswith("%"):
        try:
            volume_percent = float(normalized[1:-1])
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported DVH metric: {metric_name!r}") from exc
        return (
            DVH.dose_percent(dose_image, volume_percent=volume_percent, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "Gy",
        )
    match = re.fullmatch(r"v\s*(\d+(?:\.\d+)?)\s*(gy|%)?", normalized)
    if match is not None:
        threshold = float(match.group(1))
        suffix = match.group(2) or "gy"
        if suffix == "%":
            return (
                DVH.volume_percent(dose_image, dose_threshold=threshold, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
                "%",
            )
        return (
            DVH.volume_percent(dose_image, dose_threshold=threshold, mask_image=mask_image, dose_scaling_factor=dose_scaling_factor),
            "%",
        )
    raise ValueError(f"Unsupported DVH metric: {metric_name!r}")


def metric_result_for_name(
    dose_image: sitk.Image,
    metric_name: str,
    mask_image: sitk.Image | None = None,
    dose_scaling_factor: float = 1.0,
) -> tuple[float, str]:
    """Return a single DVH metric value and unit by name."""
    return _metric_result_for_name(
        dose_image,
        metric_name,
        mask_image=mask_image,
        dose_scaling_factor=dose_scaling_factor,
    )


def metric_value_for_name(
    dose_image: sitk.Image,
    metric_name: str,
    mask_image: sitk.Image | None = None,
    dose_scaling_factor: float = 1.0,
) -> float:
    """Return a single DVH metric value by name."""
    return metric_result_for_name(
        dose_image,
        metric_name,
        mask_image=mask_image,
        dose_scaling_factor=dose_scaling_factor,
    )[0]


def combine_dose_images_for_display(
    fixed_dose: sitk.Image | None,
    moving_dose: sitk.Image | None,
    dose_type: str,
    fixed_alpha_beta: float,
    moving_alpha_beta: float,
    fixed_number_of_fractions: float,
    moving_number_of_fractions: float,
    dose_scaling_factor: float = 1.0,
) -> sitk.Image | None:
    """Convert fixed and moving dose components independently and sum them."""
    converted_images: list[sitk.Image] = []
    if fixed_dose is not None:
        converted_images.append(
            convert_dose_image_for_display(
                fixed_dose,
                dose_type=dose_type,
                alpha_beta=fixed_alpha_beta,
                number_of_fractions=fixed_number_of_fractions,
                dose_scaling_factor=dose_scaling_factor,
            )
        )
    if moving_dose is not None:
        converted_images.append(
            convert_dose_image_for_display(
                moving_dose,
                dose_type=dose_type,
                alpha_beta=moving_alpha_beta,
                number_of_fractions=moving_number_of_fractions,
                dose_scaling_factor=dose_scaling_factor,
            )
        )
    return sum_images(*converted_images)


def build_dvh_metric_row(
    region: str,
    dose_image: sitk.Image,
    mask_image: sitk.Image,
    metrics: list[str],
    alpha_beta: float,
    number_of_fractions: float,
    dose_type: str = "physical",
    dose_scaling_factor: float = 1.0,
) -> dict[str, Any]:
    converted = convert_dose_image_for_display(
        dose_image,
        dose_type=dose_type,
        alpha_beta=alpha_beta,
        number_of_fractions=number_of_fractions,
        dose_scaling_factor=dose_scaling_factor,
    )

    row: dict[str, Any] = {
        "Region": region,
        "Alpha/Beta": float(alpha_beta),
        "Fractions": float(number_of_fractions),
        "Dose Type": str(dose_type).title(),
    }
    for metric_name in metrics:
        row[str(metric_name)] = _metric_result_for_name(
            converted,
            metric_name,
            mask_image=mask_image,
            dose_scaling_factor=1.0,
        )[0]
    return row


def build_dvh_metrics_dataframe(
    dose_image: sitk.Image,
    masks_and_settings: list[dict[str, Any]],
    dose_type: str = "physical",
    dose_scaling_factor: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_metrics: list[str] = []

    for item in masks_and_settings:
        metrics = [str(metric) for metric in item.get("metrics", [])]
        for metric in metrics:
            if metric not in all_metrics:
                all_metrics.append(metric)

    for item in masks_and_settings:
        region = str(item.get("region", "")).strip()
        mask = item.get("mask")
        if not region or mask is None:
            continue
        row = build_dvh_metric_row(
            region=region,
            dose_image=dose_image,
            mask_image=mask,
            metrics=[str(metric) for metric in item.get("metrics", [])],
            alpha_beta=float(item.get("alpha_beta", 1.0)),
            number_of_fractions=float(item.get("fractions", 1.0)),
            dose_type=dose_type,
            dose_scaling_factor=dose_scaling_factor,
        )
        rows.append(row)

    columns = ["Region", "Alpha/Beta", "Fractions", "Dose Type"] + all_metrics
    return pd.DataFrame(rows, columns=columns)


def build_dvh_curve(
    dose_image: sitk.Image,
    mask_image: sitk.Image | None,
    dose_type: str,
    alpha_beta: float,
    number_of_fractions: float,
    dose_scaling_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    converted = convert_dose_image_for_display(
        dose_image,
        dose_type=dose_type,
        alpha_beta=alpha_beta,
        number_of_fractions=number_of_fractions,
        dose_scaling_factor=dose_scaling_factor,
    )
    return DVH.cumulative_dvh(converted, mask_image=mask_image, dose_scaling_factor=1.0)
