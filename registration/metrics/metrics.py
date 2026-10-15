from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk


class RegistrationMetrics:
    @staticmethod
    def calculate_dsc(mask1: sitk.Image, mask2: sitk.Image) -> float:
        """Compute the Dice similarity coefficient between two masks."""
        RegistrationMetrics._validate_image_pair(mask1, mask2)
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(sitk.Cast(mask1 > 0, sitk.sitkUInt8), sitk.Cast(mask2 > 0, sitk.sitkUInt8))
        return float(overlap.GetDiceCoefficient())

    @staticmethod
    def calculate_jaccard(mask1: sitk.Image, mask2: sitk.Image) -> float:
        """Compute the Jaccard index between two masks."""
        RegistrationMetrics._validate_image_pair(mask1, mask2)
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(sitk.Cast(mask1 > 0, sitk.sitkUInt8), sitk.Cast(mask2 > 0, sitk.sitkUInt8))
        return float(overlap.GetJaccardCoefficient())

    @staticmethod
    def calculate_hd(mask1: sitk.Image, mask2: sitk.Image, percentile: float = 95.0) -> float:
        """Compute the percentile Hausdorff distance between two masks."""
        d12 = RegistrationMetrics._surface_distances(mask1, mask2)
        d21 = RegistrationMetrics._surface_distances(mask2, mask1)

        if d12.size == 0 and d21.size == 0:
            return 0.0

        all_distances = np.concatenate([d12, d21])
        return float(np.percentile(all_distances, percentile))

    @staticmethod
    def calculate_mda(mask1: sitk.Image, mask2: sitk.Image) -> float:
        """Compute the mean distance to agreement between two masks."""
        RegistrationMetrics._validate_image_pair(mask1, mask2)

        d12 = RegistrationMetrics._surface_distances(mask1, mask2)
        d21 = RegistrationMetrics._surface_distances(mask2, mask1)

        if d12.size == 0 and d21.size == 0:
            return 0.0

        return float(np.mean(np.concatenate([d12, d21])))

    @staticmethod
    def _surface_distances(mask_a: sitk.Image, mask_b: sitk.Image) -> np.ndarray:
        """Return distances from surface voxels of mask_a to mask_b."""
        RegistrationMetrics._validate_image_pair(mask_a, mask_b)

        surface_a = sitk.LabelContour(sitk.Cast(mask_a > 0, sitk.sitkUInt8)) > 0
        distance_b = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                sitk.Cast(mask_b > 0, sitk.sitkUInt8),
                insideIsPositive=False,
                squaredDistance=False,
                useImageSpacing=True,
            )
        )

        distance_b_arr = sitk.GetArrayViewFromImage(distance_b)
        surface_a_arr = sitk.GetArrayViewFromImage(surface_a) > 0
        distances = distance_b_arr[surface_a_arr]
        return np.asarray(distances, dtype=np.float64)

    @staticmethod
    def calculate_contour_metrics(contour1: sitk.Image, contour2: sitk.Image) -> dict[str, float]:
        """Calculate the standard contour metrics for a contour pair."""
        RegistrationMetrics._validate_image_pair(contour1, contour2)

        return {
            "dsc": RegistrationMetrics.calculate_dsc(contour1, contour2),
            "mda": RegistrationMetrics.calculate_mda(contour1, contour2),
            "jaccard": RegistrationMetrics.calculate_jaccard(contour1, contour2),
            "hd95": RegistrationMetrics.calculate_hd(contour1, contour2),
        }

    @staticmethod
    def calculate_contour_metrics_table(
        fixed_contours: dict[str, sitk.Image],
        rigid_contours: dict[str, sitk.Image],
        final_contours: dict[str, sitk.Image],
        contours_to_calculate: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a wide contour comparison table for rigid and final registration."""
        if contours_to_calculate is None:
            contours_to_calculate = sorted(set(fixed_contours.keys()) & set(rigid_contours.keys()) & set(final_contours.keys()))

        rows: list[dict[str, Any]] = []
        for contour_name in contours_to_calculate:
            if contour_name not in fixed_contours or contour_name not in rigid_contours or contour_name not in final_contours:
                continue

            rigid_metrics = RegistrationMetrics.calculate_contour_metrics(fixed_contours[contour_name], rigid_contours[contour_name])
            final_metrics = RegistrationMetrics.calculate_contour_metrics(fixed_contours[contour_name], final_contours[contour_name])

            for metric_name, rigid_value, final_value in (
                ("Dice", rigid_metrics["dsc"], final_metrics["dsc"]),
                ("Jaccard", rigid_metrics["jaccard"], final_metrics["jaccard"]),
                ("Hausdorff Distance 95", rigid_metrics["hd95"], final_metrics["hd95"]),
                ("Mean Distance To Agreement", rigid_metrics["mda"], final_metrics["mda"]),
            ):
                rows.append(
                    {
                        "Region": contour_name,
                        "Metric": metric_name,
                        "Rigid Registration": rigid_value,
                        "Final Registration": final_value,
                    }
                )

        return pd.DataFrame(rows, columns=["Region", "Metric", "Rigid Registration", "Final Registration"])

    @staticmethod
    def calculate_mse(image1: sitk.Image, image2: sitk.Image) -> float:
        """Mean-squares metric evaluation."""
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetInterpolator(sitk.sitkLinear)
        return float(reg.MetricEvaluate(sitk.Cast(image1, sitk.sitkFloat32), sitk.Cast(image2, sitk.sitkFloat32)))

    @staticmethod
    def calculate_mi(image1: sitk.Image, image2: sitk.Image, bins: int = 50) -> float:
        """Mattes mutual information metric evaluation."""
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(bins))
        reg.SetInterpolator(sitk.sitkLinear)
        return float(reg.MetricEvaluate(sitk.Cast(image1, sitk.sitkFloat32), sitk.Cast(image2, sitk.sitkFloat32)))

    @staticmethod
    def calculate_correlation(image1: sitk.Image, image2: sitk.Image) -> float:
        """Correlation metric evaluation."""
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetInterpolator(sitk.sitkLinear)
        return float(reg.MetricEvaluate(sitk.Cast(image1, sitk.sitkFloat32), sitk.Cast(image2, sitk.sitkFloat32)))

    @staticmethod
    def calculate_image_disagreement(
        fixed_image: np.ndarray|sitk.Image,
        moving_image: np.ndarray|sitk.Image,
        fixed_body_mask: sitk.Image | None = None,
        threshold_hu: float = 150.0,
        registered_moving_body_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        """
        Generate a labelled disagreement map for two registered CT images.

        The images must be NumPy arrays in the same fixed-image space.
        Differences below ``threshold_hu`` are treated as agreement.

        Labels in the returned array are:

        - ``0``: agreement or outside the comparison ROI
        - ``1``: registered moving image is brighter than the fixed image
        - ``2``: fixed image is brighter than the registered moving image

        Parameters
        ----------
        fixed_image:
            Fixed CT image as a NumPy array.

        registered_moving_image:
            Moving CT image after registration and resampling into fixed-image
            space.

        fixed_body_mask:
            Optional SimpleITK body mask for the fixed CT. When both this and
            ``registered_moving_body_mask`` are supplied, the disagreement ROI
            is their union so anatomy present in either patient outline is kept.
            It must have the same dimensions as ``fixed_image``.

        threshold_hu:
            Minimum absolute intensity difference, in Hounsfield units, needed
            to classify a voxel as disagreement.

        registered_moving_body_mask:
            Optional moving-patient body mask already resampled into fixed-image
            space. When supplied with ``fixed_body_mask``, the two masks are
            combined with a logical OR before disagreement is calculated.

        Returns
        -------
        numpy.ndarray
            A ``uint8`` label map with values 0, 1, or 2.

        Raises
        ------
        ValueError
            If the image shapes differ, the mask dimensions do not match, or
            ``threshold_hu`` is negative.
        """

        #cast to nuumpy arrays if sitk images are passed in
        if isinstance(fixed_image, sitk.Image):
            fixed_image = sitk.GetArrayFromImage(fixed_image)

        if isinstance(moving_image, sitk.Image):
            moving_image = sitk.GetArrayFromImage(moving_image)

        fixed_array = np.asarray(fixed_image)
        moving_array = np.asarray(moving_image)

        if fixed_array.shape != moving_array.shape:
            raise ValueError(
                "fixed_image and registered_moving_image must have the same shape."
            )

        if threshold_hu < 0:
            raise ValueError("threshold_hu must be non-negative.")

        expected_size = tuple(reversed(fixed_array.shape))

        def mask_array(mask: sitk.Image | None, name: str) -> np.ndarray | None:
            if mask is None:
                return None
            if mask.GetSize() != expected_size:
                raise ValueError(
                    f"{name} must have the same dimensions as fixed_image."
                )
            return sitk.GetArrayFromImage(mask) > 0

        fixed_mask_array = mask_array(fixed_body_mask, "fixed_body_mask")
        moving_mask_array = mask_array(
            registered_moving_body_mask,
            "registered_moving_body_mask",
        )

        if fixed_mask_array is None and moving_mask_array is None:
            body_mask = np.ones(fixed_array.shape, dtype=bool)
        elif fixed_mask_array is None:
            body_mask = moving_mask_array
        elif moving_mask_array is None:
            body_mask = fixed_mask_array
        else:
            # Use the union of both patient outlines.  This is important when
            # registration leaves moving anatomy outside the fixed body (or vice
            # versa); those voxels are genuine disagreement and must not be masked
            # away simply because they are outside one dataset's BODY contour.
            body_mask = fixed_mask_array | moving_mask_array

        # Positive values indicate that the registered moving image is brighter.
        difference = (
            moving_array.astype(np.float32)
            - fixed_array.astype(np.float32)
        )

        disagreement = np.zeros(fixed_array.shape, dtype=np.uint8)

        # Label 1: moving image is brighter than the fixed image.
        disagreement[
            (difference > threshold_hu) & body_mask
        ] = 1

        # Label 2: fixed image is brighter than the moving image.
        disagreement[
            (difference < -threshold_hu) & body_mask
        ] = 2

        return disagreement

    @staticmethod
    def calculate_image_metrics(image1: sitk.Image, image2: sitk.Image) -> pd.DataFrame:
        """Return a one-row dataframe with standard image similarity metrics."""
        return pd.DataFrame(
            [
                {
                    "region": "image",
                    "mean_squares_error": RegistrationMetrics.calculate_mse(image1, image2),
                    "mutual_information": RegistrationMetrics.calculate_mi(image1, image2),
                    "correlation": RegistrationMetrics.calculate_correlation(image1, image2),
                }
            ]
        )

    @staticmethod
    def calculate_image_metrics_table(
        fixed_image: sitk.Image,
        rigid_image: sitk.Image,
        final_image: sitk.Image,
    ) -> pd.DataFrame:
        """Return a comparison table for rigid and final registration image metrics."""
        rows = [
            {
                "Metric": "Mean Squares Error",
                "Rigid Registration": RegistrationMetrics.calculate_mse(fixed_image, rigid_image),
                "Final Registration": RegistrationMetrics.calculate_mse(fixed_image, final_image),
            },
            {
                "Metric": "Mutual Information",
                "Rigid Registration": RegistrationMetrics.calculate_mi(fixed_image, rigid_image),
                "Final Registration": RegistrationMetrics.calculate_mi(fixed_image, final_image),
            },
            {
                "Metric": "Correlation",
                "Rigid Registration": RegistrationMetrics.calculate_correlation(fixed_image, rigid_image),
                "Final Registration": RegistrationMetrics.calculate_correlation(fixed_image, final_image),
            },
        ]
        return pd.DataFrame(rows, columns=["Metric", "Rigid Registration", "Final Registration"])

    @staticmethod
    def calculate_jacobian(transform: sitk.Transform, reference_image: sitk.Image) -> tuple[float, float, float, float]:
        """Calculate jacobian statistics for a transform on the given reference image."""
        disp = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            reference_image.GetSize(),
            reference_image.GetOrigin(),
            reference_image.GetSpacing(),
            reference_image.GetDirection(),
        )

        jac = sitk.DisplacementFieldJacobianDeterminant(disp)
        jac_arr = sitk.GetArrayFromImage(jac)

        mean_jacobian = float(jac_arr.mean())
        std_jacobian = float(jac_arr.std())
        min_jacobian = float(jac_arr.min())
        percent_violations = float(np.mean(jac_arr <= 0) * 100.0)

        return mean_jacobian, std_jacobian, min_jacobian, percent_violations

    @staticmethod
    def calculate_all_metrics(
        image1: sitk.Image,
        image2: sitk.Image,
        contours1: dict[str, sitk.Image],
        contours2: dict[str, sitk.Image],
        transform: sitk.Transform,
        transform_name: str,
        contours_to_calculate: list[str] | None = None,
        reference_image: sitk.Image | None = None,
    ) -> pd.DataFrame:
        """Return a merged metric table for image, contour, and jacobian metrics."""
        if reference_image is None:
            reference_image = image1

        image_metrics = RegistrationMetrics.calculate_image_metrics(image1, image2)
        contour_metrics = RegistrationMetrics.calculate_contour_metrics(contours1, contours2, contours_to_calculate)
        jacobian_metrics = RegistrationMetrics.calculate_jacobian(transform, reference_image)
        jacobian_frame = pd.DataFrame(
            [
                {
                    "region": "transform",
                    "jacobian_mean": jacobian_metrics[0],
                    "jacobian_std": jacobian_metrics[1],
                    "jacobian_min": jacobian_metrics[2],
                    "jacobian_percent_violations": jacobian_metrics[3],
                }
            ]
        )

        image_metrics = image_metrics.melt(id_vars="region", var_name="metric", value_name=transform_name)
        contour_metrics = contour_metrics.melt(id_vars="region", var_name="metric", value_name=transform_name)
        jacobian_frame = jacobian_frame.melt(id_vars="region", var_name="metric", value_name=transform_name)

        return pd.concat([image_metrics, contour_metrics, jacobian_frame], ignore_index=True)



    @staticmethod
    def _validate_image_pair(image1: sitk.Image, image2: sitk.Image) -> None:
        if not isinstance(image1, sitk.Image) or not isinstance(image2, sitk.Image):
            raise TypeError("Inputs must be SimpleITK images.")
        if image1.GetSize() != image2.GetSize():
            raise ValueError("Images must have the same size.")
