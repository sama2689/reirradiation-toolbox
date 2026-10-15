from __future__ import annotations

import SimpleITK as sitk


class Masking:
    """Utilities for generating and applying binary masks in SimpleITK."""

    @staticmethod
    def contour_mask(
        contour  : sitk.Image,
        *,
        dilation : int = 1,
        erosion  : int = 0,
    ) -> sitk.Image:
        """
        Create a cleaned binary mask from an existing contour image.
        
        The input is treated as a binary-like image, converted to a mask, and then 
        post-processed by keeping the largest connected component, filling holes, and 
        optionally applying erosion and dilation.

        Args:
            contour (sitk.Image): Binary SimpleITK image representing a contoured volume. 
            dilation (int, optional): Number of voxels to dilate the mask after cleanup.
            erosion (int, optional): Number of voxels to erode the mask before component
                selection.

        Returns:
            sitk.Image: A binary SimpleITK image containing the processed mask.
        """
        return Masking._postprocess_mask(contour, dilation = dilation, erosion = erosion)

    @staticmethod
    def threshold_mask(
        image           : sitk.Image,
        lower_threshold : float = -200,
        upper_threshold : float = 3000,
        *,
        dilation        : int = 1,
        erosion         : int = 0,
    ) -> sitk.Image:
        """
        Create a cleaned binary mask by thresholding an image.
        
        Voxels with intensities inside the inclusive threshold range are kept. 
        The resulting mask is then cleaned by keeping the largest connected component, 
        filling holes, and optionally applying erosion and dilation.

        Args:
            image (sitk.Image): Input SimpleITK image to threshold.
            lower_threshold (float, optional): Inclusive lower bound for the mask range.
            upper_threshold (float, optional): Inclusive upper bound for the mask range.
            dilation (int, optional): Number of voxels to dilate the mask after cleanup.
            erosion (int, optional): Number of voxels to erode the mask before component 
                selection.

        Returns:
            sitk.Image: A binary SimpleITK image containing the processed mask.
        """
        mask = sitk.BinaryThreshold(
            image,
            lowerThreshold = lower_threshold,
            upperThreshold = upper_threshold,
        )
        return Masking._postprocess_mask(mask, dilation = dilation, erosion = erosion)

    @staticmethod
    def apply_mask_to_image(
        image            : sitk.Image,
        mask             : sitk.Image,
        *,
        background_value : float = -1000.0,
    ) -> sitk.Image:
        """
        Apply a binary mask to an image and replace masked-out voxels. 
        
        The image is cast to Float32 before masking. 
        Voxels where the mask is zero are replaced by `background_value`.

        Args:
            image (sitk.Image): Input SimpleITK image to be masked.
            mask (sitk.Image): Binary SimpleITK mask image.
            background_value (float, optional): Value applied outside the mask.

        Returns:
            sitk.Image: The masked image as a Float32 SimpleITK image.
        """
        masked_image = sitk.Cast(image, sitk.sitkFloat32)
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        return sitk.Mask(masked_image, binary_mask, outsideValue = float(background_value))

    @staticmethod
    def _postprocess_mask(
        mask     : sitk.Image,
        *,
        dilation : int = 0,
        erosion  : int = 0,
    ) -> sitk.Image:
        """Clean and refine a binary mask."""
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

        if erosion > 0:
            mask = sitk.BinaryErode(mask, [erosion] * mask.GetDimension())

        mask = Masking._largest_mask_component(mask)
        mask = sitk.BinaryFillhole(mask)

        if dilation > 0:
            mask = sitk.BinaryDilate(mask, [dilation] * mask.GetDimension())

        return mask

    @staticmethod
    def _largest_mask_component(mask: sitk.Image) -> sitk.Image:
        """Return the largest connected component of a binary mask."""
        connected_components = sitk.ConnectedComponent(mask)
        relabelled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)
        return relabelled_components == 1

    @staticmethod
    def keep_most_overlapping_component(
        moving_mask_in_fixed_space: sitk.Image,
        reference_mask: sitk.Image,
        *,
        erode_before_cc: int = 0,
        dilate_after_cc: int = 0,
    ) -> sitk.Image:
        """
        Keep the connected component that overlaps the reference mask the most.

        This is useful after resampling a moving mask into fixed space, when the
        transform leaves leftover anatomy outside the true target region.

        Args:
            moving_mask_in_fixed_space (sitk.Image): Binary mask already resampled
                into fixed/reference space.
            reference_mask (sitk.Image): Binary reference mask in the same space.
            erode_before_cc (int, optional): Erode before connected-component
                analysis to break thin bridges. Defaults to 1.
            dilate_after_cc (int, optional): Dilate the kept component after
                selection. Defaults to 1.

        Returns:
            sitk.Image: Binary mask containing the selected connected component.
        """
        moving = sitk.Cast(moving_mask_in_fixed_space > 0, sitk.sitkUInt8)
        ref = sitk.Cast(reference_mask > 0, sitk.sitkUInt8)

        if erode_before_cc > 0:
            moving = sitk.BinaryErode(
                moving,
                [erode_before_cc] * moving.GetDimension(),
            )

        cc = sitk.ConnectedComponent(moving)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        if stats.GetNumberOfLabels() == 0:
            empty = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
            empty.CopyInformation(ref)
            return empty

        best_label = None
        best_overlap = -1

        for label in stats.GetLabels():
            component = sitk.Cast(cc == label, sitk.sitkUInt8)
            overlap = int(sitk.GetArrayViewFromImage(sitk.And(component, ref)).sum())

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        if best_label is None:
            empty = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
            empty.CopyInformation(ref)
            return empty

        kept = sitk.Cast(cc == best_label, sitk.sitkUInt8)

        if dilate_after_cc > 0:
            kept = sitk.BinaryDilate(
                kept,
                [dilate_after_cc] * kept.GetDimension(),
            )

        return sitk.Cast(kept > 0, sitk.sitkUInt8)
