from __future__ import annotations
import warnings

import SimpleITK as sitk
import numpy as np


class Cropping:
    """Utilities for cropping SimpleITK images and masks."""

    @staticmethod
    def crop_image_and_mask(
        image: sitk.Image,
        mask: sitk.Image,
        *,
        padding: int = 0,
    ) -> tuple[sitk.Image, sitk.Image]:
        """
        Crop an image and mask to the mask's bounding box, with optional padding.

        Args:
            image (sitk.Image): Input SimpleITK image.
            mask (sitk.Image): Binary or binary-like mask used to define the crop region.
            padding (int): Voxel padding applied symmetrically around the mask bounding box.

        Returns:
            tuple[sitk.Image, sitk.Image]: A tuple of (cropped_image, cropped_mask).
        """

        bbox = Cropping._mask_bounding_box(mask)
        if bbox is None:
            return image, mask

        index, size = Cropping._pad_bbox(
            index=bbox[0],
            size=bbox[1],
            image_size=list(image.GetSize()),
            padding=padding,
        )

        cropped_image = sitk.RegionOfInterest(image, size, index)
        cropped_mask = sitk.RegionOfInterest(mask, size, index)
        return cropped_image, cropped_mask

    @staticmethod
    def crop_body_to_contour_extent(
        body_mask: sitk.Image,
        contour_mask: sitk.Image,
        *,
        inferior_margin_mm: float = 20.0,
        superior_margin_mm: float = 20.0,
    ) -> sitk.Image:
        """
        Crop a body mask to the axial extent of a contour, with margins in millimetres.

        This keeps only the body-mask slices within a given distance inferior and superior
        to the contour mask, which is useful for limiting a body mask to anatomy near a
        target region such as a PTV.

        Args:
            body_mask (sitk.Image): Full body mask.
            contour_mask (sitk.Image): Binary contour mask that defines the slice range.
            inferior_margin_mm (float, optional): Additional axial distance in mm to keep
                below the contour's inferior extent. Defaults to 20.0.
            superior_margin_mm (float, optional): Additional axial distance in mm to keep
                above the contour's superior extent. Defaults to 20.0.

        Returns:
            sitk.Image: The cropped body mask.
        """
        if body_mask.GetDimension() < 3:
            raise ValueError("This crop assumes a 3D image with an axial dimension.")

        # In SimpleITK, axis 2 is the z axis, which corresponds to axial slices.
        axis_index = 2

        # If the contour has no occupied voxels along z, there is no useful range to crop.
        interval = Cropping._axis_interval(contour_mask, axis_index)
        if interval is None:
            return body_mask

        # Spacing is expected to be positive for a valid CT image. If the metadata is bad,
        # use a fallback so we do not divide by zero or crash during voxel conversion.
        # here by spacing we mean something akin to slice width/resolution
        spacing = float(body_mask.GetSpacing()[axis_index])
        if spacing <= 0:
            spacing = 1.0

        # Convert millimetre margins into voxel counts along the z axis.
        inferior_voxels = max(0, int(round(inferior_margin_mm / spacing)))
        superior_voxels = max(0, int(round(superior_margin_mm / spacing)))

        #enforces that you cant start before 0th slice or after final slice with the max function, then define start and end slice numbers
        start = max(0, interval[0] - inferior_voxels)
        end = min(body_mask.GetSize()[axis_index], interval[1] + superior_voxels)

        #last slice should not be less than first slice as that makes no sense, quit and just return the body mask 
        if end <= start:
            warnings.warn(
                "Contour-based crop collapsed to zero length after clamping; "
                "returning the original body mask.",
                RuntimeWarning,
                )
            return body_mask

        #create a list of length equal to the image dimension, filled with zeros
        index = [0] * body_mask.GetDimension()

        
        size = list(body_mask.GetSize())
        index[axis_index] = start
        size[axis_index] = end - start

        return sitk.RegionOfInterest(body_mask, size, index)

    @staticmethod
    def _mask_bounding_box(
        mask: sitk.Image,
    ) -> tuple[list[int], list[int]] | None:
        """Return the bounding box of the mask, or None if empty."""
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

    @staticmethod
    def _pad_bbox(
        *,
        index: list[int],
        size: list[int],
        image_size: list[int],
        padding: int,
    ) -> tuple[list[int], list[int]]:
        """Expand a bounding box by padding and clamp it to image bounds."""
        dim = len(image_size)
        pad = [padding] * dim

        new_index: list[int] = []
        new_size: list[int] = []

        for i in range(dim):
            start = max(0, index[i] - pad[i])
            end = min(image_size[i], index[i] + size[i] + pad[i])
            new_index.append(start)
            new_size.append(max(0, end - start))

        return new_index, new_size

    @staticmethod
    def crop_by_axis_overlap(
        fixed_image: sitk.Image,
        fixed_mask: sitk.Image,
        moving_image: sitk.Image,
        moving_mask: sitk.Image,
        *,
        transform: sitk.Transform | None = None,
        axis: str = "z",
        padding: int = 0,
    ) -> tuple[tuple[sitk.Image, sitk.Image], tuple[sitk.Image, sitk.Image]]:
        """
        Crop fixed and moving image/mask pairs to a conservative overlap along one axis.

        The fixed crop is exact in fixed space. The moving crop is a conservative
        axis-aligned ROI in moving space that contains the transformed overlap.
        No returned image or mask is resampled or transformed.

        Args:
            fixed_image (sitk.Image): Fixed image.
            fixed_mask (sitk.Image): Mask associated with the fixed image.
            moving_image (sitk.Image): Moving image.
            moving_mask (sitk.Image): Mask associated with the moving image.
            transform (sitk.Transform | None, optional): Transform mapping moving
                space into fixed space. If None, the identity transform is used.
            axis (str, optional): Axis to crop along. Must be "x", "y", or "z".
                Defaults to "z".
            padding (int, optional): Symmetric voxel padding applied along the
                selected axis. Defaults to 0.

        Returns:
            tuple: ((cropped_fixed_image, cropped_fixed_mask),
                    (cropped_moving_image, cropped_moving_mask))
        """
        dim = fixed_image.GetDimension()
        if moving_image.GetDimension() != dim:
            raise ValueError("fixed and moving images must have the same dimension")
        if fixed_mask.GetDimension() != dim or moving_mask.GetDimension() != dim:
            raise ValueError("all images and masks must have the same dimension")

        if transform is None:
            transform = sitk.Transform(dim, sitk.sitkIdentity)

        axis_index = Cropping._axis_index(axis, dim)

        fixed_interval = Cropping._axis_interval(fixed_mask, axis_index)
        if fixed_interval is None:
            return (fixed_image, fixed_mask), (moving_image, moving_mask)

        moving_mask_in_fixed = sitk.Resample(
            moving_mask,
            fixed_image,
            transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        moving_interval_fixed = Cropping._axis_interval(moving_mask_in_fixed, axis_index)
        if moving_interval_fixed is None:
            return (fixed_image, fixed_mask), (moving_image, moving_mask)

        overlap_fixed = Cropping._interval_intersection(
            fixed_interval,
            moving_interval_fixed,
        )
        if overlap_fixed is None:
            return (fixed_image, fixed_mask), (moving_image, moving_mask)

        overlap_slab_fixed = Cropping._axis_slab(
            reference=fixed_image,
            axis=axis_index,
            interval=overlap_fixed,
        )

        moving_slab = sitk.Resample(
            overlap_slab_fixed,
            moving_image,
            transform.GetInverse(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        moving_interval = Cropping._axis_interval(moving_slab, axis_index)
        if moving_interval is None:
            return (fixed_image, fixed_mask), (moving_image, moving_mask)

        cropped_fixed_image, cropped_fixed_mask = Cropping._crop_along_axis(
            image=fixed_image,
            mask=fixed_mask,
            axis=axis_index,
            interval=overlap_fixed,
            padding=padding,
        )

        cropped_moving_image, cropped_moving_mask = Cropping._crop_along_axis(
            image=moving_image,
            mask=moving_mask,
            axis=axis_index,
            interval=moving_interval,
            padding=padding,
        )

        return (cropped_fixed_image, cropped_fixed_mask), (cropped_moving_image, cropped_moving_mask)

    @staticmethod
    def _axis_index(axis: str, dim: int) -> int:
        """Convert x/y/z to an image axis index."""
        mapping = {"x": 0, "y": 1, "z": 2}
        axis = axis.lower().strip()

        if axis not in mapping:
            raise ValueError('axis must be one of "x", "y", or "z"')

        axis_index = mapping[axis]
        if axis_index >= dim:
            raise ValueError(f'axis "{axis}" is not valid for {dim}D images')

        return axis_index

    @staticmethod
    def _axis_interval(mask: sitk.Image, axis: int) -> tuple[int, int] | None:
        """Return the occupied [start, end) interval along one image axis."""
        binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
        array = sitk.GetArrayViewFromImage(binary)

        if not array.any():
            return None

        # NumPy axis ordering is reversed from SimpleITK image indexing.
        array_axis = mask.GetDimension() - 1 - axis
        occupied = array.any(axis=tuple(i for i in range(array.ndim) if i != array_axis))
        indices = occupied.nonzero()[0]

        if len(indices) == 0:
            return None

        return int(indices[0]), int(indices[-1] + 1)

    @staticmethod
    def _interval_intersection(
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Return the intersection of two half-open intervals."""
        start = max(left[0], right[0])
        end = min(left[1], right[1])

        if start >= end:
            return None

        return start, end

    @staticmethod
    def _axis_slab(
        *,
        reference: sitk.Image,
        axis: int,
        interval: tuple[int, int],
    ) -> sitk.Image:
        """Create a binary slab mask covering one occupied axis interval."""
        arr_shape = tuple(reversed(reference.GetSize()))
        slab = np.zeros(arr_shape, dtype=np.uint8)

        arr_axis = reference.GetDimension() - 1 - axis
        slicer = [slice(None)] * reference.GetDimension()
        slicer[arr_axis] = slice(interval[0], interval[1])
        slab[tuple(slicer)] = 1

        out = sitk.GetImageFromArray(slab)
        out.CopyInformation(reference)
        return sitk.Cast(out, sitk.sitkUInt8)

    @staticmethod
    def _crop_along_axis(
        *,
        image: sitk.Image,
        mask: sitk.Image,
        axis: int,
        interval: tuple[int, int],
        padding: int,
    ) -> tuple[sitk.Image, sitk.Image]:
        """Crop an image and mask along one axis."""
        index = [0] * image.GetDimension()
        size = list(image.GetSize())

        start, end = interval
        start = max(0, start - padding)
        end = min(size[axis], end + padding)

        index[axis] = start
        size[axis] = end - start

        return (
            sitk.RegionOfInterest(image, size, index),
            sitk.RegionOfInterest(mask, size, index),
        )