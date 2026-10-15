from __future__ import annotations
from typing import Iterable
import SimpleITK as sitk
import copy

from registration.dataset.dataset import Dataset, Structure


class Resample:

    @staticmethod
    def _image(
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        transform: sitk.Transform | None,
        interpolation: int,
        default_value: float,
    ) -> sitk.Image:
        return sitk.Resample(
            moving_image,
            fixed_image,
            transform or sitk.Transform(),
            interpolation,
            default_value,
            moving_image.GetPixelID(),
        )

    @staticmethod
    def ct(
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        transform: sitk.Transform | None = None,
        interpolation: int = sitk.sitkLinear,
        default_value: float = -1000.0,
    ) -> sitk.Image:
        return Resample._image(
            moving_image=moving_image,
            fixed_image=fixed_image,
            transform=transform,
            interpolation=interpolation,
            default_value=default_value,
        )

    @staticmethod
    def dose(
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        transform: sitk.Transform | None = None,
        interpolation: int = sitk.sitkLinear,
        default_value: float = 0.0,
    ) -> sitk.Image:
        return Resample._image(
            moving_image=moving_image,
            fixed_image=fixed_image,
            transform=transform,
            interpolation=interpolation,
            default_value=default_value,
        )

    @staticmethod
    def mask(
        moving_image: sitk.Image,
        fixed_image: sitk.Image,
        transform: sitk.Transform | None = None,
        interpolation: int = sitk.sitkNearestNeighbor,
        default_value: int = 0,
    ) -> sitk.Image:
        return Resample._image(
            moving_image=moving_image,
            fixed_image=fixed_image,
            transform=transform,
            interpolation=interpolation,
            default_value=default_value,
        )

    @classmethod
    def structure_set(
        cls,
        structures: dict[str, Structure | sitk.Image],
        reference_image: sitk.Image,
        transform: sitk.Transform | None = None,
        interpolation: int = sitk.sitkNearestNeighbor,
        default_value: int = 0,
    ) -> dict[str, Structure | sitk.Image]:

        transformed: dict[str, Structure | sitk.Image] = {}

        for key, structure in structures.items():
            if isinstance(structure, Structure):
                transformed[key] = Structure(
                    id=structure.id,
                    import_name=structure.import_name,
                    display_name=structure.display_name,
                    image=cls.mask(
                        moving_image=structure.image,
                        fixed_image=reference_image,
                        transform=transform,
                        interpolation=interpolation,
                        default_value=default_value,
                    ),
                )
                continue

            transformed[key] = cls.mask(
                moving_image=structure,
                fixed_image=reference_image,
                transform=transform,
                interpolation=interpolation,
                default_value=default_value,
            )

        return transformed

    @classmethod
    def dataset(
        cls,
        structures: dict[str, Structure],
        reference_image: sitk.Image,
        transform: sitk.Transform | None = None,
        interpolation: int = sitk.sitkNearestNeighbor,
        default_value: int = 0,
    ) -> dict[str, Structure]:

        transformed: dict[str, Structure] = {}

        for key, structure in structures.items():
            transformed[key] = Structure(
                id=structure.id,
                import_name=structure.import_name,
                display_name=structure.display_name,
                image=cls.structure(
                    moving_image=structure.image,
                    fixed_image=reference_image,
                    transform=transform,
                    interpolation=interpolation,
                    default_value=default_value,
                ),
            )

        return transformed



def sum_images(*images: sitk.Image | None) -> sitk.Image | None:
    """Sum all non-null images, returning None when every input is missing."""
    valid_images = [sitk.Cast(image, sitk.sitkFloat32) for image in images if image is not None]
    if not valid_images:
        return None

    result = valid_images[0]
    for image in valid_images[1:]:
        result = sitk.Add(result, image)
    return result


def ensure_same_geometry(images: Iterable[sitk.Image]) -> None:
    """Raise if any image in the iterable does not match the first image's geometry."""
    iterator = iter(images)
    first_image = next(iterator, None)
    if first_image is None:
        return

    for image in iterator:
        if image.GetSize() != first_image.GetSize():
            raise ValueError("All images must have the same size.")
        if image.GetSpacing() != first_image.GetSpacing():
            raise ValueError("All images must have the same spacing.")
        if image.GetOrigin() != first_image.GetOrigin():
            raise ValueError("All images must have the same origin.")
        if image.GetDirection() != first_image.GetDirection():
            raise ValueError("All images must have the same direction.")
