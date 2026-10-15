from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid
import SimpleITK as sitk

from registration.dataset.dataset import Dataset, TransformRecord
from registration.methods.resample import Resample


class DICOMWriter:
    def export(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        reference_dataset: Dataset | None = None,
        transform_record: TransformRecord | None = None,
        overwrite: bool = False,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset.ct_image is not None and dataset.ct_metadata is not None:
            self.export_ct(
                dataset=dataset,
                output_dir=output_dir / "ct",
                reference_dataset=reference_dataset,
                transform_record=transform_record,
                overwrite=overwrite,
            )

        if dataset.dose_image is not None and dataset.dose_metadata is not None:
            self.export_dose(
                dataset=dataset,
                output_dir=output_dir / "dose",
                reference_dataset=reference_dataset,
                transform_record=transform_record,
                overwrite=overwrite,
            )

    def export_ct(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        reference_dataset: Dataset | None = None,
        transform_record: TransformRecord | None = None,
        overwrite: bool = False,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset.ct_image is None or dataset.ct_metadata is None:
            raise ValueError("CT image or CT metadata is missing.")

        if reference_dataset is None or reference_dataset.ct_metadata is None:
            raise ValueError("Reference dataset CT metadata is missing.")

        if not isinstance(dataset.ct_metadata, list):
            raise ValueError("CT metadata must be a list of per-slice DICOM headers.")

        if not isinstance(reference_dataset.ct_metadata, list):
            raise ValueError("Reference CT metadata must be a list of per-slice DICOM headers.")

        moving_slices = dataset.ct_metadata
        fixed_slices = reference_dataset.ct_metadata

        reference_image = self._reference_image(reference_dataset, dataset)

        ct_image = dataset.ct_image
        if transform_record is not None:
            ct_image = Resample.ct(
                moving_image=dataset.ct_image,
                fixed_image=reference_image,
                transform=transform_record.transform,
            )

        ct_arr = sitk.GetArrayFromImage(ct_image)  # (z, y, x)

        if len(fixed_slices) != ct_arr.shape[0]:
            raise ValueError(
                f"Number of fixed CT slices ({len(fixed_slices)}) does not match "
                f"registered CT z size ({ct_arr.shape[0]})."
            )

        new_series_uid = generate_uid()

        for z in range(ct_arr.shape[0]):
            src = moving_slices[min(z, len(moving_slices) - 1)]
            geom = fixed_slices[-1 - z]  # change to fixed_slices[z] if your slice order is not reversed

            ds = copy.deepcopy(src)

            # Convert the registered CT slice back into the source DICOM stored-value space.
            # If your ct_arr is already in stored values rather than HU, replace this block with:
            # slice_img = np.rint(ct_arr[z])
            hu = np.asarray(ct_arr[z], dtype=np.float32)
            slope = float(getattr(src, "RescaleSlope", 1.0))
            intercept = float(getattr(src, "RescaleIntercept", 0.0))
            stored = np.rint((hu - intercept) / slope)

            pixel_representation = int(getattr(src, "PixelRepresentation", 0))
            if pixel_representation == 0:
                stored = np.clip(stored, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                stored = np.clip(stored, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

            stored = np.ascontiguousarray(stored)
            ds.PixelData = stored.tobytes()
            ds.Rows, ds.Columns = stored.shape

            # Geometry from the reference slice
            ds.PixelSpacing = [float(x) for x in geom.PixelSpacing]
            ds.ImagePositionPatient = [float(x) for x in geom.ImagePositionPatient]
            ds.ImageOrientationPatient = [float(x) for x in geom.ImageOrientationPatient]
            ds.FrameOfReferenceUID = geom.FrameOfReferenceUID
            ds.SliceThickness = float(getattr(geom, "SliceThickness", src.get("SliceThickness", 1.0)))

            # Keep source pixel format / calibration consistent
            ds.BitsAllocated = int(getattr(src, "BitsAllocated", 16))
            ds.BitsStored = int(getattr(src, "BitsStored", 16))
            ds.HighBit = int(getattr(src, "HighBit", ds.BitsStored - 1))
            ds.PixelRepresentation = pixel_representation
            ds.RescaleSlope = slope
            ds.RescaleIntercept = intercept
            ds.RescaleType = getattr(src, "RescaleType", "HU")

            # Series / instance identifiers
            ds.SeriesInstanceUID = new_series_uid
            ds.SOPInstanceUID = generate_uid()
            ds.InstanceNumber = z + 1

            # Derived CT flags
            ds.ImageType = ["DERIVED", "SECONDARY"]
            ds.SeriesDescription = "Registered CT"
            ds.DerivationDescription = self._derivation_description(
                "CT resampled onto fixed CT grid after registration",
                transform_record,
            )

            out_path = output_dir / f"CT_{z + 1:03d}.dcm"
            if self._should_write(out_path, overwrite):
                ds.save_as(str(out_path), write_like_original=False)

    def _scale_dose_to_uint16(self, dose_array: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Convert a physical-dose array to uint16 stored values and return:
        - the new DoseGridScaling
        - the stored pixel array
        """
        dose_array = np.asarray(dose_array, dtype=np.float32)

        if dose_array.size == 0:
            return 1.0, np.zeros_like(dose_array, dtype=np.uint16)

        finite = np.isfinite(dose_array)
        if not finite.any():
            return 1.0, np.zeros_like(dose_array, dtype=np.uint16)

        max_val = float(np.max(dose_array[finite]))
        if max_val <= 0.0:
            return 1.0, np.zeros_like(dose_array, dtype=np.uint16)

        max_uint16 = np.iinfo(np.uint16).max
        new_scale = max_val / max_uint16

        stored = np.rint(np.clip(dose_array, 0.0, None) / new_scale)
        stored = np.clip(stored, 0, max_uint16).astype(np.uint16)

        return new_scale, stored


    def export_dose(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        reference_dataset: Dataset | None = None,
        transform_record: TransformRecord | None = None,
        overwrite: bool = False,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset.dose_image is None or dataset.dose_metadata is None:
            raise ValueError("Dose image or dose metadata is missing.")

        reference_image = self._reference_image(reference_dataset, dataset)

        dose_image = dataset.dose_image
        if transform_record is not None:
            dose_image = Resample.dose(
                moving_image=dataset.dose_image,
                fixed_image=reference_image,
                transform=transform_record.transform,
            )

        # This should be the dose in physical units at this point.
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
        dose_scale = float(getattr(dataset, "dose_scale",
                           getattr(dataset.dose_metadata, "DoseGridScaling", 1.0)))
        dose_array = dose_array * dose_scale

        dose_scale, stored = self._scale_dose_to_uint16(dose_array)
        stored = np.ascontiguousarray(stored)

        out_path = output_dir / "dose.dcm"
        if not self._should_write(out_path, overwrite):
            return

        ds = copy.deepcopy(dataset.dose_metadata)

        ds.PixelData = stored.tobytes()
        ds.Rows = int(stored.shape[1])
        ds.Columns = int(stored.shape[2])
        ds.NumberOfFrames = int(stored.shape[0])

        spacing = dose_image.GetSpacing()
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.ImagePositionPatient = list(
            map(float, dose_image.TransformIndexToPhysicalPoint((0, 0, 0)))
        )
        ds.ImageOrientationPatient = self._image_orientation_patient(dose_image)
        ds.FrameOfReferenceUID = reference_dataset.ct_metadata[0].FrameOfReferenceUID

        ds.GridFrameOffsetVector = [float(i * spacing[2]) for i in range(stored.shape[0])]
        ds.FrameIncrementPointer = [pydicom.tag.Tag(0x3004, 0x000C)]

        ds.DoseGridScaling = float(dose_scale)

        ds.Modality = "RTDOSE"
        ds.SOPClassUID = pydicom.uid.RTDoseStorage
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()

        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.SeriesDescription = "Registered RTDOSE"
        ds.DerivationDescription = self._derivation_description(
            "Dose resampled onto reference grid",
            transform_record,
        )

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        ds.DoseUnits = getattr(ds, "DoseUnits", "GY")
        ds.DoseType = getattr(ds, "DoseType", "PHYSICAL")
        ds.DoseSummationType = getattr(ds, "DoseSummationType", "PLAN")

        ds.save_as(str(out_path), write_like_original=False)

    def _reference_image(
        self,
        reference_dataset: Dataset | None,
        dataset: Dataset,
    ) -> sitk.Image:
        if reference_dataset is not None:
            if reference_dataset.ct_image is not None:
                return reference_dataset.ct_image
            if reference_dataset.dose_image is not None:
                return reference_dataset.dose_image

        if dataset.ct_image is not None:
            return dataset.ct_image
        if dataset.dose_image is not None:
            return dataset.dose_image

        raise ValueError("No reference image available.")

    def _image_orientation_patient(self, image: sitk.Image) -> list[float]:
        direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
        return [
            float(direction[0, 0]),
            float(direction[1, 0]),
            float(direction[2, 0]),
            float(direction[0, 1]),
            float(direction[1, 1]),
            float(direction[2, 1]),
        ]

    def _scale_dose_to_uint32(self, dose_array: np.ndarray) -> tuple[float, np.ndarray]:
        max_val = float(np.nanmax(dose_array)) if dose_array.size else 0.0

        if max_val <= 0.0:
            return 1.0, np.zeros_like(dose_array, dtype=np.uint32)

        max_uint32 = np.iinfo(np.uint32).max
        scale = max_val / max_uint32
        stored = np.round(dose_array / scale)
        stored = np.clip(stored, 0, max_uint32).astype(np.uint32)
        return scale, stored

    def _derivation_description(
        self,
        base: str,
        transform_record: TransformRecord | None,
    ) -> str:
        if transform_record is None:
            return base

        parts = [
            base,
            f"Transform: {transform_record.name}",
            f"Type: {transform_record.transform_type.value}",
            f"Completed: {transform_record.time_complete.isoformat()}",
        ]
        return ". ".join(parts) + "."

    def _should_write(self, filepath: Path, overwrite: bool) -> bool:
        return overwrite or not filepath.exists()