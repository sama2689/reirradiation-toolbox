from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError
from skimage.draw import polygon

from registration.dataset.dataset import Dataset, Structure
from registration.methods.resample import Resample

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]


class DICOMReader:
    """Import DICOM CT image series, RTDOSE, RTSTRUCT, and RTPLAN for image registration."""

    def read(
        self,
        directory: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> Dataset:
        """
        Read available CT, RTDOSE, RTSTRUCT, and RTPLAN data from a DICOM directory.

        Missing or unreadable components are logged and skipped when possible, so the
        import returns a partial Dataset rather than failing the entire read.

        Args:
            directory: Directory containing the DICOM files.
            progress_callback: Optional callback for reporting import progress as
                ``(percent, message)``.

        Returns:
            A Dataset containing any imported CT image, dose image, structures, and
            DICOM metadata that could be read successfully.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory.")

        self._progress_callback = progress_callback
        dataset = Dataset(directory=directory)

        try:
            self._emit_progress(0, "Reading CT series")
            try:
                dataset.ct_image, dataset.ct_metadata = self.read_ct(directory)
            except (FileNotFoundError, RuntimeError, InvalidDicomError, OSError, ValueError) as e:
                logger.warning("Could not read CT: %s", e)

            self._emit_progress(10, "Reading dose image")
            try:
                dose_image, dose_metadata = self.read_dose(directory)
                if dataset.ct_image is not None:
                    dose_image = Resample.dose(dose_image, dataset.ct_image)
                else:
                    logger.warning("RTDOSE was read, but CT is missing so dose was not resampled.")
                dataset.dose_image = dose_image
                dataset.dose_metadata = dose_metadata
            except (FileNotFoundError, RuntimeError, InvalidDicomError, OSError, ValueError) as e:
                logger.warning("Could not read RTDOSE: %s", e)

            self._emit_progress(15, "Reading structures")
            if dataset.ct_image is not None:
                try:
                    dataset.structures = self.read_structure(directory, dataset.ct_image)
                except (FileNotFoundError, RuntimeError, InvalidDicomError, OSError, ValueError) as e:
                    logger.warning("Could not read RTSTRUCT: %s", e)
            else:
                logger.warning("Skipping RTSTRUCT because CT is missing.")

            self._emit_progress(95, "Reading plan")
            try:
                dataset.plan = self.read_plan(directory)
            except (FileNotFoundError, RuntimeError, InvalidDicomError, OSError, ValueError) as e:
                logger.warning("Could not read RTPLAN: %s", e)

            self._emit_progress(100, "Complete")
            return dataset

        finally:
            self._progress_callback = None

    def read_ct(
        self,
        directory: str | Path,
    ) -> tuple[sitk.Image, pydicom.dataset.FileDataset]:
        """
        Read the CT image series from a DICOM directory.
        """
        series_id = self._find_series_by_modality(directory, "CT")
        return self._read_series(directory, series_id)

    def read_dose(
        self,
        directory: str | Path,
    ) -> tuple[sitk.Image, pydicom.dataset.FileDataset]:
        """
        Read the RTDOSE from a DICOM directory.
        """
        series_id = self._find_series_by_modality(directory, "RTDOSE")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)

        if not files:
            raise RuntimeError("RTDOSE series was found but contains no files.")

        dose_image = sitk.ReadImage(files[0])
        dose_metadata = pydicom.dcmread(files[0], stop_before_pixels=True)

        return dose_image, dose_metadata

    def read_structure(
        self,
        directory: str | Path,
        ct_image: sitk.Image,
    ) -> dict[str, Structure]:
        """
        Read the RTSTRUCT from a DICOM directory and convert its contours into
        binary mask images.
        """
        rtstruct_file = self._find_file_by_modality(directory, "RTSTRUCT")
        rtstruct = pydicom.dcmread(str(rtstruct_file), stop_before_pixels=True)

        structure_masks = self._get_all_structure_masks(rtstruct, ct_image)
        return self._structure_masks_to_structures(structure_masks, ct_image)

    def read_plan(
        self,
        directory: str | Path,
    ) -> pydicom.dataset.FileDataset:
        """
        Read the RTPLAN from a DICOM directory.
        """
        rtplan_file = self._find_file_by_modality(directory, "RTPLAN")
        return pydicom.dcmread(str(rtplan_file), stop_before_pixels=True)

    def _has_modality(self, file: str | Path, modality: str) -> bool:
        """Check whether a DICOM file has the specified modality."""
        try:
            metadata = pydicom.dcmread(str(file), stop_before_pixels=True)
        except (InvalidDicomError, OSError):
            return False

        return getattr(metadata, "Modality", None) == modality

    def _find_series_by_modality(
        self,
        directory: str | Path,
        modality: str,
    ) -> str:
        """Return the first DICOM series ID with the requested modality."""
        reader = sitk.ImageSeriesReader()
        
        #simpleitk method that will return all filenames matching series id
        # in a directory
        series_ids = reader.GetGDCMSeriesIDs(str(directory)) 

        if not series_ids:
            raise RuntimeError("No DICOM series found.")

        for series_id in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(directory), series_id)
            if files and self._has_modality(files[0], modality):
                return series_id

        raise RuntimeError(f"No {modality} series found.")

    def _find_file_by_modality(
        self,
        directory: str | Path,
        modality: str,
    ) -> Path:
        """Return the first DICOM file with the requested modality."""
        for file in Path(directory).rglob("*"):
            if not file.is_file():
                continue

            if self._has_modality(file, modality):
                return file

        raise RuntimeError(f"No {modality} file found.")

    def _read_series(
        self,
        directory: str | Path,
        series_id: str,
    ) -> tuple[sitk.Image, pydicom.dataset.FileDataset]:
        """Read a DICOM image series and return its image and metadata."""
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(directory), series_id)

        if not files:
            raise RuntimeError("DICOM series was found but contains no files.")

        reader.SetFileNames(files)
        metadata = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
        image = reader.Execute()

        return image, metadata

    def _get_roi_name_to_number(self, rtstruct: pydicom.dataset.Dataset) -> dict[str, int]:
        """Extract ROI names and numbers from an RTSTRUCT."""
        roi_name_to_number: dict[str, int] = {}

        for roi in getattr(rtstruct, "StructureSetROISequence", []):
            if hasattr(roi, "ROIName") and hasattr(roi, "ROINumber"):
                roi_name_to_number[roi.ROIName] = int(roi.ROINumber)

        return roi_name_to_number

    def _get_contour_by_roi_number(
        self,
        rtstruct: pydicom.dataset.Dataset,
    ) -> tuple[dict[int, list], int]:
        """Map ROI numbers to contour sequences and count the total contours."""
        contour_by_roi_number: dict[int, list] = {}
        total_contours = 0

        for contour in getattr(rtstruct, "ROIContourSequence", []):
            roi_number = getattr(contour, "ReferencedROINumber", None)
            contour_seq = getattr(contour, "ContourSequence", None)

            if roi_number is None or not contour_seq:
                continue

            contour_by_roi_number[int(roi_number)] = contour_seq
            total_contours += len(contour_seq)

        return contour_by_roi_number, total_contours

    def _get_all_structure_masks(
        self,
        rtstruct: pydicom.dataset.Dataset,
        ct_image: sitk.Image,
    ) -> dict[str, np.ndarray]:
        """Rasterize all RTSTRUCT contours into binary mask arrays."""
        roi_name_to_number = self._get_roi_name_to_number(rtstruct)
        contour_by_roi_number, total_contours = self._get_contour_by_roi_number(rtstruct)

        masks: dict[str, np.ndarray] = {}

        if total_contours == 0:
            logger.warning("No contours found in RTSTRUCT.")
            self._emit_progress(95, "No contours found")
            return masks

        processed_contours = 0
        start_percent = 15
        end_percent = 95
        span = end_percent - start_percent

        for structure_name, roi_number in roi_name_to_number.items():
            contour_seq = contour_by_roi_number.get(int(roi_number))
            if not contour_seq:
                continue

            try:
                mask, done = self._get_structure_mask(
                    ct_image=ct_image,
                    contour_seq=contour_seq,
                    structure_name=structure_name,
                    processed_contours=processed_contours,
                    total_contours=total_contours,
                    start_percent=start_percent,
                    span=span,
                )
                masks[structure_name] = mask
                processed_contours = done
            except Exception as e:
                logger.warning("Could not rasterize structure %s: %s", structure_name, e)
                continue

        return masks

    def _get_structure_mask(
        self,
        ct_image: sitk.Image,
        contour_seq,
        structure_name: str,
        processed_contours: int = 0,
        total_contours: int = 1,
        start_percent: int = 15,
        span: int = 80,
    ) -> tuple[np.ndarray, int]:
        """Rasterize a single RTSTRUCT contour sequence into a binary mask."""
        mask = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)

        for contour in contour_seq:
            contour_data = getattr(contour, "ContourData", None)
            if not contour_data:
                processed_contours += 1
                continue

            pts = np.asarray(contour_data, dtype=np.float64).reshape(-1, 3)
            vox = np.array(
                [ct_image.TransformPhysicalPointToIndex(tuple(p)) for p in pts],
                dtype=np.int32,
            )

            z = vox[0, 2]
            if 0 <= z < mask.shape[0]:
                rr, cc = polygon(vox[:, 1], vox[:, 0], shape=mask[z].shape)
                mask[z, rr, cc] = 1

            processed_contours += 1

            percent = start_percent + (processed_contours / max(total_contours, 1)) * span
            self._emit_progress(
                int(percent),
                f"Processing {structure_name} ({processed_contours}/{total_contours} contours)",
            )

        return mask, processed_contours

    def _structure_masks_to_structures(
        self,
        structure_masks: dict[str, np.ndarray],
        ct_image: sitk.Image,
    ) -> dict[str, Structure]:
        """Convert structure mask arrays into Structure objects."""
        structures: dict[str, Structure] = {}

        for import_name, mask in structure_masks.items():
            structure = self._create_structure(import_name, mask, ct_image)
            structures[structure.cleaned_name] = structure

        return structures

    def _create_structure(
        self,
        import_name: str,
        mask: np.ndarray,
        ct_image: sitk.Image,
    ) -> Structure:
        """Create a Structure object from a binary mask."""
        image = self._mask_to_sitk(mask, ct_image)
        return Structure(import_name=import_name, image=image)

    def _mask_to_sitk(self, mask: np.ndarray, ct_image: sitk.Image) -> sitk.Image:
        """Convert a NumPy mask to a SimpleITK image using CT geometry."""
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image.SetOrigin(ct_image.GetOrigin())
        image.SetSpacing(ct_image.GetSpacing())
        image.SetDirection(ct_image.GetDirection())
        return image

    def _emit_progress(self, percent: int, message: str) -> None:
        """Send a progress update if a callback is configured."""
        callback = getattr(self, "_progress_callback", None)
        if callback is None:
            return

        try:
            callback(int(percent), message)
        except Exception:
            pass