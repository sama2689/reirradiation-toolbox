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
        rasterize_structures: list[str]|bool = False,
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

            #Read Structures
            self._emit_progress(15, "Reading structures")
            try:
                dataset.rtstruct = self.read_rtstruct(directory)

            except (FileNotFoundError, RuntimeError, InvalidDicomError, OSError, ValueError) as e:
                logger.warning("Could not read RTSTRUCT: %s", e)


            #if user wants to rasterize structures, do it now
            if rasterize_structures:
                #make list of structures to rasterize
                if rasterize_structures is True:
                    structures_to_rasterize = dataset.structure_names
                else:
                    structures_to_rasterize=rasterize_structures

                for structure_name in structures_to_rasterize:
                    try:
                        self.get_rasterized_structure(dataset, structure_name)
                    except (ValueError, RuntimeError) as e:
                        #empty structures are normal, skip them
                        if "No contours found" in str(e):
                            continue

                        logger.warning("Could not rasterize structure '%s': %s", structure_name, e)
                

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

    def get_dose_image(
        self,
        dataset:Dataset,
        )->sitk.Image:
        """
        Returns the dose image for a Dataset, if the dose has already been loaded, return the cached image
        """

        if dataset.dose_image is not None:
            return dataset.dose_image

        dose_image, dose_metadata=self.read_dose(dataset.directory)

        dose_image=Resample.dose(moving_image=dose_image,
                                 fixed_image=dataset.ct_image)

        dataset.dose_image = dose_image
        dataset.dose_metadata = dose_metadata

        return dataset.dose_image

    def read_rtstruct(
        self,
        directory: str | Path,
    ) -> pydicom.dataset.FileDataset:
        """
        Read the raw RTSTRUCT DICOM object without creating structure masks.
        """
        rtstruct_file = self._find_file_by_modality(directory, "RTSTRUCT")
        return pydicom.dcmread(str(rtstruct_file), stop_before_pixels=True)


    def get_rasterized_structure(
            self,
            dataset: Dataset,
            structure_name: str,
    )->Structure:
        """
        Returns one RTStruct structure as a rasterized binary mask image, given the structure name. Attaches it to the dataset if it has not been rasterized yet.
        If the structure has already been rasterized, return the cached Stucture object instead of rasterising again
        """

        #can't rasterize onto ct if there is no ct
        if dataset.ct_image is None:
            raise ValueError("CT image must be available in the dataset to rasterize structures.")

        if dataset.rtstruct is None:
            raise ValueError("Dataset does not contain an RTSTRUCT.")
        
        #clean the requested name
        cleaned_name = Structure.generate_clean_name(structure_name)

        #if the structure has already been rasterized, just return it now

        if cleaned_name in dataset.rasterized_structures:
            return dataset.rasterized_structures[cleaned_name]

        #otherwise, we need to rasterize the structure and add it to the dataset
        #Map RT STRUCT ROI names to numbers
        roi_name_to_number = self._get_roi_name_to_number(dataset.rtstruct)

        #get the roi_number and import_name (aka uncleaned name) for the requested structure
        import_name = None
        roi_number=None
        for name, number in roi_name_to_number.items():
            if Structure.generate_clean_name(name) == cleaned_name:
                import_name = name
                roi_number = number
                break
        if roi_number is None:
            raise ValueError(f"Structure '{structure_name}' not found in RTSTRUCT.")


        #get the contour sequence for the requested structure
        contour_seq=self._get_contour_sequence(dataset.rtstruct, roi_number)
        if not contour_seq:
            raise ValueError(f"No contours found for structure '{structure_name}'.")

        #Rasterize the structure onto a binary mask
        mask, _ = self._get_structure_mask(
            ct_image=dataset.ct_image,
            contour_seq=contour_seq,
            structure_name=import_name,
        )

        #convert the numpy binary mask into a structure object
        structure = self._create_structure(import_name, mask, dataset.ct_image)

        #cache the rasterized structure in the dataset for future use
        dataset.rasterized_structures[cleaned_name] = structure

        return structure


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


    def _get_contour_sequence(
            self,
            rtstruct: pydicom.dataset.Dataset,
            roi_number: int
    ):
        """
        Return contour sequence belonging to an roi number
        """

        #loop through all roi_contours in the rtstruct
        for roi_contour in getattr(rtstruct, "ROIContourSequence", []): #gettr returns the sequence if it exists, else returns empty list
            
            referenced_roi_number = getattr(roi_contour, "ReferencedROINumber", None) #get the roi number for this contour

            if referenced_roi_number is None:
                continue

            if int(referenced_roi_number) == roi_number: #if the roi number matches the requested one
                return getattr(roi_contour, "ContourSequence", []) #return the contour sequence for this roi number
            # if we reach here, it means the roi number did not match, so we continue to the next roi_contour

        #if we reach here, it means we did not find a matching roi number
        return None
    

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

        #numpy array shape is (z,y,x) but sitk shape is (x,y,z) so we need to reverse the sitk shape to get the numpy shape
        mask = np.zeros(tuple(reversed(ct_image.GetSize())), dtype=np.uint8) 


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
        image.CopyInformation(ct_image)

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