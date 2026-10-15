from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pydicom
import SimpleITK as sitk
from datetime import datetime

from registration.dataset.dataset import Dataset, Structure, TransformRecord, TransformType, TransformSet


class DatasetReader:
    def read(self, directory: str | Path) -> Dataset:
        """
        Read a previously exported Dataset for image registration from a directory.

        Args:
            directory (str | Path): directory of the exported Dataset.

        Returns:
            Dataset: imported Dataset for image registration.
        """

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory.")

        metadata = self._load_metadata(directory / "metadata.json")

        ct_image = self._read_image(directory / "ct_image.nii.gz")
        dose_image = self._read_image(directory / "dose_image.nii.gz")

        ct_metadata = self._read_dicom_series_headers(directory / "ct_metadata")
        dose_metadata = self._read_dicom(directory / "dose_metadata.dcm")
        plan = self._read_dicom(directory / "plan.dcm")

        structures = self._read_structures(directory / "structures")

        dataset = Dataset(
            ct_image=ct_image,
            dose_image=dose_image,
            structures=structures,
            ct_metadata=ct_metadata,
            dose_metadata=dose_metadata,
            plan=plan,
        )

        dataset.directory = directory

        if "dataset_id" in metadata:
            try:
                dataset.id = UUID(str(metadata["dataset_id"]))
            except Exception:
                pass

        return dataset

    def _load_metadata(self, path: Path) -> dict:
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_image(self, path: Path) -> sitk.Image | None:
        if not path.exists():
            return None
        return sitk.ReadImage(str(path))

    def _read_dicom(self, path: Path) -> pydicom.dataset.FileDataset | None:
        if not path.exists():
            return None
        return pydicom.dcmread(str(path))

    def _read_structures(
        self,
        structures_directory: Path,
    ) -> dict[str, Structure]:
        """Read exported structures from disk."""

        structures: dict[str, Structure] = {}

        metadata_path = structures_directory / "structures_metadata.json"
        if not metadata_path.exists():
            return structures

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for item in metadata.get("structures", []):
            image = sitk.ReadImage(
                str(structures_directory / item["file_name"])
            )

            structure = Structure(
                id=UUID(item["id"]),
                import_name=item["import_name"],
                display_name=item["display_name"],
                image=image,
            )

            structures[structure.cleaned_name] = structure

        return structures

    def _looks_like_uuid(self, value: str) -> bool:
        try:
            UUID(value)
            return True
        except Exception:
            return False

    def _read_dicom_series_headers(
        self,
        directory: Path,
    ) -> list[pydicom.dataset.FileDataset]:
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"CT metadata directory not found: {directory}")

        files = sorted(directory.glob("CT_*.dcm"))
        if not files:
            raise FileNotFoundError(f"No CT_*.dcm files found in {directory}")

        headers = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in files]
        headers.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
        return headers
        

class TransformReader:
    def read(self, directory: str | Path) -> TransformSet:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory.")

        transforms_dir = directory / "transforms"

        metadata_path = transforms_dir / "transforms_metadata.json"
        metadata = self._load_metadata(metadata_path)

        collection = TransformSet()
        if "collection_id" in metadata:
            try:
                collection.id = UUID(str(metadata["collection_id"]))
            except Exception:
                pass

        for item in metadata.get("transforms", []):
            record = self._read_record(transforms_dir, item)
            if record is not None:
                collection.add(record)

        return collection

    def _load_metadata(self, path: Path) -> dict:
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_record(
        self,
        transforms_dir: Path,
        item: dict,
    ) -> TransformRecord | None:
        try:
            transform_id = UUID(str(item["id"]))
            file_name = item["file_name"]
            name = item["name"]
            moving_image_id = UUID(str(item["moving_image_id"]))
            fixed_image_id = UUID(str(item["fixed_image_id"]))
            transform_type = TransformType(item["transform_type"])

            time_complete = datetime.fromisoformat(item["time_complete"])
            parameters = item.get("parameters", {})

        except (KeyError, ValueError, TypeError):
            return None

        transform_path = transforms_dir / file_name
        if not transform_path.exists():
            return None

        transform = sitk.ReadTransform(str(transform_path))

        return TransformRecord(
            id=transform_id,
            name=name,
            transform=transform,
            moving_image_id=moving_image_id,
            fixed_image_id=fixed_image_id,
            transform_type=transform_type,
            time_complete=time_complete,
            parameters=parameters,
        )