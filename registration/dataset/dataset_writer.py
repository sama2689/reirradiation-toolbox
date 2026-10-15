from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk

from registration.dataset.dataset import Dataset, TransformSet


class DatasetWriter:
    def write(self, dataset: Dataset, directory: str | Path, overwrite: bool = False) -> None:
        directory = Path(directory)
        self._init_folder(directory)

        if dataset.ct_image is not None:
            ct_path = directory / "ct_image.nii.gz"
            if self._should_write(ct_path, overwrite):
                sitk.WriteImage(dataset.ct_image, str(ct_path))

        if dataset.dose_image is not None:
            dose_path = directory / "dose_image.nii.gz"
            if self._should_write(dose_path, overwrite):
                sitk.WriteImage(dataset.dose_image, str(dose_path))

        if dataset.ct_metadata is not None:
            ct_meta_path = directory / "ct_metadata"
            if self._should_write(ct_meta_path, overwrite):
                self._write_dicom_series_headers(
                    dataset.ct_metadata,
                    directory / "ct_metadata",
                    overwrite=overwrite,
                )

        if dataset.dose_metadata is not None:
            dose_meta_path = directory / "dose_metadata.dcm"
            if self._should_write(dose_meta_path, overwrite):
                dataset.dose_metadata.save_as(str(dose_meta_path))

        if dataset.plan is not None:
            plan_path = directory / "plan.dcm"
            if self._should_write(plan_path, overwrite):
                dataset.plan.save_as(str(plan_path))

        # structures
        if dataset.structures:
            structures_metadata = {
                "structures": []
            }

            for structure in dataset.structures.values():
                file_name = f"{structure.id}.nii.gz"
                structure_path = self.structures_directory / file_name

                if self._should_write(structure_path, overwrite):
                    sitk.WriteImage(structure.image, str(structure_path))

                structures_metadata["structures"].append(
                    {
                        "id": str(structure.id),
                        "file_name": file_name,
                        "import_name": structure.import_name,
                        "display_name": structure.display_name,
                        "cleaned_name": structure.cleaned_name,
                    }
                )

            metadata_path = self.structures_directory / "structures_metadata.json"

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(structures_metadata, f, indent=2)

        metadata = {
            "dataset_id": str(dataset.id),
            "directory": str(dataset.directory) if dataset.directory is not None else None,
            "ct_image": self._image_meta(dataset.ct_image),
            "dose_image": self._image_meta(dataset.dose_image)
        }

        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _init_folder(self, directory: Path) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory.")

        self.structures_directory = directory / "structures"
        self.structures_directory.mkdir(exist_ok=True)

    def _should_write(self, filepath: Path, overwrite: bool) -> bool:
        return overwrite or not filepath.exists()

    def _image_meta(self, image: sitk.Image | None) -> dict[str, object] | None:
        if image is None:
            return None

        return {
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
            "size": list(image.GetSize()),
        }

    def _write_dicom_series_headers(
        self,
        headers: list,
        directory: Path,
        overwrite: bool = False,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        for i, ds in enumerate(headers, start=1):
            path = directory / f"CT_{i:03d}.dcm"
            if path.exists() and not overwrite:
                continue

            ds.save_as(str(path), write_like_original=False)


class TransformWriter:
    def write(
        self,
        collection: TransformSet,
        directory: str | Path,
        overwrite: bool = False,
    ) -> None:
        directory = Path(directory)
        self._init_folder(directory)

        for record in collection.transforms.values():
            transform_path = self.transforms_directory / f"{record.id}.hdf5"
            if self._should_write(transform_path, overwrite):
                tx = self._prepare_transform_for_write(record.transform)
                sitk.WriteTransform(tx, str(transform_path))

        # always update metadata for any changes
        metadata_path = self.transforms_directory / "transforms_metadata.json"
        metadata = self._create_metadata(collection)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _init_folder(self, directory: Path) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a valid directory.")

        self.transforms_directory = directory / "transforms"
        self.transforms_directory.mkdir(exist_ok=True)

    def _should_write(self, filepath: Path, overwrite: bool) -> bool:
        return overwrite or not filepath.exists()

    def _prepare_transform_for_write(self, transform: sitk.Transform) -> sitk.Transform:
        """
        Flatten composite transforms so SimpleITK can write them to HDF5 safely.
        """
        if transform.GetName() != "CompositeTransform":
            return transform

        composite = sitk.CompositeTransform(transform)
        composite.FlattenTransform()
        return composite

    def _create_metadata(self, collection: TransformSet) -> dict:
        return {
            "transforms": [
                {
                    "id": str(record.id),
                    "name": record.name,
                    "file_name": f"{record.id}.hdf5",
                    "moving_image_id": str(record.moving_image_id),
                    "fixed_image_id": str(record.fixed_image_id),
                    "transform_type": record.transform_type.value,
                    "time_complete": record.time_complete.isoformat(),
                    "parameters": self._to_jsonable(record.parameters),
                }
                for record in collection.transforms.values()
            ]
        }

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]

        if hasattr(value, "tolist"):
            return value.tolist()

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if hasattr(value, "__dict__"):
            return {
                k: self._to_jsonable(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }

        return str(value)