from __future__ import annotations

from pathlib import Path

import pydicom


class DatasetDiscovery:
    """
    Utilities for discovering datasets and related files on disk.
    """

    @staticmethod
    def count_datasets(root_dir: str, progress_callback=None) -> dict:
        """
        Discover CT datasets under a directory.

        Parameters
        ----------
        root_dir:
            Root directory to scan.
        progress_callback:
            Optional callback of the form callback(current, total).

        Returns
        -------
        dict
            Summary dictionary with dataset counts and dataset descriptors.
        """
        datasets: list[dict[str, str]] = []
        seen_series: set[str] = set()

        directories = [p for p in Path(root_dir).rglob("*") if p.is_dir()]
        total_dirs = len(directories)

        for i, directory in enumerate(directories, start=1):
            if progress_callback:
                progress_callback(i, total_dirs)

            for path in directory.iterdir():
                if not path.is_file():
                    continue

                name = path.name.lower()

                # NIfTI CT
                if "ct" in name and (name.endswith(".nii") or name.endswith(".nii.gz")):
                    datasets.append(
                        {
                            "name": path.name,
                            "path": str(path),
                            "type": "nifti",
                        }
                    )
                    continue

                # DICOM CT
                try:
                    ds = pydicom.dcmread(
                        path,
                        stop_before_pixels=True,
                        specific_tags=[
                            "Modality",
                            "SeriesInstanceUID",
                        ],
                    )

                    if getattr(ds, "Modality", None) != "CT":
                        continue

                    series_uid = getattr(ds, "SeriesInstanceUID", None)

                    if not series_uid or series_uid in seen_series:
                        continue

                    seen_series.add(series_uid)

                    datasets.append(
                        {
                            "name": str(directory.relative_to(root_dir)),
                            "path": str(directory),
                            "type": "dicom",
                        }
                    )

                except Exception:
                    pass

        dicom_count = sum(dataset["type"] == "dicom" for dataset in datasets)
        nifti_count = sum(dataset["type"] == "nifti" for dataset in datasets)

        return {
            "dicom_count": dicom_count,
            "nifti_count": nifti_count,
            "total_count": len(datasets),
            "datasets": sorted(datasets, key=lambda dataset: dataset["name"].lower()),
        }

    @staticmethod
    def discover_datasets(root_dir: str, progress_callback=None) -> dict:
        """
        Alias for count_ct_datasets(), kept for readability.
        """
        return DatasetDiscovery.count_ct_datasets(root_dir, progress_callback=progress_callback)

    @staticmethod
    def find_hdf5_files(directory: str) -> list[str]:
        """
        Return all .h5 / .hdf5 files directly inside a directory.
        """
        return [
            str(file)
            for file in Path(directory).iterdir()
            if file.is_file() and file.suffix.lower() in {".h5", ".hdf5"}
        ]