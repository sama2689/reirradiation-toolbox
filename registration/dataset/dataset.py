from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import re
from pathlib import Path
from enum import Enum
from typing import Any

import pydicom
import SimpleITK as sitk
from datetime import datetime, timezone

@dataclass
class Dataset:
    """
    Dataclass to contain radiotherapy image, dose, structure, and plan data for image registration.
    
    Attributes
    ----------
    ct_image:
        CT image volume in SimpleITK format.
    dose_image:
        Radiotherapy dose distribution in SimpleITK format.
    dose_scaling:
        Scaling factor to convert raw dose values to physical units (Gy).
    structures:
        Dictionary mapping cleaned structure names to Structure objects.
    ct_metadata:
        DICOM metadata of the CT image.
    dose_metadata:
        DICOM metadata of the dose distribution image.
    plan:
        DICOM RT Plan file.
    """
    
    directory: str | Path | None = None

    # images
    ct_image: sitk.Image              | None = None
    dose_image: sitk.Image            | None = None
    structures: dict[str, Structure]  = field(default_factory=dict)
    
    # metadata
    ct_metadata: pydicom.dataset.FileDataset   | None = None
    dose_metadata: pydicom.dataset.FileDataset | None = None
    plan: pydicom.dataset.FileDataset          | None = None

    # unique identifier
    id: UUID = field(default_factory=uuid4)

    @property
    def dose_scaling(self) -> float:
        if self.dose_metadata is None:
            return 1.0

        return float(getattr(self.dose_metadata, "DoseGridScaling", 1.0))

    @property
    def prescription_dose(self) -> float | None:
        if self.plan is None:
            return None

        if "DoseReferenceSequence" not in self.plan:
            return None

        for ref in self.plan.DoseReferenceSequence:
            dose = getattr(ref, "TargetPrescriptionDose", None)
            if dose is not None:
                return float(dose)

        return None

@dataclass
class Structure:
    """
    Dataclass to contain individual contoured volumes for image registration.
    
    Attributes
    ----------
    import_name:
        The name the structure was assigned based on the import.
    image:
        The SimpleITK mask image of the structure.
    display_name:
        Cleaned name ready for display.
    cleaned_name:
        Cleaned name for safe lookups.
    id:
        Unique identifer.
    """
    
    import_name: str
    image: sitk.Image

    display_name: str | None = None
    cleaned_name: str = field(init=False)

    # unique identifier
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self):
        self.cleaned_name = self.generate_clean_name(self.import_name)
        
        if self.display_name is None:
            self.display_name = self.generate_display_name(self.import_name)

    @staticmethod
    def generate_clean_name(volume_name: str) -> str:
        """Generates a clean name from the volume's imported name for safe lookup."""
        volume_name = volume_name.lower()
        volume_name = re.sub(r"[^a-z0-9]+", "_", volume_name)
        return volume_name.strip("_")

    @staticmethod
    def generate_display_name(volume_name: str) -> str:
        """Generates a clean display name from the volume's imported name."""
        volume_name = re.sub(r"[_\-]+", " ", volume_name)
        volume_name = re.sub(r"\s+", " ", volume_name).strip()

        words = []
        acronyms = {
            "CTV", "GTV", "PTV", "ITV", "OAR",
            "PRV", "IGTV", "HRCTV", "IRCTV",
            "LN", "LNL"
        }

        for word in volume_name.split():
            if word.upper() in acronyms:
                words.append(word.upper())
            else:
                words.append(word.capitalize())

        return " ".join(words)

class TransformType(str, Enum):
    RIGID = "rigid"
    DEFORMABLE = "deformable"

@dataclass
class TransformRecord:
    """
    Dataclass to contain image registration transformations.
    
    Attributes
    ----------
    name:
        Name of the transformation.
    transform:
        The SimpleITK transform.
    transform_type:
        Rigid or deformable.
    moving_image_id:
        UID of the moving image.
    fixed_image_id:
        UID of the fixed image.
    parameters:
        Parameters used for the registration.
    time_complete:
        Time and date the registration was complete (UTC).
    id:
        UID of the transformation.
    """
    
    name: str
    transform: sitk.Transform
    transform_type: TransformType
    
    moving_image_id: UUID
    fixed_image_id: UUID


    parameters: dict[str, Any] = field(default_factory = dict)
    time_complete: datetime = field(default_factory = lambda: datetime.now(timezone.utc))

    id: UUID = field(default_factory = uuid4)

@dataclass
class TransformSet:
    "Dictionary of multiple transforms, UID: TransformRecord"
    
    transforms: dict[UUID, TransformRecord] = field(default_factory = dict)

    def add(self, transform: TransformRecord) -> None:
        self.transforms[transform.id] = transform

    def get(self, transform_id: UUID) -> TransformRecord | None:
        return self.transforms.get(transform_id)
