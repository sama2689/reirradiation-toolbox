# example of loading a DICOM folder into the registration dataset object

from registration.dataset.dicom_reader import DICOMReader

from pathlib import Path

#Change as needed to point to your dicom folders. It expects the folder to contain the ct/dose/structures/rtplan folders, but can also read just the ct/dose/structures/rtplan folders if you point it to those directly.
fixed_path = Path("path/to/fixed/data")
moving_path = Path("path/to/moving/data")


fixed_dataset = DICOMReader().read(directory=fixed_path)
moving_dataset = DICOMReader().read(directory=moving_path)


# you can also read the different ct/dose/structures/rtplan using:

ct_example = DICOMReader().read_ct(directory=fixed_path)  # returns a tuple of (ct_image, DICOM metadata)
dose_example = DICOMReader().read_dose(directory=fixed_path)  # returns a tuple of (dose_image, DICOM metadata)


# different parts of the datasets can be accessed using:

fixed_ct = fixed_dataset.ct_image
fixed_dose = fixed_dataset.dose_image
fixed_structures = fixed_dataset.rasterized_structures
fixed_rtplan = fixed_dataset.plan


# the structure names in the dict are cleaned to be lower case and spaces be _, so individual ones can be accessed:

fixed_cord = fixed_dataset.rasterized_structures["cord"]
fixed_cord_mask = fixed_cord.image  # .image returns the rasterised mask that can be used for doses/contour visualisation


# masks can be generated using contours or thresholding

from registration.methods.masking import Masking

fixed_mask = Masking().contour_mask(
    contour=fixed_dataset.rasterized_structures["body"].image,
    dilation=1,
    erosion=0,
)
moving_mask = Masking().threshold_mask(
    image=moving_dataset.ct_image,
    lower_threshold=-200,
    upper_threshold=3000,
    dilation=1,
    erosion=0,
)

fixed_working_image = Masking().apply_mask_to_image(fixed_dataset.ct_image, fixed_mask)
moving_working_image = Masking().apply_mask_to_image(moving_dataset.ct_image, moving_mask)


# images can be cropped to the region of interest

from registration.methods.cropping import Cropping

fixed_working_image, fixed_working_mask = Cropping().crop_image_and_mask(fixed_working_image, fixed_mask, padding=5)
moving_working_image, moving_working_mask = Cropping().crop_image_and_mask(moving_working_image, moving_mask, padding=5)


# simple call back to print registration progress to command line
def progress_callback(progress: dict[str, object]) -> None:
    level = progress.get("level", "?")
    total_levels = progress.get("total_levels", "?")
    iteration = progress.get("iteration", "?")
    metric = progress.get("metric")

    metric_str = "N/A" if metric is None else f"{metric:.6f}"

    print(f"\rLevel: {level}/{total_levels} | Iteration = {iteration} | Metric = {metric_str}             ", end="", flush=True)

# rigid registration can be performed using:

from registration.methods.rigid_registration import RigidRegistration

rigid_params = {}

rigid_registration = RigidRegistration().configure(rigid_params)
rigid_transform = rigid_registration.execute(
    fixed_image=fixed_working_image,
    moving_image=moving_working_image,
    fixed_mask=fixed_working_mask,
    moving_mask=moving_working_mask,
    progress_callback=progress_callback,
)
print("Rigid registration complete.")


# transforms can be applied to images, doses, contours, using:

from registration.methods.resample import Resample

rigid_image = Resample().ct(
    moving_image=moving_dataset.ct_image,
    fixed_image=fixed_dataset.ct_image,
    transform=rigid_transform
) #this resamples the moving image to the fixed image
rigid_dose = Resample().dose(
    moving_image=moving_dataset.dose_image,
    fixed_image=fixed_dataset.ct_image,
    transform=rigid_transform
)
rigid_structures = Resample().structure_set(
    structures=moving_dataset.rasterized_structures,
    reference_image=fixed_dataset.ct_image,
    transform=rigid_transform,
)


# the images can be cropped to the overlapping region, so it doesn't try to match anatomy we aren't interested in:
# it does this in the untransformed images, but uses an input transform to capture the overalp based upon that.

fixed_overlap, moving_overlap = Cropping().crop_by_axis_overlap(
    fixed_image=fixed_working_image,
    fixed_mask=fixed_working_mask,
    moving_image=moving_working_image,
    moving_mask=moving_working_mask,
    transform=rigid_transform,
)
fixed_working_image, fixed_working_mask = fixed_overlap
moving_working_image, moving_working_mask = moving_overlap


# then we can perform demons using:

from registration.methods.demons_registration import DemonsRegistration

demons_params = {}

demons_registration = DemonsRegistration().configure(demons_params)
demons_transform = demons_registration.execute(
    fixed_image=fixed_working_image,
    moving_image=moving_working_image,
    initial_transform=rigid_transform,
    progress_callback=progress_callback,
)
print("Demons registration complete.")


# again can transform the images

demons_image = Resample().ct(
    moving_image=moving_dataset.ct_image,
    fixed_image=fixed_dataset.ct_image,
    transform=demons_transform
)
demons_dose = Resample().dose(
    moving_image=moving_dataset.dose_image,
    fixed_image=fixed_dataset.ct_image,
    transform=demons_transform
)
demons_structures = Resample().structure_set(
    structures=moving_dataset.rasterized_structures,
    reference_image=fixed_dataset.ct_image,
    transform=demons_transform,
)


# the transforms can be added to a transform_set object, and saved:

from registration.dataset.dataset import TransformRecord, TransformSet, TransformType

transforms = TransformSet()
transforms.add(TransformRecord(
    name="rigid",
    transform=rigid_transform,
    transform_type=TransformType.RIGID,
    moving_image_id=moving_dataset.id,
    fixed_image_id=fixed_dataset.id,
    parameters=rigid_registration.parameters,
))
transforms.add(TransformRecord(
    name="demons",
    transform=demons_transform,
    transform_type=TransformType.DEFORMABLE,
    moving_image_id=moving_dataset.id,
    fixed_image_id=fixed_dataset.id,
    parameters=demons_registration.parameters,
))


# datasets and transform sets can be exported:


from registration.dataset.dataset_writer import DatasetWriter, TransformWriter

output_path = Path("C:/Users/pratik.samant/OneDrive - Oxford University Hospitals NHS Foundation Trust/Python/UseEveryClass/TestOutputPath")

#need to create fixed and moving folders first
DatasetWriter().write(dataset=fixed_dataset, directory=output_path / "fixed")
DatasetWriter().write(dataset=moving_dataset, directory=output_path / "moving")

TransformWriter().write(collection=transforms, directory=output_path)


# they can be read using:

from registration.dataset.dataset_reader import DatasetReader, TransformReader

fixed_imported_dataset = DatasetReader().read(directory=output_path / "fixed")
moving_imported_dataset = DatasetReader().read(directory=output_path / "moving")

transforms_imported = TransformReader().read(directory=output_path)


# dicoms can be exported using:

from registration.dataset.dicom_writer import DICOMWriter
from registration.dataset.dataset import Dataset

output_dataset = Dataset(
    ct_image=demons_image,
    ct_metadata=moving_dataset.ct_metadata,
    dose_image=demons_dose,
    dose_metadata=moving_dataset.dose_metadata,
)
DICOMWriter().export(
    dataset=output_dataset,
    output_dir=output_path / "dicom",
    reference_dataset=fixed_dataset,        # needs this for proper dicom geometry tags
)


# for dvh metrics use:

from registration.metrics.dose import DVH

cord_median_dose = DVH.dose_median(
    dose_image=demons_dose,
    mask_image=demons_structures["cord"].image,
    dose_scaling_factor=moving_dataset.dose_scaling
)
cord_1cc_dose = DVH.dose_cc(
    dose_image=demons_dose,
    volume_cc=1,
    mask_image=demons_structures["cord"].image,
    dose_scaling_factor=moving_dataset.dose_scaling
)
# can look through the DVH class for all the methods, can also generate cumulative dvh graphs as well
print("Cord, D50  =", cord_median_dose)
print("Cord, D1cc =", cord_1cc_dose)