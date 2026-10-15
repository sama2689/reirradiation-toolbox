import napari
import SimpleITK as sitk
import numpy as np
from napari.utils import Colormap

class CTOverlayViewer:
    """Display a fixed image, a moving image, and an optional disagreement mask in napari.

    The class wraps a napari viewer instance and accepts either NumPy arrays or
    SimpleITK images. SimpleITK inputs are converted to NumPy arrays before they
    are added as napari layers.
    """


    def __init__(
        self,
        title:str,
        fixed_image:np.ndarray|sitk.Image|None=None,
        resampled_moving_image: np.ndarray|sitk.Image|None=None,
        difference_array:np.ndarray|None=None,
        fixed_dose_np:np.ndarray|None=None,
        moving_dose_np:np.ndarray|None=None,
        total_dose_np:np.ndarray|None=None,
        auto_show: bool = True,
    ):
        """Create the viewer and optionally render the configured layers immediately.
            Assumes that everything is resampled to match fixed_image.
            Assumes are doses are scaled by dose_scaling factor and are absolute numpy arrays
        Parameters
        ----------
        fixed_image : np.ndarray | sitk.Image | None
            Reference image to display.
        resampled_moving_image : np.ndarray | sitk.Image | None
            Moving image to display over the fixed image.
        difference_image : np.ndarray | sitk.Image | None
            Optional label or mask image used to highlight disagreement regions.
        auto_show : bool, default False
            If True, build the layers and launch the napari window immediately.
        """

        self.viewer = napari.Viewer(title=title,show=False) #show is false by default as best to launch via viewer_show()), kept separate from auto_show

    
        if fixed_image is not None and resampled_moving_image is not None:
            self.build_image_stack(fixed_image, resampled_moving_image, difference_array)

        self.build_dose_stack(
            fixed_dose_np=fixed_dose_np,
            moving_dose_np=moving_dose_np,
            total_dose_np=total_dose_np
        )
        
        #if user wants to show on initialization then do so
        if auto_show:
            self.show()

    def build_image_stack(
        self,
        fixed_image: np.ndarray | sitk.Image,
        moving_image: np.ndarray | sitk.Image,
        difference_array: np.ndarray| None,
        fixed_contrast_limits: tuple = (-786, 1024),
        moving_contrast_limits: tuple = (-786, 1024),
    ):
        """Add the fixed, moving, and disagreement layers to the viewer. 

        Parameters
        ----------
        fixed_image : np.ndarray | sitk.Image
            Reference image to show in the viewer.
        moving_image : np.ndarray | sitk.Image
            Resampled or registered moving image to overlay on the fixed image.
        difference_image : np.ndarray | None
            Optional label or mask used to highlight disagreement regions.
        fixed_contrast_limits : tuple[float, float], optional
            Display contrast range for the fixed image layer. defualt values based on head and neck
        moving_contrast_limits : tuple[float, float], optional
            Display contrast range for the moving image layer. default values based on head and neck
        """
        # push to arrays if they aren't already
        fixed_array = self._to_array(fixed_image)
        moving_array = self._to_array(moving_image)

        # Add the fixed image layer
        self.viewer.add_image(
            fixed_array,
            name="Fixed Image",
            opacity=0.5,
            colormap="gray",
            contrast_limits=fixed_contrast_limits,
            interpolation2d="spline36"
        )

        # Add the moving image layer
        self.viewer.add_image(
            moving_array,
            name="Moving Image",
            opacity=0.5,
            colormap="gray",
            blending="additive", 
            contrast_limits=moving_contrast_limits,
            interpolation2d="spline36"
        )

        if difference_array is not None:
            difference_layer = self.viewer.add_labels(
                difference_array,
                name="Image Disagreement",
                opacity=0.1,
            )


            difference_layer.color = {
                1: "red",
                2: "blue",
            }

    def build_dose_stack(self,
            fixed_dose_np:np.ndarray|None=None,
            moving_dose_np:np.ndarray|None=None,
            total_dose_np:np.ndarray|None=None
            ):
        """Add optional dose layers to the napari viewer.

        Parameters
        ----------
        fixed_dose_np : np.ndarray | None, optional
            Dose distribution for the fixed image. Each non-zero voxel is rendered
            with an Eclipse-inspired colormap.
        moving_dose_np : np.ndarray | None, optional
            Dose distribution for the moving image.
        total_dose_np : np.ndarray | None, optional
            Accumulated dose distribution to show as an additional overlay.
        """

        #only make the cmap if any are not none. 
        if fixed_dose_np is not None or moving_dose_np is not None or total_dose_np is not None:
            eclipse_cmap = self._return_eclipsecmap()



        if fixed_dose_np is not None:
            #find dosemax
            fixed_dosemax= np.max(fixed_dose_np[fixed_dose_np>0]) if np.any(fixed_dose_np>0) else 1.0

            #add to viewer
            self.viewer.add_image(
                fixed_dose_np,
                name="Fixed Dose",
                colormap=eclipse_cmap,
                opacity=0.25,
                blending="additive",
                contrast_limits=(0,fixed_dosemax))

        if moving_dose_np is not None:
            #find dosemax
            moving_dosemax=np.max(moving_dose_np[moving_dose_np>0]) if np.any(moving_dose_np>0) else 1.0

            #add to viewer
            self.viewer.add_image(
                moving_dose_np,
                name="Moving Dose",
                colormap=eclipse_cmap,
                opacity=0.25,
                blending="additive",
                contrast_limits=(0,moving_dosemax)
            )

        if total_dose_np is not None:

            #find dosemax
            total_dosemax=np.max(total_dose_np[total_dose_np>0]) if np.any(total_dose_np>0) else 1.0

            #add to viewer
            self.viewer.add_image(
                total_dose_np,
                name="Accumulated Dose",
                colormap=eclipse_cmap,
                opacity=0.25,
                blending="additive",
                contrast_limits=(0,total_dosemax)
            )



    def show(self):
        """Display the napari viewer for this instance."""
        self.viewer.show()


    @staticmethod
    def _to_array(image:np.ndarray | sitk.Image |None)->np.ndarray:
        """Convert a SimpleITK image to a NumPy array when needed.

        Parameters
        ----------
        image : 
            The image to convert.

        Returns
        -------
        np.ndarray
            The image as a NumPy array.
        """
        if isinstance(image, sitk.Image):
            return sitk.GetArrayFromImage(image)
        return image


    @staticmethod
    def _return_eclipsecmap():
        #make a colormap that matches eclipse, used to display dose

        eclipse_colors = [
            "#00000305",  # black
            "#0000A8",  # deep blue
            "#0047AB",  # royal blue
            "#00AEEF",  # cyan
            "#00CC66",  # green
            "#D7E000",  # yellow-green
            "#FFB000",  # amber
            "#FF5A00",  # orange
            "#FF0000",  # red
            ]

        eclipse_cmap = Colormap(
            eclipse_colors,
            name="eclipse_dose",
            )

        return eclipse_cmap

 
        



    


