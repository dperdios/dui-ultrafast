from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Optional, Sequence
from matplotlib.axes import Axes
import matplotlib.patches as mpatches

from utils.types import Real, Point2D, ImageAxes2D
from utils.types import assert_positive_real_number


class BaseROI2D(metaclass=ABCMeta):
    def __init__(self, center: Point2D, color: Optional[str] = None):

        # Center
        center = np.asarray(center)
        if center.shape != (2,):
            raise ValueError(
                f"Must be of shape (2,). Current shape: {center.shape}.")

        x_c, y_c = center
        self._center = x_c, y_c

        # Color
        if color is not None:
            if not isinstance(color, str):
                raise TypeError("Color must be a string.")
        self._color = color

    # Properties
    @property
    def center(self):
        return self._center

    @property
    def color(self):
        return self._color

    # Methods
    def compute_mask(
            self,
            image_axes: ImageAxes2D,
    ) -> np.ndarray:
        """Extract mask(s)"""

        # Image meshgrid
        image_meshgrid = np.meshgrid(*image_axes, indexing='ij')

        mask = self._compute_mask(tuple(image_meshgrid))

        return mask

    def draw_boundaries(self, ax: Axes):
        self._draw_boundaries(ax=ax)

    def extract_samples(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> np.ndarray:

        # Make sure images is a NumPy array
        images = np.asarray(images)

        # Check images w.r.t. image axes
        _assert_compat_images_and_axes(images=images, image_axes=image_axes)

        # Compute mask
        mask = self.compute_mask(image_axes=image_axes)

        # Broadcast in case of batched images
        mask = np.broadcast_to(mask, images.shape)

        # Extract corresponding zone samples
        batch_shape = images.shape[:-len(image_axes)] + (-1,)
        samples = np.reshape(images[mask], newshape=batch_shape)

        return samples

    # Abstract methods
    @abstractmethod
    def _compute_mask(self, image_meshgrid: ImageAxes2D) -> np.ndarray:
        pass

    @abstractmethod
    def _draw_boundaries(self, ax: Axes):
        pass


class CircularROI(BaseROI2D):
    def __init__(
            self,
            center: Point2D,
            radius: Real,
            color: Optional[str] = None,
    ):

        # Call super constructor
        super(CircularROI, self).__init__(center=center, color=color)

        # Radius
        assert_positive_real_number(radius)
        self._radius = radius

    # Properties
    @property
    def radius(self):
        return self._radius

    # Methods
    def _compute_mask(self, image_meshgrid: ImageAxes2D) -> np.ndarray:

        x_mg, y_mg = image_meshgrid

        x_c, y_c = self._center
        radius = self._radius

        circle = (x_mg - x_c) ** 2 + (y_mg - y_c) ** 2
        mask = circle <= radius ** 2

        return mask

    def _draw_boundaries(self, ax: Axes):

        # Extract properties
        radius = self._radius
        center = self._center
        color = self._color

        # Add patch
        circle = mpatches.Circle(
            xy=center, radius=radius, edgecolor=color, fill=False
        )
        ax.add_artist(circle)


class CircularAnnulusROI(BaseROI2D):
    def __init__(
            self,
            center: Point2D,
            r_in: Real,
            r_out: Real,
            color: Optional[str] = None,
    ):

        # Call super constructor
        super(CircularAnnulusROI, self).__init__(center=center, color=color)

        # Radius
        assert_positive_real_number(r_in)
        assert_positive_real_number(r_out)
        if r_in >= r_out:
            raise ValueError("Inner radius must be smaller than outer radius.")
        self._r_in = r_in
        self._r_out = r_out

    # Properties
    @property
    def r_in(self):
        return self._r_in

    @property
    def r_out(self):
        return self._r_out

    # Methods
    def _compute_mask(self, image_meshgrid: ImageAxes2D) -> np.ndarray:

        x_mg, y_mg = image_meshgrid

        x_c, y_c = self._center
        r_in = self._r_in
        r_out = self._r_out

        circle = (x_mg - x_c) ** 2 + (y_mg - y_c) ** 2
        # mask = np.logical_and((circle >= r_in ** 2), (circle <= r_out ** 2))
        mask = (circle >= r_in ** 2) & (circle <= r_out ** 2)

        return mask

    def _draw_boundaries(self, ax: Axes):

        # Extract properties
        center = self._center
        r_in = self._r_in
        r_out = self._r_out
        color = self._color

        # Add patches
        c_out = mpatches.Circle(
            xy=center, radius=r_out, edgecolor=color, fill=False
        )
        ax.add_artist(c_out)
        if r_in > 0:
            c_in = mpatches.Circle(
                xy=center, radius=r_in, edgecolor=color, fill=False
            )
            ax.add_artist(c_in)


class RectangularROI(BaseROI2D):
    def __init__(
            self,
            center: Point2D,
            width: Real,
            height: Real,
            color: Optional[str] = None,
    ):

        # Call super constructor
        super(RectangularROI, self).__init__(center=center, color=color)

        # Extract properties
        x_c, y_c = self._center

        # Width and height
        assert_positive_real_number(width)
        assert_positive_real_number(height)
        self._width = width
        self._height = height

        # Limits
        self._xlim = x_c - width / 2, x_c + width / 2
        self._ylim = y_c - height / 2, y_c + height / 2

    # Properties
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    # Methods
    def _compute_mask(self, image_meshgrid: ImageAxes2D) -> np.ndarray:

        x_mg, y_mg = image_meshgrid
        (x_min, x_max), (y_min, y_max) = self._xlim, self._ylim

        rect_x = np.logical_and(x_mg >= x_min, x_mg <= x_max)
        rect_y = np.logical_and(y_mg >= y_min, y_mg <= y_max)

        mask = np.logical_and(rect_x, rect_y)

        return mask

    def extract_cartesian_samples(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> np.ndarray:

        # Check images w.r.t. image axes
        _assert_compat_images_and_axes(images=images, image_axes=image_axes)

        # Get cartesian image slicer
        image_slicer = self.get_cartesian_image_slicer(image_axes=image_axes)

        # Broadcast
        slicer = Ellipsis, *image_slicer

        return np.copy(images[slicer])

    def get_cartesian_image_slicer(
            self, image_axes: ImageAxes2D
    ) -> Sequence[slice]:

        x_axis, y_axis = image_axes
        x_min, x_max = self._xlim
        y_min, y_max = self._ylim
        x_ind = np.where(np.logical_and(x_axis >= x_min, x_axis <= x_max))[0]
        y_ind = np.where(np.logical_and(y_axis >= y_min, y_axis <= y_max))[0]
        slicer = slice(x_ind[0], x_ind[-1] + 1), slice(y_ind[0], y_ind[-1] + 1)

        return slicer

    def _draw_boundaries(self, ax: Axes):

        # Extract properties
        width, height = self._width, self._height
        center = self.center
        color = self._color

        # Bottom left rectangle coordinate
        bot_left = np.array(center) - np.array([width, height]) / 2
        xy = tuple(bot_left)

        # Add patch
        rect = mpatches.Rectangle(
            xy=xy, width=width, height=height, color=color, fill=False
        )
        ax.add_artist(rect)


def _assert_compat_images_and_axes(
    images: np.ndarray, image_axes: Sequence[np.ndarray]
) -> None:

    image_ndim = len(image_axes)
    image_shape = tuple([ax.size for ax in image_axes])
    if images.ndim < image_ndim:
        raise ValueError("Incompatible image and image axes dimensions")
    #   Batched case assumes last dimensions to be image axes
    else:
        if images.shape[-image_ndim:] != image_shape:
            err_msg = "Incompatible image and axes shapes."
            err_msg += " {} != {}".format(images.shape, image_shape)
            raise ValueError(err_msg)