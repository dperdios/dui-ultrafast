import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Sequence, Dict

from utils.metrics.resolution import compute_fwhm_line
from utils.metrics.signals import compute_contrast, compute_auto_cov
from utils.types import Real, assert_real_number, ImageAxes2D
from utils.metrics.rois import RectangularROI, CircularROI


class NumericalPhantom:

    def __init__(self):

        # Characteristic wavelength considered (to define some ROIs)
        wavelength = 1540. / (250e6 / 48)

        # Phantom zones
        #   Block
        bk_center = -5e-3, 20e-3
        bk_width, bk_height = 20e-3, 20e-3
        bk_echo = 10
        bk_zone = RectangularROI(
            center=bk_center, width=bk_width, height=bk_height
        )
        self._bk_echo = bk_echo
        self._bk_zone = bk_zone
        #   Circular inclusion
        ci_center = bk_center
        ci_radius = 8.5e-3 / 2
        ci_echo = -26
        ci_zone = CircularROI(center=ci_center, radius=ci_radius)
        self._ci_echo = ci_echo
        self._ci_zone = ci_zone
        #   Log-linear gradient
        lg_center = 0, 5e-2
        lg_width = 43.93e-3  # GE9LD "exact" width
        lg_height = 10e-3
        lg_echo = [30, -50]
        lg_zone = RectangularROI(
            center=lg_center, width=lg_width, height=lg_height
        )
        self._lg_echo = lg_echo
        self._lg_zone = lg_zone
        #   Field points
        fp_pos_z = 1e-3 * np.array([10, 20, 30, 40])
        fp_pos_x = 12.5e-3 * np.ones(fp_pos_z.size)
        fp_echo = 30 * np.ones(fp_pos_x.size)
        fp_pos = np.stack([fp_pos_x, fp_pos_z])
        self._fp_echo = fp_echo
        self._fp_pos = fp_pos

        # Metrics ROIs
        #   Margin ("resolution")
        res_margin = 0.75e-3
        #   Contrast: inner region
        ir_radius = ci_radius - res_margin
        ir_color = 'C1'
        ir_roi = CircularROI(center=ci_center, radius=ir_radius, color=ir_color)
        self._ir_roi = ir_roi
        #   Contrast: outer region
        or_roi_w = 0.5 * np.pi / 2 * ir_roi.radius
        or_roi_h = 2 * ir_roi.radius
        or_roi_dx = ci_radius + res_margin + or_roi_w / 2
        or_roi_1_c = ir_roi.center[0] - or_roi_dx, ir_roi.center[1]
        or_roi_2_c = ir_roi.center[0] + or_roi_dx, ir_roi.center[1]
        or_color = 'C0'
        or_roi_1 = RectangularROI(
            center=or_roi_1_c, width=or_roi_w, height=or_roi_h, color=or_color
        )
        or_roi_2 = RectangularROI(
            center=or_roi_2_c, width=or_roi_w, height=or_roi_h, color=or_color
        )
        self._or_roi_1 = or_roi_1
        self._or_roi_2 = or_roi_2
        #   Linear gradient
        lg_color = 'C6'
        lg_roi = RectangularROI(
            center=lg_center, width=lg_width, height=lg_height, color=lg_color
        )
        self._lg_roi = lg_roi
        self._lg_color = lg_color
        #   Speckle region
        sr_width = sr_height = 10 * wavelength
        sr_center = 0e-3, 27e-3
        sr_color = 'C3'
        sr_roi = RectangularROI(
            center=sr_center, width=sr_width, height=sr_height, color=sr_color
        )
        self._sr_roi = sr_roi
        #   Grating-lobe region
        gl_center = 17e-3, 15e-3
        gl_width = gl_height = 6e-3
        gl_color = 'C8'
        gl_roi = RectangularROI(
            center=gl_center, width=gl_width, height=gl_height, color=gl_color
        )
        self._gl_roi = gl_roi
        #   Side-lobe region
        sl_center = 12.5e-3, 42.5e-3
        sl_width = 15e-3
        sl_height = 3e-3
        sl_color = 'C2'
        sl_roi = RectangularROI(
            center=sl_center, width=sl_width, height=sl_height, color=sl_color
        )
        self._sl_roi = sl_roi
        #   Edge-wave region
        ew_center = -5e-3, 32.5e-3
        ew_width = 16e-3
        ew_height = 3e-3
        ew_color = 'C4'
        ew_roi = RectangularROI(
            center=ew_center, width=ew_width, height=ew_height, color=ew_color
        )
        self._ew_roi = ew_roi
        #   Field points
        fp_roi_w = fp_roi_h = 2 * wavelength
        fp_color = 'C9'
        fp_kw = dict(width=fp_roi_w, height=fp_roi_h, color=fp_color)
        fp_roi_seq = [
            RectangularROI(center=(fp_x, fp_z), **fp_kw)
            for fp_x, fp_z in zip(fp_pos_x, fp_pos_z)
        ]
        self._fp_roi_seq = fp_roi_seq

    # Properties

    # Methods
    def compute_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict[str, Dict]:

        metrics_kwargs = {'images': images, 'image_axes': image_axes}

        # Inclusion metrics
        incl_metrics = self._compute_inclusion_metrics(**metrics_kwargs)

        # Artifact metrics
        artifact_metrics = self._compute_artifact_metrics(**metrics_kwargs)

        # Speckle metrics
        speckle_metrics = self._compute_speckle_metrics(**metrics_kwargs)

        # Resolution: FWHM + oversampled cross lines
        res_metrics = self._compute_field_point_metrics(**metrics_kwargs)

        # Gradient region
        grad_metrics = self._compute_gradient_metrics(**metrics_kwargs)

        # Create output dictionary
        metrics_dict = {
            'inclusion': incl_metrics,
            'resolution': res_metrics,
            'gradient': grad_metrics,
            'speckle': speckle_metrics,
            'artifacts': artifact_metrics
        }
        return metrics_dict

    def _compute_inclusion_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Define output metrics dict
        metrics = {'contrast': None}

        # Get ROIs
        ir_roi = self._ir_roi
        or_roi_1 = self._or_roi_1
        or_roi_2 = self._or_roi_2

        # Extract samples
        sig_ir = ir_roi.extract_samples(images=images, image_axes=image_axes)
        sig_or_1 = or_roi_1.extract_samples(images=images, image_axes=image_axes)
        sig_or_2 = or_roi_2.extract_samples(images=images, image_axes=image_axes)
        sig_or = np.concatenate([sig_or_1, sig_or_2], axis=-1)

        # Compute contrast
        contrast = compute_contrast(sig1=sig_ir, sig2=sig_or, axis=-1)

        # Store metrics in output dict
        metrics['contrast'] = contrast

        return metrics

    def _compute_speckle_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Define output metrics dict
        #   1st order statistics: SNR
        #   2nd order statistics: speckle spot size (FWHM of ACF)
        metrics = {'snr': None, 'fwhm': None}

        # Get ROI
        sr_roi = self._sr_roi

        # Extract cartesian samples
        samples = sr_roi.extract_cartesian_samples(
            images=images, image_axes=image_axes
        )
        samples_slicer = sr_roi.get_cartesian_image_slicer(
            image_axes=image_axes
        )

        # Define computation axes
        axes = -2, -1

        # SNR (defined as the reciprocal of coefficient of variation)
        mean = np.mean(samples, axis=axes)
        std = np.std(samples, axis=axes)
        snr = np.squeeze(mean / std)

        # Autocovariance function (ACF)
        acf = compute_auto_cov(x=samples, axes=axes)

        # FWHM of the ACF
        fwhm_batch_seq = []
        ac_axes = tuple([image_axes[ax][samples_slicer[ax]] for ax in axes])

        for ac in acf:
            fwhm_seq = []

            for ax_ind in axes:
                ac_ax = ac_axes[ax_ind]
                line_slicer = [np.s_[s // 2] for s in ac.shape]
                line_slicer[ax_ind] = slice(None)
                ac_line = ac[tuple(line_slicer)]

                # Compute FWHM
                fwhm = compute_fwhm_line(x=ac_ax, y=ac_line)
                fwhm_seq.append(fwhm)

            fwhm_batch_seq.append(fwhm_seq)

        # fwhm = np.array(np.array(fwhm_batch_seq).T, order='C')
        fwhm = np.array(fwhm_batch_seq)

        # Store metrics in output dict
        metrics['snr'] = snr
        metrics['fwhm'] = fwhm

        return metrics

    def _compute_artifact_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Define output metrics dict
        metrics = {'gl': None, 'sl': None, 'ew': None}

        # Get ROIs
        gl_roi = self._gl_roi
        sl_roi = self._sl_roi
        ew_roi = self._ew_roi

        # Compute mean amplitudes
        gl_sig = gl_roi.extract_samples(images=images, image_axes=image_axes)
        sl_sig = sl_roi.extract_samples(images=images, image_axes=image_axes)
        ew_sig = ew_roi.extract_samples(images=images, image_axes=image_axes)
        gl_mean = np.mean(gl_sig, axis=-1)
        sl_mean = np.mean(sl_sig, axis=-1)
        ew_mean = np.mean(ew_sig, axis=-1)

        # Store metrics in output dict
        metrics['gl'] = gl_mean
        metrics['sl'] = sl_mean
        metrics['ew'] = ew_mean

        return metrics

    def _compute_gradient_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Define output metrics dict
        metrics = {'mean': None, 'axis': None}

        # Get ROI
        lq_roi = self._lg_roi

        # Extract samples
        samples = lq_roi.extract_cartesian_samples(
            images=images, image_axes=image_axes
        )

        # Compute mean amplitude along axial dimension
        mean = np.mean(samples, axis=-1)

        # Store metrics in output dict
        metrics['mean'] = mean
        metrics['axis'] = image_axes[0]

        return metrics

    def _compute_field_point_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Extract properties
        fp_roi_seq = self._fp_roi_seq

        # Input settings
        xaxis, zaxis = image_axes
        bbox_samples = 201  # results in ~2.96 μm, i.e, λ/100
        num_x = num_z = bbox_samples

        # Compute FWHM of each field point for each image
        metrics = {}
        for ii, fp_roi in enumerate(fp_roi_seq):

            # Key
            k = f'p{ii}'

            # Get bounding box
            bbox_sizes = fp_roi.width, fp_roi.height

            # Compute bounding box (including considered resolution)
            fp_x, fp_z = fp_roi.center
            bbox_x, bbox_z = bbox_sizes
            xx_min, xx_max = fp_x - bbox_x / 2, fp_x + bbox_x / 2
            zz_min, zz_max = fp_z - bbox_z / 2, fp_z + bbox_z / 2

            # Create oversampled grid lines
            xx = np.linspace(xx_min, xx_max, num=num_x)
            zz = np.linspace(zz_min, zz_max, num=num_z)

            # Instantiate empty lists
            line_list = []
            fwhm_list = []

            for im in images:
                # Create spline (degree 3) interpolator
                interp2d = RectBivariateSpline(x=xaxis, y=zaxis, z=im)

                # Perform interpolation on corresponding grid (-> surface)
                srf = interp2d(x=xx, y=zz, grid=True)

                # Find maximum values
                srf_max = np.max(srf)  # assumed strictly positive (env, bm)
                srf_max_x_ind, srf_max_y_ind = np.where(srf == srf_max)

                # Normalization
                srf /= srf_max

                # Extract lines including surface peak
                line_x = srf[:, srf_max_y_ind].squeeze()
                line_z = srf[srf_max_x_ind].squeeze()

                # Compute FWHM
                fwhm_x = compute_fwhm_line(xx, line_x)
                fwhm_z = compute_fwhm_line(zz, line_z)

                # Store in lists
                fwhm_list.append(np.array([fwhm_x, fwhm_z]))
                line_list.append(np.array([line_x, line_z]))

            # Store metrics in output dict
            point_metrics = {
                'axes': (xx, zz),
                'lines': np.array(line_list),
                'fwhm': np.array(fwhm_list),
            }
            metrics[k] = point_metrics

        return metrics

    def draw_geometry(
            self,
            ax: plt.Axes,
            cmap: str = 'gray',
            vmin: Real = -62,
            vmax: Real = 36,
            extent: Sequence[Real] = None,
    ) -> None:

        def _get_mapped_color(val: Real, vmin: Real, vmax: Real) -> str:
            assert_real_number(val)
            assert_real_number(vmin)
            assert_real_number(vmax)
            clr = np.clip((val - vmin) / (vmax - vmin), a_min=0, a_max=1)
            return str(clr)

        # TODO: Check inputs
        assert_real_number(vmin)
        assert_real_number(vmax)
        if not isinstance(cmap, str):
            raise TypeError("Must be a string.")
        cm_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}

        # Limits
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_aspect(1)
        zmin, zmax = ymax, ymin

        # Background
        bg_xy = xmin, zmin  # axes reverted (bottom left)
        bg_echo = -np.inf
        bg_color = _get_mapped_color(val=bg_echo, vmin=vmin, vmax=vmax)
        bg_patch = mpatches.Rectangle(
            xy=bg_xy, width=xmax - xmin, height=zmax - zmin,
            color=bg_color,
            zorder=0
        )
        ax.add_artist(bg_patch)

        # Block
        bk_zone = self._bk_zone
        bk_echo = self._bk_echo
        bk_center = bk_zone.center
        bk_width, bk_height = bk_zone.width, bk_zone.height
        bk_xy = np.array(bk_center) - np.array([bk_width, bk_height]) / 2
        bk_xy = tuple(bk_xy)
        bk_color = _get_mapped_color(val=bk_echo, vmin=vmin, vmax=vmax)
        bk_patch = mpatches.Rectangle(
            xy=bk_xy, width=bk_width, height=bk_height, color=bk_color,
        )
        ax.add_artist(bk_patch)

        # Circular inclusion
        ci_zone = self._ci_zone
        ci_echo = self._ci_echo
        ci_color = _get_mapped_color(val=ci_echo, vmin=vmin, vmax=vmax)
        ci_patch = mpatches.Circle(
            xy=ci_zone.center, radius=ci_zone.radius, color=ci_color,
        )
        ax.add_artist(ci_patch)

        # Linear gradient
        lg_zone = self._lg_zone
        lg_echo = self._lg_echo
        lg_center = np.array(lg_zone.center)
        lg_width, lg_height = lg_zone.width, lg_zone.height
        lg_xmin, lg_ymin = lg_center - 0.5 * np.array([lg_width, lg_height])
        lg_xmax, lg_ymax = lg_center + 0.5 * np.array([lg_width, lg_height])
        lg_qmesh = ax.pcolormesh(
            [lg_xmin, lg_xmax], [lg_ymin, lg_ymax], [lg_echo, lg_echo],
            shading='gouraud', **cm_kwargs
        )

        # Field points
        fp_echo = self._fp_echo
        fp_pos_x, fp_pos_z = self._fp_pos
        br_s = 0.4
        fp_coll = ax.scatter(
            x=fp_pos_x, y=fp_pos_z, s=br_s, c=fp_echo, linewidths=0, **cm_kwargs
        )

    def draw_metric_rois(self, ax: plt.Axes) -> None:

        # Get metrics ROIs
        roi_seq = [
            self._ir_roi,
            self._or_roi_1, self._or_roi_2,
            self._lg_roi,
            self._sr_roi,
            self._gl_roi, self._sl_roi, self._ew_roi
        ]

        # Draw boundaries
        for roi in roi_seq:
            roi.draw_boundaries(ax=ax)

    def draw_metric_labels(self, ax: plt.Axes) -> None:

        # Annotation labels
        sl_label = r'$\Omega_{\mathrm{SL}}$'
        gl_label = r'$\Omega_{\mathrm{GL}}$'
        ew_label = r'$\Omega_{\mathrm{EW}}$'
        ir_label = r'$\Omega_{\mathrm{I}}$'
        or_label = r'$\Omega_{\mathrm{B}}$'
        sr_label = r'$\Omega_{\mathrm{S}}$'
        lg_label = r'$\Omega_{\mathrm{LG}}$'

        # Annotation settings
        ann_offset_pt = 1
        # ann_offset_pt = 0
        # ann_offset_pt = 3
        base_ann_kw = {
            'xycoords': 'data',
            'textcoords': 'offset points',
        }

        # Contrast: outer region
        or_roi_1 = self._or_roi_1
        or_roi_2 = self._or_roi_2
        or_ann_1_kw = {
            'xy': (
            or_roi_1.center[0], or_roi_1.center[1] - or_roi_1.height / 2),
            'xytext': (0.0, ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'color': or_roi_1.color,
            **base_ann_kw
        }
        or_ann_1 = ax.annotate(text=or_label, **or_ann_1_kw)
        or_ann_2_kw = {**or_ann_1_kw}
        or_ann_2_kw['xy'] = (
            or_roi_2.center[0], or_roi_2.center[1] - or_roi_2.height / 2
        )
        or_ann_2_kw['color'] = or_roi_2.color
        or_ann_2 = ax.annotate(text=or_label, **or_ann_2_kw)

        # Contrast: inner region
        ir_roi = self._ir_roi
        ir_ann_kw = {
            'xy': (ir_roi.center[0], ir_roi.center[1]),
            'xytext': (0.0, 0.0),
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'color': ir_roi.color,
            **base_ann_kw
        }
        ir_ann = ax.annotate(text=ir_label, **ir_ann_kw)

        # Speckle
        sr_roi = self._sr_roi
        sr_ann_kw = {
            'xy': (sr_roi.center[0] - sr_roi.width / 2, sr_roi.center[1]),
            'xytext': (-ann_offset_pt, 0.0),
            'horizontalalignment': 'right',
            'verticalalignment': 'center',
            'color': sr_roi.color,
            **base_ann_kw
        }
        sr_ann = ax.annotate(text=sr_label, **sr_ann_kw)

        # Linear gradient
        lg_roi = self._lg_roi
        lg_ann_kw = {
            'xy': (lg_roi.center[0], lg_roi.center[1] + lg_roi.height / 2),
            'xytext': (0.0, -ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'top',
            'color': lg_roi.color,
            **base_ann_kw
        }
        lg_ann = ax.annotate(text=lg_label, **lg_ann_kw)

        # GL
        gl_roi = self._gl_roi
        gl_ann_kw = {
            'xy': (gl_roi.center[0], gl_roi.center[1] + gl_roi.height / 2),
            'xytext': (0.0, -ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'top',
            'color': gl_roi.color,
            **base_ann_kw
        }
        gl_ann = ax.annotate(text=gl_label, **gl_ann_kw)

        # SL
        sl_roi = self._sl_roi
        sl_ann_kw = {
            'xy': (sl_roi.center[0] - sl_roi.width / 2, sl_roi.center[1]),
            'xytext': (-ann_offset_pt, 0.0),
            'horizontalalignment': 'right',
            'verticalalignment': 'center',
            'color': sl_roi.color,
            **base_ann_kw
        }
        sl_ann = ax.annotate(text=sl_label, **sl_ann_kw)

        # EW
        ew_roi = self._ew_roi
        ew_ann_kw = {
            'xy': (ew_roi.center[0], ew_roi.center[1] + ew_roi.height / 2),
            'xytext': (0.0, -ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'top',
            'color': ew_roi.color,
            **base_ann_kw
        }
        ew_ann = ax.annotate(text=ew_label, **ew_ann_kw)

        # Field points
        fp_roi_seq = self._fp_roi_seq
        fp_ann_seq = []
        for ii, fp_roi in enumerate(fp_roi_seq):
            # Boundary
            fp_roi.draw_boundaries(ax=ax)

            # Label
            fp_label = f'$p_{{{ii}}}$'
            fp_xy = (
                fp_roi.center[0] - fp_roi.width / 2,
                fp_roi.center[1] - fp_roi.height / 2
            )
            fp_ann_kw = {
                'xy': fp_xy,
                'xytext': (0.0, 0.0),
                'horizontalalignment': 'right',
                'verticalalignment': 'bottom',
                'color': fp_roi.color,
                **base_ann_kw
            }
            fp_ann_seq.append(ax.annotate(text=fp_label, **fp_ann_kw))
