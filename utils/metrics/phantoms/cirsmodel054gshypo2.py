import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict

from utils.metrics.resolution import compute_fwhm_line
from utils.metrics.signals import compute_contrast, compute_auto_cov
from utils.types import Real, assert_real_number, ImageAxes2D
from utils.metrics.rois import RectangularROI, CircularROI


class CIRSModel054GSHypo2:

    def __init__(self):

        # Characteristic wavelength considered (to define some ROIs)
        wavelength = 1540. / (250e6 / 48)

        # Phantom zones
        #   Circular inclusions: global
        ci_radius = 8e-3 / 2
        ci_dx = 12e-3
        ci_center_z = 39.5e-3  # Note: theoretical: 40e-3
        #   Circular inclusions: anechoic (A)
        ci_a_center_x = -11.8e-3
        ci_a_center = ci_a_center_x, ci_center_z
        ci_a_zone = CircularROI(center=ci_a_center, radius=ci_radius)
        self._ci_a_zone = ci_a_zone
        #   Circular inclusions: -6 dB (B)
        ci_b_center_x = ci_a_center_x + ci_dx
        ci_b_center = ci_b_center_x, ci_center_z
        ci_b_zone = CircularROI(center=ci_b_center, radius=ci_radius)
        self._ci_b_zone = ci_b_zone
        #   Circular inclusions: -3 dB (C)
        ci_c_center_x = ci_b_center_x + ci_dx
        ci_c_center = ci_c_center_x, ci_center_z
        ci_c_zone = CircularROI(center=ci_c_center, radius=ci_radius)
        self._ci_c_zone = ci_c_zone

        # Metrics ROIs
        #   Margin ("resolution")
        res_margin = 1e-3
        #   Circular inclusions: global
        ir_radius = ci_radius - res_margin
        #   Circular inclusions: anechoic (A)
        ci_a_clr = 'C0'
        ci_a_roi = CircularROI(
            center=ci_a_center, radius=ir_radius, color=ci_a_clr
        )
        self._ci_a_roi = ci_a_roi
        #   Circular inclusions: -6 dB (B)
        ci_b_clr = 'C1'
        ci_b_roi = CircularROI(
            center=ci_b_center, radius=ir_radius, color=ci_b_clr
        )
        self._ci_b_roi = ci_b_roi
        #   Circular inclusions: -3 dB (C)
        ci_c_clr = 'C2'
        ci_c_roi = CircularROI(
            center=ci_c_center, radius=ir_radius, color=ci_c_clr
        )
        self._ci_c_roi = ci_c_roi
        #   Contrast: outer regions (reference background)
        or_roi_w = 0.5 * np.pi / 2 * ci_a_roi.radius
        or_roi_h = 2 * ci_a_roi.radius
        or_color = 'C8'
        or_roi_dx = ci_radius + res_margin + or_roi_w / 2
        or_roi_1_c_x = 0.5 * (ci_a_roi.center[0] + ci_b_roi.center[0])
        or_roi_1_c_z = 0.5 * (ci_a_roi.center[1] + ci_b_roi.center[1])
        or_roi_1_c = or_roi_1_c_x, or_roi_1_c_z
        or_roi_2_c_x = 0.5 * (ci_c_roi.center[0] + ci_b_roi.center[0])
        or_roi_2_c_z = 0.5 * (ci_c_roi.center[1] + ci_b_roi.center[1])
        or_roi_2_c = or_roi_2_c_x, or_roi_2_c_z
        or_roi_1 = RectangularROI(
            center=or_roi_1_c, width=or_roi_w, height=or_roi_h, color=or_color
        )
        or_roi_2 = RectangularROI(
            center=or_roi_2_c, width=or_roi_w, height=or_roi_h, color=or_color
        )
        self._or_roi_1 = or_roi_1
        self._or_roi_2 = or_roi_2
        #   Speckle region
        sr_width = sr_height = 10 * wavelength
        sr_center = 0e-3, 27e-3
        sr_color = 'C3'
        sr_roi = RectangularROI(
            center=sr_center, width=sr_width, height=sr_height, color=sr_color
        )
        self._sr_roi = sr_roi

    # Properties

    # Methods
    def compute_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict[str, Dict]:

        metrics_kwargs = {'images': images, 'image_axes': image_axes}

        # Inclusion metrics
        incl_metrics = self._compute_inclusion_metrics(**metrics_kwargs)

        # Speckle metrics
        speckle_metrics = self._compute_speckle_metrics(**metrics_kwargs)

        # Create output dictionary
        metrics_dict = {
            'inclusions': incl_metrics,
            'speckle': speckle_metrics,
        }
        return metrics_dict

    def _compute_inclusion_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # Get ROIs
        ci_a_roi = self._ci_a_roi
        ci_b_roi = self._ci_b_roi
        ci_c_roi = self._ci_c_roi
        ci_rois = ci_a_roi, ci_b_roi, ci_c_roi
        or_roi_1 = self._or_roi_1
        or_roi_2 = self._or_roi_2

        # Define output metrics dict
        metrics = {'a': None, 'b': None, 'c': None}

        # Extract background samples
        sig_or_1 = or_roi_1.extract_samples(images=images, image_axes=image_axes)
        sig_or_2 = or_roi_2.extract_samples(images=images, image_axes=image_axes)
        sig_or = np.concatenate([sig_or_1, sig_or_2], axis=-1)

        # Compute contrast
        for k, roi in zip(metrics.keys(), ci_rois):
            # Extract inclusion samples
            sig_ir = roi.extract_samples(images=images, image_axes=image_axes)

            # Contrast
            contrast = compute_contrast(sig1=sig_ir, sig2=sig_or, axis=-1)

            # Store metrics in output dict
            metrics[k] = {'contrast': contrast}
            # metrics[k] = {'contrast': contrast.tolist()}  # JSON serializable

        return metrics

    def _compute_speckle_metrics(
            self, images: np.ndarray, image_axes: ImageAxes2D
    ) -> Dict:

        # TODO: share code with numerical phantom

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
        snr = mean / std

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

    def draw_metric_rois(self, ax: plt.Axes) -> None:

        # Get metric ROIs
        roi_seq = [
            self._ci_a_roi,
            self._ci_b_roi,
            self._ci_c_roi,
            self._or_roi_1, self._or_roi_2,
            self._sr_roi,
        ]

        # Draw ROI boundaries
        for roi in roi_seq:
            roi.draw_boundaries(ax=ax)

        # Draw zone boundaries
        ci_a_roi = self._ci_a_roi
        ci_b_roi = self._ci_b_roi
        ci_c_roi = self._ci_c_roi
        ci_a_zone = self._ci_a_zone
        ci_radius = ci_a_zone.radius
        for roi in [ci_a_roi, ci_b_roi, ci_c_roi]:
            # Add patch
            circle = mpatches.Circle(
                xy=roi.center, radius=ci_radius, edgecolor=roi.color,
                fill=False, linestyle='--'
            )
            ax.add_artist(circle)

    def draw_metric_labels(self, ax: plt.Axes) -> None:

        # Annotation labels
        ci_a_label = r'$\Omega_{\mathrm{A}}$'
        ci_b_label = r'$\Omega_{\mathrm{B}}$'
        ci_c_label = r'$\Omega_{\mathrm{C}}$'
        or_label = r'$\Omega_{\mathrm{D}}$'
        sr_label = r'$\Omega_{\mathrm{S}}$'

        # Annotation settings
        ann_offset_pt = 1
        base_ann_kw = {
            'xycoords': 'data',
            'textcoords': 'offset points',
        }

        # Contrast: outer regions
        or_roi_1 = self._or_roi_1
        or_roi_2 = self._or_roi_2
        or_ann_1_kw = {
            'xy': (
                or_roi_1.center[0], or_roi_1.center[1] + or_roi_1.height / 2
            ),
            'xytext': (0.0, -ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'top',
            'color': or_roi_1.color,
            **base_ann_kw
        }
        or_ann_1 = ax.annotate(text=or_label, **or_ann_1_kw)
        or_ann_2_kw = {**or_ann_1_kw}
        or_ann_2_kw['xy'] = (
            or_roi_2.center[0], or_roi_2.center[1] + or_roi_2.height / 2
        )
        or_ann_2_kw['color'] = or_roi_2.color
        or_ann_2 = ax.annotate(text=or_label, **or_ann_2_kw)

        # Circular inclusions
        ci_radius = self._ci_a_zone.radius
        ci_a_roi = self._ci_a_roi
        ci_b_roi = self._ci_b_roi
        ci_c_roi = self._ci_c_roi
        ci_roi_seq = [ci_a_roi, ci_b_roi, ci_c_roi]
        ci_lbl_seq = [ci_a_label, ci_b_label, ci_c_label]
        for roi, lbl in zip(ci_roi_seq, ci_lbl_seq):
            ci_ann_kw = {
                'xy': (roi.center[0], roi.center[1] + ci_radius),
                'xytext': (0.0, -ann_offset_pt),
                'horizontalalignment': 'center',
                'verticalalignment': 'top',
                'color': roi.color,
                **base_ann_kw
            }
            ci_ann = ax.annotate(text=lbl, **ci_ann_kw)

        # Speckle
        sr_roi = self._sr_roi
        sr_ann_kw = {
            'xy': (sr_roi.center[0], sr_roi.center[1] - sr_roi.height / 2),
            'xytext': (0.0, ann_offset_pt),
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'color': sr_roi.color,
            **base_ann_kw
        }
        sr_ann = ax.annotate(text=sr_label, **sr_ann_kw)
