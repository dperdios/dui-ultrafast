#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils.plot as pplot
from utils.datasets import load_and_preprocess_images
from utils.training.run_configs import get_experimental_test_managers
from utils.metrics.phantoms.cirsmodel054gshypo2 import CIRSModel054GSHypo2

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
datasets_dir = Path('./data/datasets/test')
group_key = 'images'

# Trained model managers
trained_models_dir = Path('./data/trained-models')
model_managers = get_experimental_test_managers(trained_models_dir)

# Configurations
config_pht = {
    'dset_path': datasets_dir.joinpath(
        '20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.hdf5'
    ),
    'vmin': -42, 'vmax': 36,
    'zlim': (15e-3, 50e-3),
    'samples_slicer': slice(0, 1)
}
config_caro = {
    'dset_path': datasets_dir.joinpath(
        '20200527-ge9ld-experimental-test-set-carotid-long.hdf5'
    ),
    'vmin': -14, 'vmax': 36,
    'zlim': (5e-3, 40e-3),
    'samples_slicer': slice(45, 46)
}
configs = {'phantom': config_pht, 'carotid': config_caro}

# Results dict (to store data required for figure)
results_dict = {k: {} for k in configs.keys()}

# -----------------------------------------------------------------------------
# Load images: inputs and targets
# -----------------------------------------------------------------------------
image_keys = 'lq', 'hq'
input_signal = 'iq'
output_signal = 'bm'

for cfg, res in zip(configs.values(), results_dict.values()):

    # dset_path = dset_path.resolve(strict=True)
    dset_path = cfg['dset_path'].resolve(strict=True)

    # Load images
    for k in image_keys:
        # Build dataset key
        dset_name = '/'.join([group_key, k])
        print(f"Loading '{dset_name}' from '{dset_path}'")

        # Load images as envelope to compute metrics
        images, image_axes = load_and_preprocess_images(
            path=dset_path,
            name=dset_name,
            input_signal=input_signal,
            input_factor='0db',
            output_signal=output_signal,
            samples_slicer=cfg['samples_slicer'],
        )

        # Crop
        xaxis, zaxis = image_axes
        zmin_crop, zmax_crop = cfg['zlim']
        zmin_ind = np.where(zaxis < zmin_crop)[0][-1]
        zmax_ind = np.where(zaxis > zmax_crop)[0][0]
        slice_crop_z = slice(zmin_ind, zmax_ind + 1)
        slice_crop = Ellipsis, slice_crop_z
        zaxis = zaxis[slice_crop_z]
        image_axes = xaxis, zaxis
        images = images[slice_crop]

        # Store
        fig_dict = {
            'bmode': images[0],
            'bmode_axes': image_axes,
            'vmin': cfg['vmin'],
            'vmax': cfg['vmax'],
        }
        res[k] = {'figures': fig_dict}

# -----------------------------------------------------------------------------
# Load images: predictions
# -----------------------------------------------------------------------------
for cfg, res in zip(configs.values(), results_dict.values()):

    dset_path = cfg['dset_path'].resolve(strict=True)

    predictions_dir = datasets_dir
    pred_suffix = 'predictions'
    pred_path = predictions_dir.joinpath(dset_path.stem + '-' + pred_suffix)
    pred_path = pred_path.with_suffix(dset_path.suffix)
    pred_path = pred_path.resolve(strict=True)

    # Load images
    pred_dict = {}
    dset_path = pred_path
    for k, mgr in model_managers.items():
        # Build dataset key
        dset_name = '/'.join([group_key, k])
        print(f"Loading '{dset_name}' from '{dset_path}'")

        # Convert prediction signal (CNN output) to envelope
        input_signal = mgr.run_config.mapping_config.output_signal
        images, image_axes = load_and_preprocess_images(
            path=dset_path,
            name=dset_name,
            input_signal=input_signal,
            output_signal=output_signal,
            samples_slicer=cfg['samples_slicer']
        )

        # Crop
        xaxis, zaxis = image_axes
        zmin_crop, zmax_crop = cfg['zlim']
        zmin_ind = np.where(zaxis < zmin_crop)[0][-1]
        zmax_ind = np.where(zaxis > zmax_crop)[0][0]
        slice_crop_z = slice(zmin_ind, zmax_ind + 1)
        slice_crop = Ellipsis, slice_crop_z
        zaxis = zaxis[slice_crop_z]
        image_axes = xaxis, zaxis
        images = images[slice_crop]

        # Store
        fig_dict = {
            'bmode': images[0],
            'bmode_axes': image_axes,
            'vmin': cfg['vmin'],
            'vmax': cfg['vmax'],
        }
        res[k] = {'figures': fig_dict}

# -----------------------------------------------------------------------------
# Metrics phantom
# -----------------------------------------------------------------------------
phantom = CIRSModel054GSHypo2()

# -----------------------------------------------------------------------------
# Figure: sample B-mode images
# -----------------------------------------------------------------------------
keys = 'lq', 'mslae-16', 'hq'

# Axes labels and titles
label_seq = ["LQ (CNN Input)"] + [k.upper() for k in keys[1:]]
y_label = "Axial Dimension (mm)"
x_label = "Lateral Dimension (mm)"
pht_title = "Phantom Geometry"

# Axes ticks
x_ticks = [-0.02, -0.01, 0, 0.01, 0.02]

# Figure kwargs
figsize = 12.8, 7.66
fig_kwargs = {'figsize': figsize, 'constrained_layout': True}

# -----------------------------------------------------------------------------
# Figure: sample B-mode images
# -----------------------------------------------------------------------------
# Create figure and image grid
fig, axes = plt.subplots(
    nrows=2, ncols=3, sharex='row', sharey='row', **fig_kwargs)

# --------
# In vitro phantom
# --------
# Axes pointers
ax_seq = axes[0]

res_dict = results_dict['phantom']
bm_seq = [res_dict[k]['figures']['bmode'] for k in keys]
xaxis, zaxis = res_dict['lq']['figures']['bmode_axes']

# Limits and extent
xmin, xmax = xaxis[0], xaxis[-1]
zmin, zmax = zaxis[0], zaxis[-1]
dx = (xmax - xmin) / (xaxis.size - 1)
dz = (zmax - zmin) / (zaxis.size - 1)
extent = [xmin - dx / 2, xmax + dx / 2, zmax + dz / 2, zmin - dz / 2]

# Axes ticks
y_ticks = [0.02, 0.03, 0.04, 0.05]

# Settings
vmin = res_dict['lq']['figures']['vmin']
vmax = res_dict['lq']['figures']['vmax']
db_range = vmax - vmin
cmap = 'gray'
cm_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
im_kwargs = {**cm_kwargs, 'extent': extent}

# B-mode images
for ax, bm, lbl in zip(ax_seq, bm_seq, label_seq):
    bm: np.ndarray
    ax: plt.Axes
    ax.imshow(bm.T, **im_kwargs)
    ax.set_title(lbl)
    # Metrics ROIs
    phantom.draw_metric_rois(ax=ax)
    phantom.draw_metric_labels(ax=ax)

# Ticks
for ax in ax_seq:
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)

# --------
# Carotid
# --------
# Axes pointers
ax_seq = axes[1]

res_dict = results_dict['carotid']
bm_seq = [res_dict[k]['figures']['bmode'] for k in keys]
xaxis, zaxis = res_dict['lq']['figures']['bmode_axes']

# Limits and extent
xmin, xmax = xaxis[0], xaxis[-1]
zmin, zmax = zaxis[0], zaxis[-1]
dx = (xmax - xmin) / (xaxis.size - 1)
dz = (zmax - zmin) / (zaxis.size - 1)
extent = [xmin - dx / 2, xmax + dx / 2, zmax + dz / 2, zmin - dz / 2]

# Axes ticks
y_ticks = [0.01, 0.02, 0.03, 0.04]

# Settings
vmin = res_dict['lq']['figures']['vmin']
vmax = res_dict['lq']['figures']['vmax']
db_range = vmax - vmin
cmap = 'gray'
cm_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
im_kwargs = {**cm_kwargs, 'extent': extent}

# B-mode images
for ax, bm, lbl in zip(ax_seq, bm_seq, label_seq):
    bm: np.ndarray
    ax: plt.Axes
    ax.imshow(bm.T, **im_kwargs)
    ax.set_title(lbl)

# Ticks
for ax in ax_seq:
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)

# --------
# General stuff
# --------
# Ticks formatting and fine tuning
for ax in axes.ravel():
    pplot.format_axes(axes=ax, scale=1e3, decimals=0)

# Axes labels
for ax in axes[:, 0]:
    ax.set_ylabel(y_label)
for ax in axes[-1]:
    ax.set_xlabel(x_label)

# Show
plt.show()
