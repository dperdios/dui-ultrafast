#!/usr/bin/env python3
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utils.plot as pplot
from utils.datasets import load_and_preprocess_images
from utils.training.run_configs import get_test_samples_managers

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
datasets_dir = Path('./data/datasets/test')
dset_path = datasets_dir.joinpath('20200304-ge9ld-random-phantom-test-set.hdf5')
group_key = 'images'

# Sample index to display (index displayed in Fig. S4: 125)
sample_index = 125

# Trained model managers
trained_models_dir = Path('./data/trained-models')
model_managers = get_test_samples_managers(trained_models_dir)

# Data ranges
vmin, vmax = -62, 36

# -----------------------------------------------------------------------------
# Load ellipsoidal inclusion definitions
# -----------------------------------------------------------------------------
incl_keys = 'amp', 'ang', 'pos', 'semiaxes',
incl_dict = {}
with h5py.File(dset_path, 'r') as h5r:
    for k in incl_keys:
        dset = h5r['inclusions/' + k]
        incl_dict[k] = dset[sample_index]

# -----------------------------------------------------------------------------
# Load images: inputs and targets
# -----------------------------------------------------------------------------
bm_dict = {}

image_keys = 'lq', 'hq', 'uq'
input_signal = 'iq'
output_signal = 'bm'

dset_path = dset_path.resolve(strict=True)

# Load images
for k in image_keys:
    # Build dataset key
    dset_name = '/'.join([group_key, k])
    print(f"Loading '{dset_name}' from '{dset_path}'")

    # Load images as B-mode (display only)
    images, image_axes = load_and_preprocess_images(
        path=dset_path,
        name=dset_name,
        input_signal=input_signal,
        input_factor='0db',
        output_signal=output_signal,
        samples_slicer=sample_index,
    )

    # Store
    bm_dict[k] = np.copy(images)

# -----------------------------------------------------------------------------
# Load images: predictions
# -----------------------------------------------------------------------------
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

    # Load images as B-mode (display only)
    input_signal = mgr.run_config.mapping_config.output_signal
    images, image_axes = load_and_preprocess_images(
        path=dset_path,
        name=dset_name,
        input_signal=input_signal,
        output_signal=output_signal,
        samples_slicer=sample_index
    )

    # Store
    bm_dict[k] = images

# -----------------------------------------------------------------------------
# Figures setups
# -----------------------------------------------------------------------------
# Extract figure properties
keys = (
    'lq', 'hq', 'uq',
    'bm-uq-mae',
    'env-uq-mslae',
    'iq-hq-mslae',
    'iq-uq-mslae'
)
bm_seq = [bm_dict[k] for k in keys]
xaxis, zaxis = image_axes

# Axes labels and titles
label_seq = (
    "LQ (CNN Input)", "HQ", "UQ",
    "UQ + B-mode + MAE",
    "UQ + Envelope + MSLAE",
    "HQ + IQ + MSLAE",
    "UQ + IQ + MSLAE",
)
y_label = "Axial Dimension (mm)"
x_label = "Lateral Dimension (mm)"
pht_title = "Phantom Geometry"

# Axes ticks
y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
x_ticks = [-0.02, -0.01, 0, 0.01, 0.02]

# Figure kwargs
figsize = 12.8, 12.8 / 1.5
fig_kwargs = {'figsize': figsize, 'constrained_layout': True}

# Limits and extent
xaxis, zaxis = image_axes
xmin, xmax = xaxis[0], xaxis[-1]
zmin, zmax = zaxis[0], zaxis[-1]
dx = (xmax - xmin) / (xaxis.size - 1)
dz = (zmax - zmin) / (zaxis.size - 1)
extent = [xmin - dx / 2, xmax + dx / 2, zmax + dz / 2, zmin - dz / 2]

# Imshow settings
db_range = vmax - vmin
cmap = 'gray'
cm_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
im_kwargs = {**cm_kwargs, 'extent': extent}

# -----------------------------------------------------------------------------
# Figure: sample B-mode images
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=4, sharex='all', sharey='all', **fig_kwargs)

# Axes pointers
ax_pg: plt.Axes = axes[0, 0]
ax_seq = axes.ravel()[1:]

# Phantom geometry
ax_pg.set_title("Phantom Geometry")
ax_pg.set_xlim(xmin=xmin, xmax=xmax)  # not needed with share all
ax_pg.set_ylim(ymin=zmax, ymax=zmin)  # not needed with share all
ax_pg.set_aspect(1)
#   Background
bckg_botleft = xmin, zmin
echogen = 0  # filled with 0 dB
color = np.clip((echogen - vmin) / (vmax - vmin), a_min=0, a_max=1)
bckg = mpatches.Rectangle(
    # xy=bckg_botleft, width=xmax - xmin, height=zmax - zmin, color="0"
    xy=bckg_botleft, width=xmax - xmin, height=zmax - zmin, color=str(color)
)
ax_pg.add_artist(bckg)
#   Inclusions
incl_zipper = zip(
    incl_dict['amp'], incl_dict['ang'],
    incl_dict['pos'].T, incl_dict['semiaxes'].T
)
for amp, ang, pos, semiaxes in incl_zipper:
    # Color
    echogen = 20 * np.log10(amp)
    color = np.clip((echogen - vmin) / (vmax - vmin), a_min=0, a_max=1)
    color = str(color)

    # Ellipse
    el_width, el_height = 2 * semiaxes[0], 2 * semiaxes[1]
    el_center = pos[0], pos[2]
    # ang_deg = np.rad2deg(ang)
    ang_deg = -np.rad2deg(ang)  # Note: were inverted
    ellipse = mpatches.Ellipse(
        xy=el_center, width=el_width, height=el_height, angle=ang_deg,
        color=color, linewidth=0
    )
    ax_pg.add_artist(ellipse)

# B-mode images
for ax, bm, lbl in zip(ax_seq, bm_seq, label_seq):
    bm: np.ndarray
    ax: plt.Axes
    ax.imshow(bm.T, **im_kwargs)
    ax.set_title(lbl)

# Ticks formatting and fine tuning
for ax in axes.ravel():
    pplot.format_axes(axes=ax, scale=1e3, decimals=0)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)

# Axes labels
for ax in axes[:, 0]:
    ax.set_ylabel(y_label)
for ax in axes[-1]:
    ax.set_xlabel(x_label)

# Show
plt.show()
