#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils.plot as pplot
from utils.datasets import load_and_preprocess_images

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
datasets_dir = Path('./data/datasets/train')
dset_path = datasets_dir.joinpath('20200304-ge9ld-random-phantom.hdf5')
group_key = 'images'

# Sample index to display (index displayed in Fig. 3: 30000)
sample_index = 30000

# Data ranges
vmin, vmax = -62, 36

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
# Figure: sample B-mode images
# -----------------------------------------------------------------------------
# Axes labels and titles
label_seq = "LQ", "HQ", "UQ"
y_label = "Axial Dimension (mm)"
x_label = "Lateral Dimension (mm)"

# Axes ticks
y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
x_ticks = [-0.02, -0.01, 0, 0.01, 0.02]

# Figure kwargs
figsize = 10, 4.5
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

# Create figure
fig, axes = plt.subplots(
    nrows=1, ncols=3, sharex='all', sharey='all', **fig_kwargs)

# Figure title
fig.suptitle(f"Sample {sample_index}")

# B-mode images
ax_seq = axes.ravel()
bm_seq = bm_dict.values()
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
axes[0].set_ylabel(y_label)
for ax in axes:
    ax.set_xlabel(x_label)

# Show
plt.show()