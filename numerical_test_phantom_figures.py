#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import utils.plot as pplot
from utils.signal import convert_lin_to_db
from utils.metrics.phantoms.numericalphantom import NumericalPhantom

# -----------------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------------
results_dir = Path('./data/metrics')
results_fn = '20200304-ge9ld-numerical-test-phantom-results'
results_fp = results_dir.joinpath(results_fn).with_suffix('.pickle')

# Load results
results_dict: dict
with open(results_fp, mode='rb') as fp:
    results_dict = pickle.load(file=fp)

# -----------------------------------------------------------------------------
# Figures setups
# -----------------------------------------------------------------------------
phantom = NumericalPhantom()

# Extract figure properties from results
keys = list(results_dict)
bm_seq = [results_dict[k]['figures']['bmode'] for k in keys]
bm_mean_seq = [results_dict[k]['figures']['bmode_mean'] for k in keys]
xaxis, zaxis = results_dict['lq']['figures']['bmode_axes']
vmin = results_dict['lq']['figures']['vmin']
vmax = results_dict['lq']['figures']['vmax']

# Axes labels and titles
label_seq = ["LQ (CNN Input)"] + [k.upper() for k in keys[1:]]
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
xmin, xmax = xaxis[0], xaxis[-1]
zmin, zmax = zaxis[0], zaxis[-1]
dx = (xmax - xmin) / (xaxis.size - 1)
dz = (zmax - zmin) / (zaxis.size - 1)
extent = [xmin - dx / 2, xmax + dx / 2, zmax + dz / 2, zmin - dz / 2]

# Settings
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
phantom.draw_geometry(ax=ax_pg, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
phantom.draw_metric_rois(ax=ax_pg)
phantom.draw_metric_labels(ax=ax_pg)
ax_pg.set_title(pht_title)

# B-mode images
for ax, bm, lbl in zip(ax_seq, bm_seq, label_seq):
    bm: np.ndarray
    ax: plt.Axes
    ax.imshow(bm.T, **im_kwargs)
    ax.set_title(lbl)
    # Metrics ROIs
    phantom.draw_metric_rois(ax=ax)

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

# -----------------------------------------------------------------------------
# Figure: mean B-mode images
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=4, sharex='all', sharey='all', **fig_kwargs)

# Axes pointers
ax_pg: plt.Axes = axes[0, 0]
ax_rvl = axes.ravel()
ax_seq = ax_rvl[1:]

# Phantom geometry
phantom.draw_geometry(ax=ax_pg, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
ax_pg.set_title(pht_title)

# B-mode images
for ax, bm, lbl in zip(ax_seq, bm_mean_seq, label_seq):
    bm: np.ndarray
    ax: plt.Axes
    ax.imshow(bm.T, **im_kwargs)
    ax.set_title(lbl)

# Artifact annotations
clr_ew = 'C0'
clr_gl = 'C3'
clr_sl = 'C1'
ann_arrow_kw = {
    'xycoords': 'data',
    'textcoords': 'data',
    'ha': 'center',
    'va': 'center',
}
#   SL
sl_text = "SL"
sl_xy_seq = (9.5e-3, 40e-3), (9.5e-3, 31e-3), (6e-3, 27.5e-3)
sl_xytext = 0e-3, 35e-3
ann_arrowprops = pplot.get_ann_arrowprops(color=clr_sl)
sl_ax_seq = [ax_rvl[ii] for ii in [3, 6, 7]]
for ax in sl_ax_seq:
    for ii, xy in enumerate(sl_xy_seq):
        text_clr = clr_sl if ii == 0 else 'none'
        ax.annotate(
            text=sl_text, xytext=sl_xytext, xy=xy, color=text_clr,
            **ann_arrow_kw, arrowprops=ann_arrowprops
        )
#   EW
ew_text = "EW"
ew_xy_seq = (-10.5e-3, 32.5e-3), (13e-3, 46e-3), (-17.5e-3, 57.5e-3)
ew_xytext = -10e-3, 40e-3
ann_arrowprops = pplot.get_ann_arrowprops(color=clr_ew)
ew_ax_seq = [ax_rvl[ii] for ii in [5, 6, 7]]
for ax in ew_ax_seq:
    for ii, xy in enumerate(ew_xy_seq):
        text_clr = clr_ew if ii == 0 else 'none'
        ax.annotate(
            text=ew_text, xytext=ew_xytext, xy=xy, color=text_clr,
            **ann_arrow_kw, arrowprops=ann_arrowprops
        )

# Ticks formatting and fine tuning
for ax in ax_rvl:
    pplot.format_axes(axes=ax, scale=1e3, decimals=0)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)

# Axes labels
for ax in axes[:, 0]:
    ax.set_ylabel(y_label)
for ax in axes[-1]:
    ax.set_xlabel(x_label)

# -----------------------------------------------------------------------------
# Figure: linear gradient
# -----------------------------------------------------------------------------
lg_echo = [30, -50]
lg_axis = results_dict['lq']['metrics']['gradient']['axis']
xmin, xmax = lg_axis[0], lg_axis[-1]

fig_kw = {
    'constrained_layout': True,
}
fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kw)
ax.plot([xmin, xmax], lg_echo, '--k', label="Theory")
for k, res in results_dict.items():
    lbl = k.upper()
    lg_mean_seq = res['metrics']['gradient']['mean']
    lg_mean = np.mean(lg_mean_seq, axis=0)
    eps = np.spacing(1, dtype=lg_mean.dtype)
    lg_mean_db = convert_lin_to_db(lg_mean, x_min=eps)
    ax.plot(lg_axis, lg_mean_db, label=lbl)

# Axes labels
ax.set_xlabel(x_label)
ax.set_ylabel("Amplitude (dB)")

# Axis limits
ax.set_ylim(ymin=-55)

# Ticks formatting and fine tuning
pplot.format_axis(axis=ax.xaxis, scale=1e3, decimals=0)
ax.set_xticks(x_ticks)
ax.set_yticks(np.arange(-50, 50, 20))

# Legend
ax.legend(loc='lower left')

# Grid
ax.grid()

# -----------------------------------------------------------------------------
# Show all figures
# -----------------------------------------------------------------------------
plt.show()
