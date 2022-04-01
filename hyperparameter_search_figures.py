#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl

from pathlib import Path
from typing import Iterable, Optional

from utils.training.manager import RunManager
from utils.training.run_configs import get_image_representation_managers
from utils.training.run_configs import get_reference_image_managers
from utils.training.run_configs import get_training_loss_managers
from utils.training.run_configs import get_conv_block_and_skip_conn_managers
from utils.training.run_configs import get_channel_number_managers
from utils.training.run_configs import get_train_size_managers
from utils.training.run_configs import get_kernel_init_managers
from utils.training.run_configs import get_learning_rate_managers

# -----------------------------------------------------------------------------
# Settings and utils
# -----------------------------------------------------------------------------
models_dir = Path('./data/trained-models')

mpl.rc('figure.constrained_layout', use=True)

iter_label = "Number of Iterations"


def plot_bmode_ssim_curves(
        managers: Iterable[RunManager],
        labels: Iterable[str] = None,
        ax: Optional[plt.Axes] = None
) -> plt.Axes:
    # Load training logs
    logs = [m.load_training_log() for m in managers]

    # Labels
    _labels = len(logs) * [None] if labels is None else labels

    # Create axes
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1)
        ax: plt.Axes
        ax.set_xlabel(iter_label)
        ax.set_ylabel("B-mode SSIM")
        ax.grid()

    # Plot
    for log, lbl in zip(logs, _labels):
        ax.plot(log.iteration, log.val_mapped_clipped_ssim, label=lbl)

    if labels is not None:
        ax.legend(loc="lower right")

    return ax

# -----------------------------------------------------------------------------
# Image representations
# Reference image configurations
# -----------------------------------------------------------------------------
img_rep_mgrs = get_image_representation_managers(models_dir)
ref_img_mgrs = get_reference_image_managers(models_dir)
model_managers = [
    img_rep_mgrs['rf-uq-mslae'],
    img_rep_mgrs['iq-uq-mslae'],
    img_rep_mgrs['env-uq-mslae'],
    img_rep_mgrs['bm-uq-mae'],
    ref_img_mgrs['hq'],
]
model_labels = [
    "UQ + RF + MSLAE",
    "UQ + IQ + MSLAE",
    "UQ + Envelope + MSLAE",
    "UQ + B-mode + MAE",
    "HQ + IQ + MSLAE",
]

plot_bmode_ssim_curves(managers=model_managers, labels=model_labels)

# -----------------------------------------------------------------------------
# Training losses
# -----------------------------------------------------------------------------
loss_mgrs = get_training_loss_managers(models_dir)
model_managers = list(loss_mgrs.values())
model_labels = [k.upper() for k in loss_mgrs.keys()]

plot_bmode_ssim_curves(managers=model_managers, labels=model_labels)

# -----------------------------------------------------------------------------
# Convolutional Blocks and Skip Connections
# Initial Channel Expansion Numbers
# -----------------------------------------------------------------------------
block_skip_mgrs = get_conv_block_and_skip_conn_managers(models_dir)
chan_numb_mgrs = get_channel_number_managers(models_dir)

tmp_mgrs = list(block_skip_mgrs.values()) + list(chan_numb_mgrs.values())

model_managers = [tmp_mgrs[0]]
for m in tmp_mgrs[1:]:
    if m.run_config not in [t.run_config for t in model_managers]:
        model_managers.append(m)

model_labels = []
for m in model_managers:
    lbl_seq = [
        f"{m.run_config.network_config.channel_number} chan.",
        f"RCBs" if m.run_config.network_config.residual_block else f"FCBs",
        f"{m.run_config.network_config.skip_connection}."
    ]
    model_labels.append(" + ".join(lbl_seq))

plot_bmode_ssim_curves(managers=model_managers, labels=model_labels)

# -----------------------------------------------------------------------------
# Training Set Sizes
# -----------------------------------------------------------------------------
train_size_managers = get_train_size_managers(models_dir)

model_managers = list(train_size_managers.values())
model_labels = [
    f"{m.run_config.training_config.train_size}" for m in model_managers
]

model_logs = [m.load_training_log() for m in model_managers]

fig, ax = plt.subplots()
ax: plt.Axes
ax.grid()
ax.set_xlabel(iter_label)
ax.set_ylabel("MSLAE")

ls_train = '--'
ls_valid = '-'

# Generic line-style legend handles for train and valid
lt0 = mlines.Line2D([], [], ls=ls_train, color='k', label='train')
lv0 = mlines.Line2D([], [], ls=ls_valid, color='k', label='valid')
lgd_handles = [lt0, lv0]

for log, lbl in zip(model_logs, model_labels):
    # Plot train curve
    lt, = ax.plot(log.iteration, log.loss, ls=ls_train)

    # Plot valid curve (with the same color as the train curve)
    lv, = ax.plot(
        log.iteration, log.val_loss, ls=ls_valid, color=lt.get_color(),
        label=lbl
    )
    lgd_handles.append(lv)

# Adjust limits manually
ax.set_ylim(top=0.41)

# Legend
ncol = int(np.ceil(len(lgd_handles) / 2))
lgd = ax.legend(handles=lgd_handles, loc='upper center', ncol=ncol)

# -----------------------------------------------------------------------------
# Kernel Initializer
# -----------------------------------------------------------------------------
bki_mgrs = get_kernel_init_managers(models_dir)
model_managers = list(bki_mgrs.values())
model_labels = [
    m.run_config.network_config.block_kernel_initializer.replace('_', ' ').title()
    for m in model_managers
]

plot_bmode_ssim_curves(managers=model_managers, labels=model_labels)

# -----------------------------------------------------------------------------
# Learning Rates
# -----------------------------------------------------------------------------
lr_mgrs = get_learning_rate_managers(models_dir)
model_managers = list(lr_mgrs.values())
model_labels = []
for m in lr_mgrs.values():
    lr = m.run_config.training_config.learning_rate
    lr_str_seq = [f"{int(s)}" for s in f"{lr:1.0e}".split('e')]
    lr_str = (
            f"$\mathdefault{{{lr_str_seq[0]}}}\,$" + "×"
            + f"$\,\mathdefault{{10^{{{lr_str_seq[1]}}}}}$"
    )
    model_labels.append(lr_str)

plot_bmode_ssim_curves(managers=model_managers, labels=model_labels)

# -----------------------------------------------------------------------------
# Show plots
# -----------------------------------------------------------------------------
plt.show()
