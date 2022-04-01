#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Union

from utils.signal import convert_lin_to_db


# Methods to compute upper and lower bounds from Equation (S14)
def lower_bound(x: Union[float, np.ndarray]) -> np.ndarray:
    return 2 * np.sqrt(-1 / np.pi * np.log((1 + x) / 2))


def upper_bound(x: Union[float, np.ndarray]) -> np.ndarray:
    return 2 * np.sqrt(-1 / np.pi * np.log((1 - x) / 2))


# Compute upper and lower bounds
num = 1000
val = np.linspace(0, 1, num=num + 2)[:-1]
bound_l = lower_bound(val)
bound_u = upper_bound(val)
bound_db_l = convert_lin_to_db(bound_l)
bound_db_u = convert_lin_to_db(bound_u)

# Compute other relevant statistical properties
median = 2 * np.sqrt(1 / np.pi * np.log(2))
median_db = convert_lin_to_db(median)
std_normal = np.sqrt(2 / np.pi)
mean = std_normal * np.sqrt(np.pi / 2)
mean_db = convert_lin_to_db(mean)
mode = std_normal
mode_db = convert_lin_to_db(mode)

# Figure
val_seq = 0.9,
clr_seq = 'C7',
val_p = val * 100

fig, ax = plt.subplots(constrained_layout=True)
ax: plt.Axes
ax.plot(val_p, bound_db_u, label='Upper Bound', color='C0')
ax.plot(val_p, bound_db_l, label='Lower Bound', color='C1')
ax.axhline(0, ls='--', label='Mean', color='C2')
ax.axhline(median_db, ls='--', label='Median', color='C3')
ax.axhline(mode_db, ls='--', label='Mode', color='C4')
bl_db_seq = []
bu_db_seq = []
for v, clr in zip(val_seq, clr_seq):
    bl, bu = lower_bound(v), upper_bound(v)
    bl_db = convert_lin_to_db(bl)
    bu_db = convert_lin_to_db(bu)
    ax.axhline(bl_db, ls='--', color=clr)
    ax.axhline(bu_db, ls='--', color=clr)
for v, clr in zip(val_seq, clr_seq):
    v_p = 100 * v
    bl, bu = lower_bound(v), upper_bound(v)
    bl_db = convert_lin_to_db(bl)
    bu_db = convert_lin_to_db(bu)
    label = f"{v_p:.0f}%".replace('-', '−')
    ax.axvline(v_p, ls='--', color=clr, label=label)
    print("Confidence {:.1f}%: [{:+.2f}, {:+.2f}]".format(v_p, bl_db, bu_db))

ax.grid()
ax.legend()
ax.set_ylabel("Normalized Amplitude (dB)")
ax.set_xlabel("Confidence Level (%)")
ax.set_ylim(ymin=-21.95, ymax=+8.29)
ax.yaxis.set_major_locator(ticker.MultipleLocator(6))

plt.show()
