import numpy as np
from typing import Optional, Union, Sequence


def compute_contrast(
        sig1: np.ndarray,
        sig2: np.ndarray,
        axis: Optional[Union[int, Sequence[int]]] = None
) -> np.ndarray:

    # Compute contrast
    contrast = np.mean(sig1, axis=axis) / np.mean(sig2, axis=axis)

    return contrast


def compute_auto_cov(
        x: np.ndarray,
        normalize: bool = True,
        axes=(-2, -1)
) -> np.ndarray:
    # Auto-covariance method, see equation (8) from
    #   Foster et al. "Computer Simulations of Speckle in B-Scan Images"

    # Pre-compute stats
    x_mean = np.mean(x, axis=axes, keepdims=True)
    x_var = np.var(x, axis=axes, keepdims=True)

    # Mean subtraction
    x_shifted = x - x_mean

    # FFT shape for 'full' correlation (2 * n - 1)
    # TODO: use "fast" FFT shapes (esp. for large arrays)
    ac_shape = [2 * x_shifted.shape[ax] - 1 for ax in axes]
    fft_shape = ac_shape

    # Compute auto-correlation along axes
    cf = np.fft.fftn(x_shifted, s=fft_shape, axes=axes)
    sf = cf * np.conj(cf)
    acf_full = np.fft.fftshift(np.fft.ifftn(sf, axes=axes).real, axes=axes)

    # Extract valid part
    x_shape = np.array(x_shifted.shape)
    a_shape = np.array(acf_full.shape)
    start_idx = (a_shape[list(axes)] - x_shape[list(axes)]) // 2
    stop_idx = start_idx + x_shape[list(axes)]
    valid_slicer = [slice(None) for _ in range(x_shifted.ndim)]
    axes_slicer = [
        np.s_[start:stop] for start, stop in zip(start_idx, stop_idx)
    ]
    for ax_ind, slc in zip(axes, axes_slicer):
        valid_slicer[ax_ind] = slc
    acf_valid = np.copy(acf_full[tuple(valid_slicer)])

    # Normalization w.r.t. variance (auto covariance)
    if normalize:
        sample_number = np.prod([x_shape[ax] for ax in axes])
        acf_valid /= x_var * sample_number

    return acf_valid
