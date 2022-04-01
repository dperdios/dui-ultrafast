import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare_psnr_batch(
        true: np.ndarray,
        test: np.ndarray,
        data_range: int = None
) -> np.ndarray:

    # Check input shapes
    if test.shape != true.shape:
        raise ValueError()
    if test.ndim != 3:
        raise ValueError()

    res = [
        peak_signal_noise_ratio(tr, te, data_range=data_range)
        for tr, te in zip(true, test)
    ]

    return np.array(res)


def compare_ssim_batch(
        true: np.ndarray,
        test: np.ndarray,
        data_range: int = None
) -> np.ndarray:

    # Check input shapes
    if test.shape != true.shape:
        raise ValueError()
    if test.ndim != 3:
        raise ValueError()

    # Attributes to match the SSIM implementation of Wang et. al.
    kwargs = {
        'gaussian_weights': True,
        'sigma': 1.5,
        'use_sample_covariance': False
    }

    res = [
        structural_similarity(tr, te, data_range=data_range, **kwargs)
        for tr, te in zip(true, test)
    ]

    return np.array(res)
