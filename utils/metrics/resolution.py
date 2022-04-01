import warnings
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def compute_fwhm_line(x: np.ndarray, y: np.ndarray) -> float:

    # Check inputs
    if x.ndim != 1:
        raise ValueError("Must be a 1-D array")
    if y.ndim != 1:
        raise ValueError("Must be a 1-D array")
    if x.size != y.size:
        raise ValueError("Incompatible array sizes")

    # Sub-pixel peak finder
    #   Note: Even suboptimal, the use of a spline of degree 4 enables taking
    #   the derivative (resulting in a spline of degree 3) and computing the
    #   roots for extracting the interpolated maximum value directly.
    spl = InterpolatedUnivariateSpline(x=x, y=y, k=4, ext='zeros')
    spl_der = spl.derivative(n=1)
    x_roots = spl_der.roots()
    if x_roots.size == 0:
        warnings.warn(
            "FWHM computation failed (no zero derivative found), NaN returned"
        )
        return np.nan
    y_max = np.max(spl(x=x_roots))

    # Compute FWHM
    #   Spline representation "at half maximum," i.e. max / 2
    y_hm = y - y_max / 2
    spl_hm = InterpolatedUnivariateSpline(x=x, y=y_hm, k=3, ext='zeros')
    #   Roots and difference
    fwhm_roots = spl_hm.roots()
    if len(fwhm_roots) != 2:
        warnings.warn("FWHM computation failed (too many roots), NaN returned")
        return np.nan
    else:
        return float(np.abs(np.diff(fwhm_roots)))
