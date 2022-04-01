import numpy as np
from pathlib import Path
from typing import Union, Tuple, Sequence

# Types
TPath = Union[str, Path]
Real = Union[int, float]
Point2D = Union[Tuple[Real, Real], Sequence[Real], np.ndarray]
ImageAxes2D = Tuple[np.ndarray, np.ndarray]


def assert_positive_real_number(x):
    """Assert if `x` is not a positive real number"""
    assert_real_number(x=x)
    if x < 0:
        raise TypeError("Must be a positive real number")


def assert_int_number(x):
    """Assert if `x` is not integer"""
    valid_type = int, np.int, np.int32, np.int64
    if not type(x) in valid_type:
        raise TypeError("Must be an integer")


def assert_real_number(x):
    """Assert if `x` is not a real number"""
    valid_type = (
        int, float,
        np.float, np.float32, np.float64,
        np.int, np.int32, np.int64
    )
    if not type(x) in valid_type:
        raise TypeError("Must be a real number")
