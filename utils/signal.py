import numpy as np

from utils.types import Real, assert_positive_real_number


def convert_db_to_lin(x) -> np.ndarray:

    x = np.asarray(x)

    return np.power(10, x / 20)


def convert_lin_to_db(x, x_min: Real = None) -> np.ndarray:

    x = np.asarray(x)

    # Clip values smaller or equal to `x_min`
    if x_min:
        assert_positive_real_number(x=x_min)
        x = np.maximum(x, x_min)

    return 20 * np.log10(x)
