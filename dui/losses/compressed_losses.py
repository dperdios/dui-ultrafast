import tensorflow as tf
from typing import Optional

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K


class MeanMuAbsoluteError(LossFunctionWrapper):
    def __init__(
            self,
            mu: float = 255.,
            reduction=losses_utils.ReductionV2.AUTO,
            name='mu_absolute_error',
    ):

        # TODO: store mu?

        super(MeanMuAbsoluteError, self).__init__(
            fn=mean_mu_absolute_error,
            name=name,
            reduction=reduction,
            mu=mu
        )


# Aliases
MMUAE = MeanMuAbsoluteError


class MeanSignedLogAbsoluteError(LossFunctionWrapper):
    def __init__(
            self,
            min_value: Optional[float] = None,
            reduction=losses_utils.ReductionV2.AUTO,
            name='signed_log_absolute_error',
    ):

        # TODO: add test on min_value here also?
        # TODO: store min_value?

        super(MeanSignedLogAbsoluteError, self).__init__(
            fn=mean_signed_log_absolute_error,
            name=name,
            reduction=reduction,
            min_value=min_value
        )


# Alias
MSLAE = MeanSignedLogAbsoluteError


def mean_mu_absolute_error(
        y_true,
        y_pred,
        mu: float = 255.,
):
    # Inputs
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    # Compression
    c_pred = _compress_to_mu_law(y_pred, mu=mu)
    c_true = _compress_to_mu_law(y_true, mu=mu)

    return K.mean(math_ops.abs(c_pred - c_true), axis=-1)


def mean_signed_log_absolute_error(
        y_true,
        y_pred,
        min_value: Optional[float] = None,
):
    # Inputs
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    # Compression
    c_pred = _compress_to_signed_log(y_pred, min_value=min_value)
    c_true = _compress_to_signed_log(y_true, min_value=min_value)

    return K.mean(math_ops.abs(c_pred - c_true), axis=-1)


def _compress_to_mu_law(x, mu: float = 255.) -> tf.Tensor:

    x = ops.convert_to_tensor(x)
    mu = tf.constant(mu, dtype=x.dtype.base_dtype)

    num = tf.math.log(1. + mu * tf.math.abs(x))
    den = tf.math.log(1. + mu)

    return tf.math.sign(x) * num / den


def _compress_to_signed_log(
        x,
        min_value: Optional[float] = None,
) -> tf.Tensor:

    x = ops.convert_to_tensor(x)
    eps = K.epsilon()

    if not 0 < min_value < 1:
        raise ValueError('Must satisfy 0 < min_value < 1')

    if min_value is None or min_value < eps:
        min_value = eps

    # Clip
    x_min = math_ops.cast(min_value, x.dtype)
    x_clp = K.maximum(tf.math.abs(x), x_min)

    # Compress
    x_cmp = tf.math.log(x_min / x_clp) / tf.math.log(x_min)

    return tf.math.sign(x) * x_cmp


def _log10(x) -> tf.Tensor:

    x = ops.convert_to_tensor(x)

    num = tf.math.log(x)
    den = tf.math.log(tf.constant(10., dtype=x.dtype.base_dtype))

    return num / den
