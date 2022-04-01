import numpy as np
import scipy.signal
from typing import Optional
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K

from dui.layers.utils import get_channel_axis, get_low_level_data_format
from dui.layers.utils import get_ndim_from_tensor


SUP_SIG_TYPE = 'rf', 'iq', 'env', 'bm'


def convert_to_bmode(
        tensor: tf.Tensor,
        signal_type: str,
        data_format: str = 'channels_last',
        # TODO: dtype? for now inferred from tensor
) -> tf.Tensor:

    # Check signal type
    signal_type = check_signal_type(signal_type)

    # Make sure `tensor` is a Tensor
    tensor = tf.convert_to_tensor(tensor)

    # Already B-mode
    if signal_type == 'bm':
        return tensor

    # Extract envelope
    if signal_type == 'env':
        pass
    elif signal_type in ('rf', 'iq'):
        tensor = envelope(
            tensor, signal_type=signal_type, data_format=data_format
        )
    else:
        raise RuntimeError()  # should never happen

    # Compress dB
    return compress_db(tensor)


def envelope(
        tensor: tf.Tensor,
        signal_type: str,
        data_format: str = 'channels_last',
        # TODO: dtype? for now inferred from tensor
) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)

    # Check signal type
    signal_type = check_signal_type(signal_type)

    # TODO: name_scope?

    if signal_type == 'bm':
        raise ValueError('Cannot extract envelope from B-mode signal')
    elif signal_type == 'env':
        return tensor
    elif signal_type == 'iq':
        # Compute envelope from IQ (as channels)
        return envelope_from_iq_chan(tensor=tensor, data_format=data_format)
    elif signal_type == 'rf':
        # Get channel axis
        channel_axis = get_channel_axis(data_format=data_format)

        # Compute envelope from RF
        env_rf_kwargs = {
            'filter_size': 33, 'beta': 8, 'axis': channel_axis,
            'data_format': data_format
        }
        return envelope_fir(tensor=tensor, **env_rf_kwargs)
    else:
        raise RuntimeError()  # should never happen


def to_bmode(
        tensor: tf.Tensor,
        signal_type: str,
        data_format: str = 'channels_last',
        # TODO: dtype? for now inferred from tensor
) -> tf.Tensor:

    # Check signal type
    signal_type = check_signal_type(signal_type)

    # Channel axis
    channel_axis = get_channel_axis(data_format=data_format)

    # Make sure `tensor` is a Tensor
    # TODO: dtype?
    tensor = tf.convert_to_tensor(tensor)

    # Compression w.r.t. signal type
    if signal_type == 'rf':
        # TODO: provide access to some kwargs
        bm_rf_kwargs = {
            'filter_size': 33, 'beta': 8, 'axis': channel_axis,
            'data_format': data_format
        }
        return bmode_from_rf(tensor, **bm_rf_kwargs)
    elif signal_type == 'iq':
        return bmode_from_iq_chan(tensor, data_format=data_format)
    elif signal_type == 'env':
        return compress_db(tensor)
    elif signal_type == 'bm':
        return tensor
    else:
        err_msg = _err_msg_signal_type(name=signal_type)
        raise ValueError(err_msg)


def log10(
        tensor: tf.Tensor,
        name: Optional[str] = None
) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)
    dtype = tensor.dtype.name

    with ops.name_scope(name, 'log10', values=[tensor]):
        num = tf.math.log(tensor)
        den = tf.math.log(tf.constant(10, dtype=dtype))
        return num / den


def compress_db(tensor: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)
    dtype = tensor.dtype.name

    with ops.name_scope(name, 'db-comp', values=[tensor]):

        # Clip values smaller or equal to numerical precision
        # eps = tf.constant(np.spacing(1, dtype=dtype), dtype=dtype)
        # tensor = tf.math.maximum(tensor, eps)
        tensor = tf.math.maximum(tensor, K.epsilon())

        # Compute db compression (20 * log10(x))
        fct = tf.constant(20, dtype=dtype)
        out = tf.math.multiply(fct, log10(tensor))

    return out


def bmode_from_iq_chan(
        tensor: tf.Tensor,
        data_format: str = 'channels_last',
        name: Optional[str] = None
) -> tf.Tensor:

    with ops.name_scope(name, 'bmode-iq-chan', values=[tensor]):
        # Compute envelope
        env = envelope_from_iq_chan(
            tensor=tensor,
            data_format=data_format
        )

        # Compress
        out = compress_db(env)

    return out


def bmode_from_rf(
        tensor: tf.Tensor,
        filter_size: int,
        beta: float = 8.,
        axis: int = -1,
        data_format: str = 'channels_last',
        name: Optional[str] = None
) -> tf.Tensor:

    with ops.name_scope(name, 'bmode-rf', values=[tensor]):
        # Compute envelope
        env = envelope_fir(
            tensor=tensor,
            axis=axis,
            filter_size=filter_size,
            beta=beta,
            data_format=data_format
        )

        # Compress
        out = compress_db(env)

    return out


def envelope_from_iq_chan(  # TODO: from_iq_as_channels
        tensor: tf.Tensor,
        data_format: str = 'channels_last',
        name: Optional[str] = None
) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)

    # Get channel axis
    channel_axis = get_channel_axis(data_format=data_format)

    with ops.name_scope(name, 'envelope-iq', values=[tensor]):
        env = tf.norm(tensor, ord='euclidean', axis=channel_axis, keepdims=True)
        return env


def envelope_fir(
        tensor: tf.Tensor,
        filter_size: int,
        beta: float = 8.,  # MATLAB default value
        axis: int = -1,
        data_format: str = 'channels_last',
        name: Optional[str] = None
) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)

    with ops.name_scope(name, 'envelope-fir', values=[tensor]):
        # Compute FIR hilbert (analytical signal)
        x_ana = hilbert_fir(
            tensor=tensor,
            filter_size=filter_size,
            beta=beta,
            axis=axis,
            data_format=data_format,
        )

        # Take modulus
        out = tf.math.abs(x_ana)

    return out


def hilbert_fir(
        tensor: tf.Tensor,
        filter_size: int,
        beta: float,
        axis: int,
        data_format: str = 'channels_last',
        name: Optional[str] = None
) -> tf.Tensor:

    tensor = tf.convert_to_tensor(tensor)

    # Extract values
    dtype = tensor.dtype.name
    # input_shape = tensor.shape
    # TODO: issues with eager here (if numpy array input)?
    input_shape = tensor.shape.as_list()
    ndim = get_ndim_from_tensor(x=tensor)
    channel_axis = get_channel_axis(data_format=data_format)
    low_level_data_format = get_low_level_data_format(ndim, data_format)

    # Check axis
    #   Note: `axis` corresponds to the "image" axis (i.e. batch not included)
    if not isinstance(axis, int):
        raise ValueError('Must be a `int`')
    if axis < 0:
        filter_axis = ndim + axis
    else:
        filter_axis = axis
    if filter_axis < 0 or filter_axis > ndim - 1:
        raise ValueError('Invalid `axis` value')

    # Create FIR filter (complex)
    fir_filter = compute_fir_hilbert(filter_size=filter_size, beta=beta)
    #   Flip filter since TensorFlow actually performs correlations
    fir_filter = np.copy(np.flip(fir_filter))
    # fir_filter = np.flip(fir_filter)

    # Compute filter shape
    # TODO: why automatic computation sometimes fail?
    fir_shape = len(input_shape) * [1]
    fir_shape[-2:] = 2 * [input_shape[channel_axis]]
    # fir_shape[filter_axis] = fir_filter.size
    fir_shape[filter_axis] = filter_size
    # TODO: hardcoded version is ok...
    #   Seems like tensor.shape does (in some settings, return NoneTypes...)
    if data_format != 'channels_last':
        raise NotImplementedError
    # fir_shape = [1, filter_size, 1, 1]  # this works
    fir_shape = len(input_shape) * [1]  # this works too
    fir_shape[filter_axis] = filter_size  # this works too...

    # Create real and imag FIR parts
    fir_filter_real = tf.constant(
        fir_filter.real.astype(dtype='float32', order='C'),
        shape=fir_shape, dtype='float32'
    )
    fir_filter_imag = tf.constant(
        fir_filter.imag.astype(dtype='float32', order='C'),
        shape=fir_shape, dtype='float32'
    )

    # Compute FIR-based envelope detection
    with ops.name_scope(name, 'hilbert-fir', values=[tensor]):

        # Create real and imag filters
        fir_real = tf.convert_to_tensor(
            fir_filter_real, dtype='float32', name='fir-real'
        )
        fir_imag = tf.convert_to_tensor(
            fir_filter_imag, dtype='float32', name='fir-imag'
        )

        # Perform convolutions for real and imag parts
        conv_kwargs = {
            'input': tensor,
            'padding': 'SAME',
            'data_format': low_level_data_format
        }
        out_real = tf.nn.convolution(
            filter=fir_real, name='conv-real', **conv_kwargs
        )
        out_imag = tf.nn.convolution(
            filter=fir_imag, name='conv-imag', **conv_kwargs
        )
        out = tf.complex(out_real, out_imag)

    return out


def compute_fir_hilbert(filter_size: int, beta: float) -> np.ndarray:
    """Compute FIR Hilbert transformer (filter) approximation using a Kaiser
    window

    Note: same as Matlab implementation
    """
    # Check inputs
    if filter_size % 2 == 0:
        raise ValueError('Must be odd')

    # Construct ideal hilbert filter truncated to desired length
    fc = 1
    t = fc / 2 * np.arange((1 - filter_size) / 2, filter_size / 2)
    fir_hilbert = np.sinc(t) * np.exp(1j * np.pi * t)

    # Multiply ideal filter with tapered window
    fir_hilbert *= scipy.signal.windows.kaiser(filter_size, beta)
    fir_hilbert /= np.sum(fir_hilbert.real)

    return fir_hilbert


def check_signal_type(name: str) -> str:
    assert_signal_type(name=name)
    return name.lower()


def assert_signal_type(name: str) -> None:
    if not isinstance(name, str):
        raise ValueError('Must be a `str`')
    if not name in SUP_SIG_TYPE:
        err_msg = _err_msg_signal_type(name=name)
        raise ValueError(err_msg)


def _err_msg_signal_type(name: str) -> str:
    err_msg = 'Unsupported signal type: {}. '.format(name)
    sup_list_str = ["'{}'".format(t) for t in SUP_SIG_TYPE]
    sup_str = ','.join(sup_list_str)
    err_msg += 'Supported types: {}'.format(sup_str)

    return err_msg
