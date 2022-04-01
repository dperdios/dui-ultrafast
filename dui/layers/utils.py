import tensorflow as tf


def get_channel_axis(data_format: str) -> int:
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return -1
    else:
        err_msg = _err_msg_invalid_data_format(data_format=data_format)
        raise ValueError(err_msg)


def get_low_level_data_format(ndim: int, data_format: str):

    if not isinstance(ndim, int):
        raise ValueError('Must be a `int`')

    if ndim not in (1, 2, 3):
        raise ValueError('`ndim` must be 1, 2, or 3')

    if not isinstance(data_format, str):
        raise ValueError('Must be a `str`')

    if data_format == 'channels_first':
        if ndim == 1:
            return 'NCW'
        elif ndim == 2:
            return 'NCHW'
        else:  # ndim == 3
            return 'NCDHW'
    elif data_format == 'channels_last':
        if ndim == 1:
            return 'NWC'
        elif ndim == 2:
            return 'NHWC'
        else:  # ndim == 3
            return 'NDHWC'


def get_ndim_from_tensor(x: tf.Tensor) -> int:
    return len(x.shape) - 2


def _err_msg_invalid_data_format(data_format: str) -> str:

    # Create unsupported data format error message
    sup_formats = "'channels_first'", "'channels_last'"
    err_msg = 'Unsupported data format {}. '.format(data_format)
    err_msg += 'Supported formats: {}'.format(sup_formats)

    return err_msg
