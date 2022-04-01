import numpy as np
from typing import Sequence, Tuple
from tensorflow.python.keras.layers import Input, MaxPool2D
from tensorflow.python.keras.models import Model
import warnings

from dui.layers import ChannelExpansion2D, ChannelContraction2D
from dui.layers import Conv2DBlock, ResConv2DBlock
from dui.layers import SkipConcat, SkipAdd
from dui.layers import DownScaling2D, UpScaling2D
from dui.layers.utils import get_channel_axis


def create_gunet_model(
        input_shape: Sequence[int],
        channel_number: int,
        level_number: int,
        block_size: int,
        kernel_size: int,
        multiscale_factor: int,
        channel_factor: int,
        residual: bool = True,
        residual_block: bool = False,
        skip_connection: str = 'add',
        padding: str = 'same',
        data_format: str = 'channels_last',
        activation: str = 'relu',
        use_bias: bool = True,
        block_kernel_initializer: str = 'glorot_uniform',
        block_bias_initializer: str = 'zeros',
        channel_kernel_initializer: str = 'glorot_uniform',
        channel_bias_initializer: str = 'zeros',
        dtype: str = 'float32'
) -> Model:

    # Call generic factory function
    model = _create_gunet_model(
        legacy=False,
        input_shape=input_shape,
        channel_number=channel_number,
        level_number=level_number,
        block_size=block_size,
        kernel_size=kernel_size,
        multiscale_factor=multiscale_factor,
        channel_factor=channel_factor,
        residual=residual,
        residual_block=residual_block,
        skip_connection=skip_connection,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        block_kernel_initializer=block_kernel_initializer,
        block_bias_initializer=block_bias_initializer,
        channel_kernel_initializer=channel_kernel_initializer,
        channel_bias_initializer=channel_bias_initializer,
        dtype=dtype
    )

    return model


def create_gunet_model_leg(
        input_shape: Sequence[int],
        channel_number: int,
        level_number: int,
        block_size: int,
        kernel_size: int,
        multiscale_factor: int,
        channel_factor: int,
        residual: bool = True,
        residual_block: bool = False,
        skip_connection='concat',
        padding='same',
        data_format='channels_last',
        activation='relu',
        use_bias=True,
        block_kernel_initializer='glorot_uniform',
        block_bias_initializer='zeros',
        channel_kernel_initializer='glorot_uniform',
        channel_bias_initializer='zeros',
        dtype: str = 'float32'
) -> Model:

    # Call generic factory function
    model = _create_gunet_model(
        legacy=True,
        input_shape=input_shape,
        channel_number=channel_number,
        level_number=level_number,
        block_size=block_size,
        kernel_size=kernel_size,
        multiscale_factor=multiscale_factor,
        channel_factor=channel_factor,
        residual=residual,
        residual_block=residual_block,
        skip_connection=skip_connection,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        block_kernel_initializer=block_kernel_initializer,
        block_bias_initializer=block_bias_initializer,
        channel_kernel_initializer=channel_kernel_initializer,
        channel_bias_initializer=channel_bias_initializer,
        dtype=dtype
    )

    return model


def _create_gunet_model(
        input_shape: Sequence[int],
        channel_number: int,
        level_number: int,
        block_size: int,
        kernel_size: int,
        multiscale_factor: int,
        channel_factor: int,
        legacy: bool = False,
        residual: bool = True,
        residual_block: bool = False,
        skip_connection: str = 'add',
        padding: str = 'same',
        data_format: str = 'channels_last',
        activation: str = 'relu',
        use_bias: bool = True,
        block_kernel_initializer: str = 'glorot_uniform',
        block_bias_initializer: str = 'zeros',
        channel_kernel_initializer: str = 'glorot_uniform',
        channel_bias_initializer: str = 'zeros',
        dtype: str = 'float32'
) -> Model:

    # TODO: create utility function to check all input parameters
    #  that could be called when checking network parameters before launching
    #  large training loops

    # Create model name
    # TODO: naming convention is quite outdated and not really used. Remove?
    model_name_short = 'gunet'
    if legacy:
        model_name_short += 'leg'
    sep, delim_o, delim_c = '_', '-', ''  # ok
    model_name_list = [
        model_name_short,
        'ch{:s}{:d}{:s}'.format(delim_o, channel_number, delim_c),
        'ln{:s}{:d}{:s}'.format(delim_o, level_number, delim_c),
        'bs{:s}{:d}{:s}'.format(delim_o, block_size, delim_c),
        'ks{:s}{:d}{:s}'.format(delim_o, kernel_size, delim_c),
        'mf{:s}{:d}{:s}'.format(delim_o, multiscale_factor, delim_c),
        'af{:s}{:s}{:s}'.format(delim_o, activation, delim_c),
        'sc{:s}{:s}{:s}'.format(delim_o, skip_connection, delim_c),
        'rb{:s}{}{:s}'.format(delim_o, residual_block, delim_c),
    ]
    model_name_full = sep.join(model_name_list).lower()
    model_name = model_name_full

    # Input shape
    input_shape = tuple(input_shape)
    if not all(isinstance(s, int) for s in input_shape):
        raise ValueError('Must be a sequence of `int`')
    if len(input_shape) != 3:
        raise NotImplementedError("(H, W, C) or (C, H, W)")

    if data_format == 'channels_last':
        image_channels = input_shape[-1]  # (H, W, C) or (..., C)
    elif data_format == 'channels_first':
        image_channels = input_shape[0]  # (C, H, W) or (C, ...)
        warnings.warn(
            "'{}' really needs to be checked (tensorboard)".format(data_format))
    else:
        raise ValueError

    # TODO: downsampling checks w.r.t. to image shape

    # Check input shape compatibility
    _assert_compat_input_shape(
        input_shape=input_shape, multiscale_factor=multiscale_factor,
        level_number=level_number, data_format=data_format
    )

    # Global arguments and layers
    #   Channel expansion and contraction
    ch_kwargs = {
        'kernel_size': 1,
        'padding': padding,
        'data_format': data_format,
        'activation': None,
        'use_bias': False,
        'kernel_initializer': channel_kernel_initializer,
        'bias_initializer': channel_bias_initializer,
        'dtype': dtype
    }
    #   Convolutional blocks
    conv_block_kwargs = {
        'conv_number': block_size,
        'kernel_size': kernel_size,
        'strides': 1,
        'padding': padding,
        'data_format': data_format,
        'activation': activation,
        'use_bias': use_bias,
        'kernel_initializer': block_kernel_initializer,
        'bias_initializer': block_bias_initializer,
        'dtype': dtype
    }
    if residual_block:
        if skip_connection == 'concat':
            err_msg = "Incompatible residual block with 'concat' skip"
            raise ValueError(err_msg)
        conv_block_cls = ResConv2DBlock
    else:
        conv_block_cls = Conv2DBlock
    #   Downscaling layer
    ds_layer_kwargs = {
        'factor': multiscale_factor,
        'channel_factor': channel_factor,
        'kernel_size': kernel_size,
        'padding': padding,
        'data_format': data_format,
        'activation': activation,
        'kernel_initializer': block_kernel_initializer,
        'use_bias': use_bias,
        'bias_initializer': block_bias_initializer,
        'dtype': dtype
    }
    ds_layer_cls = DownScaling2D
    us_layer_kwargs = dict(ds_layer_kwargs)  # copy
    us_layer_cls = UpScaling2D
    #   Skip connections
    sc_kwargs = {
        'data_format': data_format,
        'dtype': dtype
    }
    if skip_connection is not None:
        if skip_connection == 'add':
            sc_cls = SkipAdd
        elif skip_connection == 'concat':
            sc_cls = SkipConcat
        else:
            raise ValueError
    else:
        sc_cls = None
    #   Legacy updates (downscaling layer)
    pool_cls, pool_kwargs = None, None
    if legacy:
        ds_layer_kwargs['factor'] = 1
        pool_cls = MaxPool2D
        pool_kwargs = {
            'pool_size': multiscale_factor,
            'strides': multiscale_factor,
            'padding': padding,
            'data_format': data_format
        }

    # Inputs
    inputs = Input(shape=input_shape, name='inputs', dtype=dtype)

    # Initial channel expansion
    outputs = ChannelExpansion2D(channels=channel_number, **ch_kwargs)(inputs)

    # Downward path (encoder)
    skip_list = list()
    for _ in range(level_number - 1):
        # Convolutional block
        outputs = conv_block_cls(**conv_block_kwargs)(outputs)
        # Store tensors for skip connections
        skip_list.append(outputs)
        # Multiscale operation
        if legacy:
            outputs = pool_cls(**pool_kwargs)(outputs)
        outputs = ds_layer_cls(**ds_layer_kwargs)(outputs)

    # Bottom block (convolutional block)
    outputs = conv_block_cls(**conv_block_kwargs)(outputs)

    # Upward path (decoder)
    channel_axis = get_channel_axis(data_format=data_format)
    for _, sc in zip(range(level_number - 1), reversed(skip_list)):
        # Multiscale operation
        outputs = us_layer_cls(**us_layer_kwargs)(outputs)
        # Skip connections
        if skip_connection is not None:
            outputs = sc_cls(**sc_kwargs)([outputs, sc])
            if skip_connection == 'concat':  # special case for concatenation
                output_shape = outputs.shape.as_list()
                filters = output_shape[channel_axis] // channel_factor
                # Specify `filters` (to bottleneck concatenated input)
                conv_block_kwargs['filters'] = filters
        # Convolutional block
        outputs = conv_block_cls(**conv_block_kwargs)(outputs)

    # Channel contraction
    outputs = ChannelContraction2D(channels=image_channels, **ch_kwargs)(
        outputs)

    # Global residual skip connection
    if residual:
        outputs = SkipAdd(**sc_kwargs)([outputs, inputs])

    # Create functional model
    return Model(inputs=inputs, outputs=outputs, name=model_name)


def _get_caxis(
        input_shape: Sequence[int],
        data_format: str
) -> int:

    # Note: does not account for batch size!

    if data_format == 'channels_last':
        caxis = -1
    elif data_format == 'channels_first':
        caxis = 0
    else:
        raise ValueError

    return caxis


def _get_image_axes(
        input_shape: Sequence[int],
        data_format: str,
) -> Sequence[int]:

    # Note: does not account for batch size!

    caxis = _get_caxis(input_shape=input_shape, data_format=data_format)
    input_axes = tuple(range(len(input_shape)))
    image_axes = tuple([a for a in input_axes if a != input_axes[caxis]])

    return image_axes


def _get_closest_pad_and_crop_shapes(
        input_shape: Sequence[int],
        multiscale_factor,
        level_number: int,
        data_format: str,
) -> Tuple[Sequence[int], Sequence[int]]:

    # Extract properties
    caxis = _get_caxis(input_shape=input_shape, data_format=data_format)
    pool_size = multiscale_factor
    imaxes = _get_image_axes(input_shape=input_shape, data_format=data_format)
    imaxes = list(imaxes)

    # Compute closest input shapes (downsampling or upsampling)
    tot_div = np.zeros_like(input_shape)
    tot_div[caxis] = input_shape[caxis]
    tot_div[imaxes] = np.power(pool_size, level_number - 1)
    mod = np.mod(input_shape, tot_div)
    crop_shape = mod
    pad_shape = np.subtract(
        tot_div, mod, where=mod != 0, out=np.zeros_like(input_shape)
    )

    return tuple(pad_shape), tuple(crop_shape)


def _get_closest_compat_input_shape(
        input_shape: Sequence[int],
        multiscale_factor,
        level_number: int,
        data_format: str,
) -> Tuple[Sequence[int], Sequence[int]]:

    # Get closed pad and crop shapes
    pad_shape, crop_shape = _get_closest_pad_and_crop_shapes(
        input_shape=input_shape,
        multiscale_factor=multiscale_factor,
        level_number=level_number,
        data_format=data_format,
    )

    up_shape = tuple(np.add(input_shape, pad_shape))
    dw_shape = tuple(np.subtract(input_shape, crop_shape))

    # Make sure they are Python int
    up_shape = tuple([int(s) for s in up_shape])
    dw_shape = tuple([int(s) for s in dw_shape])

    return up_shape, dw_shape


def _get_closest_sample_pad_and_crop_shapes(
        input_shape: Sequence[int],
        multiscale_factor,
        level_number: int,
) -> Tuple[Sequence[int], Sequence[int]]:

    # Compute closest input shapes (downsampling or upsampling)
    tot_div = np.power(multiscale_factor, level_number - 1)
    mod = np.mod(input_shape, tot_div)
    crop_shape = mod
    pad_shape = np.subtract(
        tot_div, mod, where=mod != 0, out=np.zeros_like(input_shape)
    )

    return tuple(pad_shape), tuple(crop_shape)


def _assert_compat_input_shape(
        input_shape: Sequence[int],
        multiscale_factor,
        level_number: int,
        data_format: str,
) -> None:

    up_shape, dw_shape = _get_closest_compat_input_shape(
        input_shape=input_shape, multiscale_factor=multiscale_factor,
        level_number=level_number, data_format=data_format
    )

    if up_shape != dw_shape:
        err_msg = 'Incompatible input shape {}. '.format(input_shape)
        err_msg += 'Closest compatible cropped shape: {}. '.format(dw_shape)
        err_msg += 'Closest compatible padded shape: {}. '.format(up_shape)
        raise ValueError(err_msg)
