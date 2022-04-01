import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Conv2D, Conv3D
from tensorflow.python.keras.layers import Conv2DTranspose, Conv3DTranspose
from typing import Optional

from .utils import get_channel_axis


class _MultiscaleLayer(Layer):

    def __init__(
            self,
            method: str,
            dim: int,
            factor: int,
            kernel_size,
            data_format='channels_last',
            channel_factor: Optional[int] = None,  # defaults to `factor`
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            trainable=True,
            dtype=None,
            **kwargs
    ):

        # TODO: really need these additional **kwargs?

        # Call super constructor
        super(_MultiscaleLayer, self).__init__(
            name=name, trainable=trainable, dtype=dtype, **kwargs
        )

        # Check inputs
        #   Method
        if not isinstance(method, str):
            raise TypeError('Must be a `str`')
        method = method.lower()
        if not method in ('up', 'down'):
            raise ValueError("Must be either 'up' or 'down'")
        self._method = method
        #   Dimension
        if not isinstance(dim, int):
            raise TypeError('Must be a `int`')
        if dim not in (2, 3):
            raise ValueError('Must be 2 or 3')
        self._dim = dim
        #   Downsampling/upsampling (downscaling/upscaling) factor
        if not isinstance(factor, int):
            raise TypeError('Must be a `int`')
        if factor < 1:
            raise ValueError('Must be >=1')
        # TODO: really even?
        if not factor == 1:
            if not factor % 2 == 0:
                raise ValueError('Must be even')
        self._factor = factor
        #   Channel (expansion) factor
        channel_factor = channel_factor or factor
        if not isinstance(channel_factor, int):
            raise TypeError('Must be a `int`')
        if channel_factor < 1:
            raise ValueError('Must be >=1')
        if not channel_factor % 2 == 0:
            raise ValueError('Must be even')
        self._channel_factor = channel_factor
        #   Data format
        if not isinstance(data_format, str):
            raise TypeError('Must be a `str`')
        data_format = data_format.lower()
        if data_format not in ('channels_first', 'channels_last'):
            raise ValueError('Unsupported data format {}'.format(data_format))
        self._data_format = data_format

        # Class (daughters) arguments
        common_kwargs = {
            'kernel_size': kernel_size,
            'padding': padding,
            'data_format': data_format,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'use_bias': use_bias,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint,
        }
        cls_config = {
            'factor': factor,
            'channel_factor': channel_factor,
        }
        cls_config.update(common_kwargs)
        self._cls_config = cls_config

        # Convolutional layer
        self._conv = None  # will be constructed in `build`
        conv_kwargs = {
            'filters': None,  # filed by `build` method
            'strides': self._factor,
            'dilation_rate': 1,
        }
        conv_kwargs.update(common_kwargs)
        self._conv_kwargs = conv_kwargs
        if dim == 2:
            if method == 'down':
                self._conv_cls = Conv2D
            elif method == 'up':
                self._conv_cls = Conv2DTranspose
            else:
                raise ValueError
        elif dim == 3:
            if method == 'down':
                self._conv_cls = Conv3D
            elif method == 'up':
                self._conv_cls = Conv3DTranspose
            else:
                raise ValueError
        else:
            raise ValueError

    def build(self, input_shape):

        # Extract properties
        conv_kwargs = self._conv_kwargs
        conv_cls = self._conv_cls
        factor = self._factor
        channel_factor = self._channel_factor
        data_format = self._data_format
        method = self._method

        # Get channel axis
        channel_axis = get_channel_axis(data_format=data_format)

        # TODO: add check on input_shape (for proper ndim)

        # Check output shape w.r.t. `factor`
        input_shape_list = input_shape.as_list()
        _arr_shape = list(input_shape_list)
        _arr_shape.pop(channel_axis)  # remove channel axis from test
        _arr_shape.pop(0)  # remove batch size from test
        if np.any(np.array(_arr_shape) // factor == 0):
            err_msg = 'Incompatible factor resulting in a 0 output dim'
            raise ValueError(err_msg)

        # Compute output channel number
        inp_chan_nb = input_shape_list[channel_axis]
        if method == 'up':
            out_chan_nb = inp_chan_nb // channel_factor
            if out_chan_nb == 0:
                err_msg = 'Incompatible channel factor resulting in a 0 output dim'
                raise ValueError(err_msg)
        elif method == 'down':
            out_chan_nb = inp_chan_nb * channel_factor
        else:
            raise ValueError  # should never happen

        # Create convolutional layer
        conv_kwargs['filters'] = out_chan_nb
        self._conv = conv_cls(**conv_kwargs)

    def call(self, inputs):
        return self._conv(inputs)

    def get_config(self):

        # Get base config
        config = super(_MultiscaleLayer, self).get_config()

        # Update with class config
        cls_config = self._cls_config
        config.update(cls_config)

        return config


class UpScaling2D(_MultiscaleLayer):

    def __init__(
            self,
            factor: int,
            kernel_size,
            data_format='channels_last',
            channel_factor: Optional[int] = None,  # defaults to `factor`
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            trainable=True,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(UpScaling2D, self).__init__(
            dim=2,
            method='up',
            factor=factor,
            kernel_size=kernel_size,
            data_format=data_format,
            channel_factor=channel_factor,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )


class UpScaling3D(_MultiscaleLayer):

    def __init__(
            self,
            factor: int,
            kernel_size,
            data_format='channels_last',
            channel_factor: Optional[int] = None,  # defaults to `factor`
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            trainable=True,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(UpScaling3D, self).__init__(
            dim=3,
            method='up',
            factor=factor,
            kernel_size=kernel_size,
            data_format=data_format,
            channel_factor=channel_factor,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )


class DownScaling2D(_MultiscaleLayer):

    def __init__(
            self,
            factor: int,
            kernel_size,
            data_format='channels_last',
            channel_factor: Optional[int] = None,  # defaults to `factor`
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            trainable=True,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(DownScaling2D, self).__init__(
            dim=2,
            method='down',
            factor=factor,
            kernel_size=kernel_size,
            data_format=data_format,
            channel_factor=channel_factor,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )


class DownScaling3D(_MultiscaleLayer):

    def __init__(
            self,
            factor: int,
            kernel_size,
            data_format='channels_last',
            channel_factor: Optional[int] = None,  # defaults to `factor`
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            trainable=True,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(DownScaling3D, self).__init__(
            dim=3,
            method='down',
            factor=factor,
            kernel_size=kernel_size,
            data_format=data_format,
            channel_factor=channel_factor,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )
