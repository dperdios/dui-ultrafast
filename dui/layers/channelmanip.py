import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv1D, Conv2D, Conv3D

from .utils import get_channel_axis


class _ChannelManipulator(Layer):

    def __init__(
            self,
            method: str,
            dim: int,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(_ChannelManipulator, self).__init__(
            name=name, trainable=trainable, dtype=dtype, **kwargs
        )

        # Check inputs
        #   Method
        if not isinstance(method, str):
            raise TypeError('Must be a `str`')
        method = method.lower()
        if not method in ('expansion', 'contraction'):
            raise ValueError("Must be either 'up' or 'down'")
        self._method = method
        #   Dimension
        if not isinstance(dim, int):
            raise TypeError('Must be a `int`')
        if dim not in (1, 2, 3):
            raise ValueError('Must be 1, 2 or 3')
        self._dim = dim
        #   Output channel number
        if not isinstance(channels, int):
            raise TypeError('Must be a `int`')
        if channels < 1:
            raise ValueError('Must be >=1')
        self._channels = channels
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
        cls_config = {'channels': channels}
        cls_config.update(common_kwargs)
        self._cls_config = cls_config

        # Convolutional layer
        self._conv = None  # will be constructed in `build`
        conv_kwargs = {
            'filters': channels,
            'strides': 1,
            'dilation_rate': 1,
        }
        conv_kwargs.update(common_kwargs)
        self._conv_kwargs = conv_kwargs
        if dim == 1:
            self._conv_cls = Conv1D
        elif dim == 2:
            self._conv_cls = Conv2D
        elif dim == 3:
            self._conv_cls = Conv3D
            raise ValueError

    def build(self, input_shape):

        # Extract properties
        conv_kwargs = self._conv_kwargs
        conv_cls = self._conv_cls
        method = self._method
        data_format = self._data_format
        channels = self._channels
        dtype = self.dtype

        # Get channel axis and channel number
        channel_axis = get_channel_axis(data_format=data_format)
        input_shape_list = input_shape.as_list()
        inp_chan_nb = input_shape_list[channel_axis]
        out_chan_nb = channels

        # Specific kernel initializer
        kernel_initializer = conv_kwargs.get('kernel_initializer')
        if kernel_initializer == 'scaled':
            if method == 'expansion':
                _scaling_chan_nb = out_chan_nb
                _corresp_chan_nb = inp_chan_nb
            elif method == 'contraction':
                _scaling_chan_nb = inp_chan_nb
                _corresp_chan_nb = out_chan_nb
            else:
                raise RuntimeError()  # should never happen
            if _corresp_chan_nb != 1:
                raise NotImplementedError("Unsupported 'scaled'")
            value = _scaling_chan_nb * [1 / np.sqrt(_scaling_chan_nb)]
            kernel_initializer = tf.initializers.constant(
                value=value,
                dtype=dtype
            )
            conv_kwargs['kernel_initializer'] = kernel_initializer

        # Check `method` w.r.t. output channel number
        out_chan_nb = channels
        if method == 'expansion' and out_chan_nb < inp_chan_nb:
            raise ValueError('Must use a contraction layer')
        if method == 'contraction' and out_chan_nb > inp_chan_nb:
            raise ValueError('Must use an expansion layer')
        if out_chan_nb == inp_chan_nb:
            raise ValueError('Not a contraction or expansion layer')

        # Create convolutional layer
        self._conv = conv_cls(**conv_kwargs)

    def call(self, inputs):
        return self._conv(inputs)

    def get_config(self):

        # Get base config
        config = super(_ChannelManipulator, self).get_config()

        # Update with class config
        cls_config = self._cls_config
        config.update(cls_config)

        return config


class _ChannelExpansion(_ChannelManipulator):

    def __init__(
            self,
            dim: int,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(_ChannelExpansion, self).__init__(
            method='expansion',
            dim=dim,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class _ChannelContraction(_ChannelManipulator):

    def __init__(
            self,
            dim: int,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(_ChannelContraction, self).__init__(
            method='contraction',
            dim=dim,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelExpansion1D(_ChannelExpansion):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelExpansion1D, self).__init__(
            dim=1,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelExpansion2D(_ChannelExpansion):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelExpansion2D, self).__init__(
            dim=2,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelExpansion3D(_ChannelExpansion):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelExpansion3D, self).__init__(
            dim=3,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelContraction1D(_ChannelContraction):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelContraction1D, self).__init__(
            dim=1,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelContraction2D(_ChannelContraction):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelContraction2D, self).__init__(
            dim=2,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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


class ChannelContraction3D(_ChannelContraction):

    def __init__(
            self,
            channels: int,
            kernel_size: int = 1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=False,
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
        super(ChannelContraction3D, self).__init__(
            dim=3,
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
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
