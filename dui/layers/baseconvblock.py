from tensorflow.python.keras.layers import Layer, Conv1D, Conv2D, Conv3D, Add
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.framework import tensor_shape

from .utils import get_channel_axis


class _BaseConvBlock(Layer):

    def __init__(
            self,
            dim,
            residual: bool,
            conv_number,
            kernel_size,
            filters=None,
            strides=1,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
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

        # TODO: expand channels?
        # TODO: really need these additional **kwargs?

        # Call super constructor
        # Note: not providing name will convert class name
        super(_BaseConvBlock, self).__init__(
            name=name, trainable=trainable, dtype=dtype, **kwargs
        )

        # Check inputs
        #   Filters (used to specify the `filters` arg throughout the block)
        self._filters = filters
        # TODO: proper check on this one
        #   Residual
        if not isinstance(residual, bool):
            raise TypeError('Must be a `bool`')
        self._residual = residual
        #   Dimension
        if not isinstance(dim, int):
            raise TypeError('Must be a `int`')
        if dim < 1:
            raise ValueError('Must be >=1')
        self._dim = dim
        #   Layer number
        if not isinstance(conv_number, int):
            raise TypeError('Must be a `int`')
        if conv_number < 1:
            raise ValueError('Must be >=1')
        self._conv_number = conv_number
        #   Data format (already checked by super constructor
        self._data_format = data_format

        # Class argument (config)
        common_kwargs = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
            'padding': padding,
            'data_format': data_format,
            'dilation_rate': dilation_rate,
            'activation': activation,
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint,
            'trainable': trainable,
            'dtype': dtype
        }
        cls_config = {
            'conv_number': conv_number,
        }
        cls_config.update(common_kwargs)
        self._cls_config = cls_config

        # Conv arguments
        # conv_kwargs = {'filters': None}
        # conv_kwargs.update(common_kwargs)
        conv_kwargs = dict(common_kwargs)

        self._conv_kwargs = conv_kwargs

        # Convolution sequence
        self._conv_seq = []

    def build(self, input_shape):

        # Extract properties
        data_format = self._data_format
        conv_kwargs = self._conv_kwargs
        conv_list = self._conv_seq
        conv_number = self._conv_number
        dim = self._dim
        filters = self._filters

        # Check `input_shape` w.r.t. `rank`
        expected_inp_dim = dim + 2
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if len(input_shape) != expected_inp_dim:
            err_msg = 'Incompatible input shape: {}. '.format(tuple(input_shape))
            err_msg += 'Expect input dimension: {}'.format(expected_inp_dim)
            raise ValueError(err_msg)

        # Get `filters` argument if not provided
        # Note: `tensor_shape.TensorShape(input_shape)` required?
        if filters is None:
            channel_axis = get_channel_axis(data_format=data_format)
            filters = input_shape[channel_axis]
            conv_kwargs['filters'] = filters

        # Fill convolution list
        #   Note: seems better to use Conv1D, Conv2D, Conv3D for standard ranks
        if dim == 1:
            conv_cls = Conv1D
        elif dim == 2:
            conv_cls = Conv2D
        elif dim == 3:
            conv_cls = Conv3D
        else:
            conv_cls = Conv
            conv_kwargs['rank'] = dim
        for _ in range(conv_number):
            conv_list.append(conv_cls(**conv_kwargs))

    def call(self, inputs):

        # Apply convolution sequence
        x = inputs
        for conv in self._conv_seq:
            x = conv(x)

        # Residual connection
        if self._residual:
            return Add()([inputs, x])
        else:
            return x

    def get_config(self):

        # Get base config
        config = super(_BaseConvBlock, self).get_config()

        # Update with class config
        cls_config = self._cls_config
        config.update(cls_config)

        return config


class ConvBlock(_BaseConvBlock):

    def __init__(
            self,
            dim,
            conv_number,
            kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
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
        super(ConvBlock, self).__init__(
            dim=dim,
            residual=False,
            conv_number=conv_number,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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


class ResConvBlock(_BaseConvBlock):

    def __init__(
            self,
            dim,
            conv_number,
            kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
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
        super(ResConvBlock, self).__init__(
            dim=dim,
            residual=True,
            conv_number=conv_number,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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
