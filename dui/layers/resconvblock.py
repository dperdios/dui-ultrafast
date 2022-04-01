from .baseconvblock import ResConvBlock


class ResConv1DBlock(ResConvBlock):

    def __init__(
            self,
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
        super(ResConv1DBlock, self).__init__(
            dim=1,
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


class ResConv2DBlock(ResConvBlock):

    def __init__(
            self,
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
        super(ResConv2DBlock, self).__init__(
            dim=2,
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


class ResConv3DBlock(ResConvBlock):

    def __init__(
            self,
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
        super(ResConv3DBlock, self).__init__(
            dim=3,
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
