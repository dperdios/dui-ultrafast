import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Add, Concatenate
from typing import Sequence

from .utils import get_channel_axis


class _SkipConnection(Layer):

    def __init__(
            self,
            method: str,
            data_format='channels_last',
            name=None,
            trainable=False,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(_SkipConnection, self).__init__(
            name=name, trainable=trainable, dtype=dtype, **kwargs
        )

        # Check inputs
        #   Method
        if not isinstance(method, str):
            raise TypeError('Must be a `str`')
        method = method.lower()
        if not method in ('add', 'concat'):
            raise ValueError("Must be either 'add' or 'concat'")
        self._method = method
        #   Data format
        if not isinstance(data_format, str):
            raise TypeError('Must be a `str`')
        data_format = data_format.lower()
        if data_format not in ('channels_first', 'channels_last'):
            raise ValueError('Unsupported data format {}'.format(data_format))
        self._data_format = data_format

        # Skip connection class
        self._skip_connection = None  # instantiated in `build`

        # Class config
        self._cls_config = {
            # 'method': method,  # TODO: provide serialization here too?
            'data_format': data_format
        }

    def build(self, input_shape):
        # Note:
        #   Best practice: deferring weight creation until the shape of the
        #   inputs is known

        # Check inputs
        if len(input_shape) != 2:
            raise ValueError('Must be a length-2 sequence')

        # Extract properties
        method = self._method
        data_format = self._data_format

        # TODO: add checks on shapes

        # Create skip connection layer
        if method == 'add':
            self._skip_connection = Add()
        elif method == 'concat':
            channel_axis = get_channel_axis(data_format=data_format)
            self._skip_connection = Concatenate(axis=channel_axis)
        else:
            raise RuntimeError()  # should never happen (checked at construction)

    def call(self, inputs: Sequence[tf.Tensor]):

        # TODO: add test on sequence of tensor?
        # if not all([isinstance(t, tf.Tensor) for t in inputs]):
        #     raise ValueError('Must be a sequence of `tf.Tensor`')

        # Apply skip connection
        return self._skip_connection(inputs)

    def get_config(self):

        # Get base config
        config = super(_SkipConnection, self).get_config()

        # Update with class config
        cls_config = self._cls_config
        config.update(cls_config)

        return config


class SkipAdd(_SkipConnection):

    def __init__(
            self,
            data_format='channels_last',
            name=None,
            trainable=False,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(SkipAdd, self).__init__(
            method='add',
            data_format=data_format,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )


class SkipConcat(_SkipConnection):

    def __init__(
            self,
            data_format='channels_last',
            name=None,
            trainable=False,
            dtype=None,
            **kwargs
    ):

        # Call super constructor
        super(SkipConcat, self).__init__(
            method='concat',
            data_format=data_format,
            name=name,
            trainable=trainable,
            dtype=dtype,
            **kwargs
        )
