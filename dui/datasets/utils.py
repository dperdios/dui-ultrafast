import tensorflow as tf
import numpy as np
from typing import Union, Optional, Sequence
from pathlib import Path

from dui.datasets.hdf5datasetfactory import HDF5DatasetFactory
from dui.utils.signal import compress_db
from dui.layers.utils import get_channel_axis


def create_image_dataset(
        path: Union[str, Path],
        name: str,
        factor: Union[str, float] = '0db',
        # TODO: None as default or 1?
        signal_type: str = 'rf',
        # TODO: None or 'raw' as default?
        data_format: str = 'channels_last',
        # TODO: patch paddings typing elsewhere if validated
        # paddings: Optional[Union[Sequence[int], np.ndarray]] = None,
        paddings: Optional[Union[Sequence[Sequence[int]], np.ndarray]] = None,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        slicer: Optional[Sequence[slice]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_parallel_calls: Optional[int] = None,
        seed: Optional[int] = None,
) -> tf.data.Dataset:

    # Factory
    dataset_factory = HDF5DatasetFactory(
        path=path,
        name=name,
        start=start,
        stop=stop,
        step=step,
        slicer=slicer,
        shuffle=shuffle,
        seed=seed
    )

    # Check sample shape
    base_sample_shape = dataset_factory._output_sample_shape
    if len(base_sample_shape) != 2:
        raise ValueError(
            "Dataset sample must be a 2D array. Current shape: {}".format(
                base_sample_shape
            )
        )

    # Normalization factor
    if isinstance(factor, str):
        attr_key = factor
        factor = dataset_factory.attrs.get(attr_key)
        if factor is None:
            raise ValueError(
                "No attribute '{}' for dataset '{}' in '{}'".format(
                    attr_key, dataset_factory._dataset.name,
                    dataset_factory._dataset.file.filename
                )
            )
    elif type(factor) in (int, float):
        pass
    else:
        raise TypeError("Unsupported type for 'factor'")

    # Create dataset
    dataset = dataset_factory.create_dataset()
    # TODO: include factor directly and specialize the pre-processing
    #  for US-specific only?

    # Hack to avoid having an <unknown> shape (probably unsafe)
    # TODO: handle this in factory or by sub-classing tf.data.Dataset
    #  Note: Probably below some Dataset._element_structure properties
    #  Note: most probably not compatible with 1.15
    dataset._element_structure._shape = tf.TensorShape(base_sample_shape)

    # Pre-processing
    dataset = dataset.batch(batch_size=batch_size)
    # TODO: use `dataset.padded_batch` instead and remove following
    #  `paddings` option from following pre-processing
    # TODO: apply normalization factor before
    dataset = _preprocess_image_dataset(
        dataset=dataset,
        factor=factor,
        data_format=data_format,
        signal_type=signal_type,
        paddings=paddings,
        num_parallel_calls=num_parallel_calls
    )

    return dataset


def _preprocess_image_dataset(
        dataset: tf.data.Dataset,
        factor: Optional[float] = None,
        data_format: str = 'channels_last',
        signal_type: Optional[str] = None,
        paddings: Optional[Union[Sequence[int], np.ndarray]] = None,
        num_parallel_calls: Optional[int] = None
) -> tf.data.Dataset:

    # Specify pre-processing function as a mapping function
    def map_func(x: tf.Tensor) -> tf.Tensor:
        return _image_preproc_fun(
            x,
            factor=factor,
            data_format=data_format,
            signal_type=signal_type,
            paddings=paddings
        )

    return dataset.map(
        map_func=map_func,
        num_parallel_calls=num_parallel_calls
    )


def _image_preproc_fun(
        x: tf.Tensor,
        factor: Optional[float] = None,
        data_format: str = 'channels_last',
        signal_type: Optional[str] = None,
        paddings: Optional[Union[Sequence[int], np.ndarray]] = None,
) -> tf.Tensor:

    # TODO: check inputs
    x = tf.convert_to_tensor(x)

    # Normalization factor
    if factor:
        # TODO: apply factor before and keep this pre-proc
        #  function only for US-specific transformations?
        x /= factor

    # Paddings
    if paddings is not None:
        # TODO: would probably make more sense to remove paddings
        #  from this US-specific pre-processing function
        # x = _batched_pad(x, paddings=paddings)
        paddings = np.array(paddings)
        valid_pad_shape = 2, 2
        pad_shape = paddings.shape
        # TODO: this test is too restrictive in general (e.g. 3D)
        #  but ok for now as we only work on 2D images
        if pad_shape != valid_pad_shape:
            raise ValueError(
                "Incompatible 'paddings' shape. Current: {}. "
                "Expected {}".format(pad_shape, valid_pad_shape)
            )
        paddings = [[0, 0], *paddings.tolist()]
        pad_kwargs = {
            'paddings': tf.constant(paddings, dtype='int32'),
            'mode': 'CONSTANT',
            'constant_values': 0
        }
        x = tf.pad(x, **pad_kwargs)

    # Channel axis
    channel_axis = get_channel_axis(data_format=data_format)

    # Signal type
    if signal_type is not None:
        if signal_type == 'rf':
            x = tf.math.real(x)
        elif signal_type == 'iq':
            # Stack complex components in channels
            x = tf.stack((tf.math.real(x), tf.math.imag(x)), axis=channel_axis)
        elif signal_type == 'env':
            # Takes modulus of complex IQ signal
            x = tf.math.abs(x)
        elif signal_type == 'bm':
            # Takes modulus of complex IQ signal
            x = tf.math.abs(x)
            # Compress to dB
            x = compress_db(tensor=x)
        elif signal_type == 'raw':
            pass
        else:
            raise ValueError("Invalid signal type")

    # Expand dimension
    if signal_type != 'iq':
        x = tf.expand_dims(x, axis=channel_axis)

    return x
