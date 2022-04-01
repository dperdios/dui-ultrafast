import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from typing import Union, Sequence, Optional


class HDF5DatasetFactory:

    def __init__(
            self,
            path: Union[str, Path],
            name: str,
            start: int = 0,
            stop: Optional[int] = None,
            step: int = 1,
            # TODO: slicer -> Sequence[Union[int, List, slice]]?
            slicer: Optional[Sequence[slice]] = None,
            load: bool = False,
            shuffle: bool = False,
            seed: Optional[int] = None,
    ):

        # Path
        path = Path(path).absolute()
        if not path.is_file():
            raise FileNotFoundError("File '{}' does not exist".format(path))
        self._path = path

        # Dataset name
        if not isinstance(name, str):
            raise TypeError('Dataset name must be a string')
        self._name = name

        # Slicer
        # TODO: check slicer
        slicer = slicer or (slice(None),)  # default slicer
        self._slicer = slicer

        # Check remaining inputs
        if not isinstance(start, int):
            raise TypeError('Start must be a `int`')
        if start < 0:
            raise ValueError('Start must be >= 0')
        if stop is not None:
            if not isinstance(stop, int):
                raise TypeError('Stop must be a `int`')
        if not isinstance(step, int):
            raise TypeError('Step must be a `int`')
        if not isinstance(load, bool):
            raise TypeError('Load must be a `bool`')
        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a `bool`')
        self._shuffle = shuffle
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError('Must be a `int`')
        self._seed = seed

        # Create HDF5 file object reader
        reader = h5py.File(path, mode='r')
        self._is_closed = False
        self._reader = reader
        # TODO: need to implement proper destructor?

        # Check dataset name existence and type
        dataset = reader.get(name)
        if dataset is None:
            name_seq = []
            reader.visit(name_seq.append)
            dset_name_seq = [
                k for k in name_seq if isinstance(reader[k], h5py.Dataset)
            ]
            valid_names = ', '.join(["'{}'".format(s) for s in dset_name_seq])
            raise KeyError(
                "'{}' does not exist in '{}'. "
                "Valid names (keys): {}".format(name, path, valid_names)
            )
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("'{}' must link to an HDF5 Dataset".format(name))
        self._dataset = dataset

        dset_shape = dataset.shape
        self._dset_shape = dset_shape
        dtype = dataset.dtype.name
        self._dtype = dtype
        dset_len = dset_shape[0]
        self._dset_len = dset_len
        base_sample_shape = dset_shape[1:]
        self._base_sample_shape = base_sample_shape

        # Compute nbytes
        size = dset_len * np.prod(base_sample_shape)
        self._nbytes = size * np.dtype(dtype).itemsize

        # Check indexes
        stop = stop or dset_len
        # TODO: improve tests
        if start >= stop:
            ind_err = "Index start ({}) must be < stop ({})".format(start, stop)
            raise IndexError(ind_err)
        if stop > dset_len:
            err = "Index stop ({}) must be <= dataset length ({})".format(
                stop, dset_len
            )
            raise IndexError(err)
        indexes = np.arange(start=start, stop=stop, step=step)
        self._indexes = indexes

        # Output sample shape
        empty_sample = np.empty(base_sample_shape)
        try:
            dumb_output_sample = empty_sample[slicer]
        except IndexError:
            ind_err = (
                "Incompatible indexing slicer {} with dataset sample "
                "of shape {}".format(slicer, base_sample_shape)
            )
            raise IndexError(ind_err)

        # TODO: output_shapes, output_types?
        output_sample_shape = dumb_output_sample.shape
        self._output_sample_shape = output_sample_shape

        # TODO: load option
        if load:
            raise NotImplementedError()

    # Properties
    @property
    def attrs(self):
        return self._dataset.attrs

    # Methods
    def create_dataset(self) -> tf.data.Dataset:

        # Extract properties
        dtype = self._dtype
        indexes = self._indexes
        seed = self._seed
        shuffle = self._shuffle

        # Create tf.data.Dataset
        #   Index data for shuffling and loading
        ind_dataset = tf.data.Dataset.from_tensor_slices(indexes)
        if shuffle:
            ind_dataset = ind_dataset.shuffle(
                buffer_size=indexes.size,
                seed=seed
            )
        dataset = ind_dataset.map(
            map_func=lambda index: tf.numpy_function(
                    self._load_sample, [index], Tout=dtype
                ),
            # Note: not good idea to use parallel calls here
            # num_parallel_calls=num_parallel_calls
        )

        return dataset

    def _load_sample(self, index: int) -> np.ndarray:

        if not isinstance(index, (np.int64, int)):
            raise ValueError('Must be a NumPy int64')

        # Extract properties
        dataset = self._dataset
        slicer = self._slicer

        return dataset[index][slicer]
