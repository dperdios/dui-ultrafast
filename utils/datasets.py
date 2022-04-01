import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Union, Sequence, Tuple

from utils.signal import convert_lin_to_db, convert_db_to_lin
from utils.types import TPath, ImageAxes2D


def load_and_preprocess_images(
        path: TPath,
        name: str,
        input_signal: Optional[str] = None,
        input_factor: Optional[Union[str, float]] = None,
        output_signal: Optional[str] = None,
        output_factor: Optional[Union[str, float]] = None,
        # TODO: sample_paddings? (input and/or output?)
        samples_slicer: Optional[Union[slice, int, Sequence[int]]] = None,
) -> Tuple[np.ndarray, ImageAxes2D]:
    """Utility to load HDF5 datasets and transform the signal type.
    Also returns the corresponding image axes.
    """

    # TODO: when converting to dB ('bm'), attention to zero or
    #  negative values
    # TODO: merge with `dui` utils when generalized enough on the
    #  slicing mechanism and signal transformation.
    #  Note: a first step might be to share input checks at least.
    #  Note: considering input and output slicers could be helpful

    # Check path
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File '{path}' does not exist")

    # Check dataset name existence and type
    if not isinstance(name, str):
        raise TypeError("Dataset name must be a string")
    with h5py.File(path, mode='r') as h5r:
        dset = h5r.get(name)
        if dset is None:
            name_seq = []
            h5r.visit(name_seq.append)
            dset_name_seq = [
                k for k in name_seq if isinstance(h5r[k], h5py.Dataset)
            ]
            valid_names = ', '.join(["'{}'".format(s) for s in dset_name_seq])
            raise KeyError(
                f"'{name}' does not exist in '{path}'. "
                f"Valid names (keys): {valid_names}"
            )
        if not isinstance(dset, h5py.Dataset):
            raise ValueError(f"'{name}' must link to an HDF5 Dataset")

    # Sample slicing or indexing
    if samples_slicer is None:
        samples_slicer = slice(None)
    #   Get dataset length
    with h5py.File(path, mode='r') as h5r:
        dset = h5r[name]
        dset_len = len(dset)
    if isinstance(samples_slicer, slice):
        # Case check samples slicing
        start, stop, step = (
            samples_slicer.start, samples_slicer.stop, samples_slicer.step
        )
        start = start or 0  # Defaults to first sample (similar to slice(None))
        if not isinstance(start, int):
            raise TypeError("Start must be a 'int'")
        if start < 0:
            raise ValueError("Start must be >= 0")
        stop = stop or dset_len  # Defaults to dataset length
        if not isinstance(stop, int):
            raise TypeError("Stop must be a 'int'")
        if start >= stop:
            ind_err = f"Index start ({start}) must be < stop ({stop})"
            raise IndexError(ind_err)
        if stop > dset_len:
            err = f"Index stop ({stop}) must be <= dataset length ({dset_len})"
            raise IndexError(err)
        step = step or 1  # Defaults to 1 (similar to slice(None))
        if not isinstance(step, int):
            raise TypeError("Step must be a 'int'")
        if step < 0:
            raise ValueError("Step must be >= 0")
        samples_slicer = slice(start, stop, step)
    else:
        # Indexes
        # TODO: add few checks (e.g., dtype, sorted)
        # samples_slicer = np.asarray(samples_slicer)
        samples_slicer = np.s_[samples_slicer]

    # Normalization factors
    # TODO: probably a bit overkill for output factor
    input_factor = _check_factor(factor=input_factor, path=path, name=name)
    output_factor = _check_factor(factor=output_factor, path=path, name=name)

    # Downsampling slicers
    default_slicer = slice(None), slice(None)
    downsamp_slicer = slice(None), slice(None, None, 2)

    # Check provided input_signal and output_signal
    if input_signal:
        _assert_valid_signal_type(input_signal, name='input_signal')
    if output_signal:
        _assert_valid_signal_type(input_signal, name='output_signal')
    if input_signal is None and output_signal is not None:
        raise ValueError(
            "Invalid combination. Output prescribed without providing input."
        )
    if output_signal is None and input_signal is not None:
        output_signal = input_signal

    inp_out_comb = input_signal, output_signal

    def identity_map(x: np.ndarray) -> np.ndarray:
        return x

    if input_signal == output_signal:
        image_slicer = default_slicer
        map_func = identity_map
    else:
        if input_signal in ('env', 'bm'):
            if output_signal in ('rf', 'iq'):
                comb_err = (
                    "Invalid signal type combination "
                    f"'{input_signal}' -> '{output_signal}'. "
                    "Cannot convert envelope-detected signals back to "
                    "radio-frequency signals."
                )
                raise ValueError(comb_err)
            image_slicer = default_slicer
            # Envelope to B-bmode or B-mode to envelope
            if inp_out_comb == ('env', 'bm'):
                map_func = convert_lin_to_db
            elif inp_out_comb == ('bm', 'env'):
                map_func = convert_db_to_lin
            else:
                raise RuntimeError()  # should never happen
        elif input_signal == 'rf':
            # Not considered in this study
            # TODO: should still be available for generic support
            raise NotImplementedError()
        elif input_signal == 'iq':
            def iq_to_env(x):
                return np.abs(x)

            def iq_to_bm(x):
                return convert_lin_to_db(iq_to_env(x))
            if output_signal == 'rf':
                map_func = np.real
                image_slicer = default_slicer
            elif output_signal == 'env':
                map_func = iq_to_env
                image_slicer = downsamp_slicer
            elif output_signal == 'bm':
                map_func = iq_to_bm
                image_slicer = downsamp_slicer
            else:
                raise RuntimeError()  # should never happen

    with h5py.File(path, mode='r') as h5r:
        dset = h5r[name]
        # Load (faster to slice after loading)
        images = dset[samples_slicer][(Ellipsis, *image_slicer)]
        # Input normalization
        if input_factor:
            images /= input_factor
        # Mapping transform
        images = map_func(images)
        # Output normalization
        if output_factor:
            images /= output_factor
        # Image axes
        slicer_x, slicer_z = image_slicer
        xaxis = dset.attrs['xaxis'][slicer_x]
        zaxis = dset.attrs['zaxis'][slicer_z]
        image_axes = xaxis, zaxis

    return images, image_axes


def _assert_valid_signal_type(t: str, name: str = ''):
    valid_types = 'rf', 'iq', 'env', 'bm'
    name = name or f'{name}'
    if not isinstance(t, str):
        type_err = f"{name} must be a 'str'"
        raise TypeError(type_err)
    if t not in valid_types:
        val_types_str = ", ".join([f"'{t}'" for t in valid_types])
        raise ValueError(f"Invalid {name} '{t}'. Supported: {val_types_str}.")


def _check_factor(
        factor: Union[str, float],
        path: TPath,
        name: str,
) -> float:

    if factor:
        if isinstance(factor, str):
            attr_key = factor
            with h5py.File(path, mode='r') as h5r:
                dset = h5r[name]
                factor = dset.attrs[attr_key]
                if factor is None:
                    fct_err = (
                        f"No attribute '{attr_key}' for dataset '{dset.name}' "
                        f"in '{dset.file.filename}'"
                    )
                    raise ValueError(fct_err)
        elif type(factor) in (int, float):
            pass
        else:
            raise TypeError("Unsupported type for 'factor'.")

    return factor
