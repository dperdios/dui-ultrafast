#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import h5py
from pathlib import Path

from dui.datasets.utils import create_image_dataset

from utils.training.run_configs import get_numerical_test_phantom_managers
from utils.training.run_configs import get_test_samples_managers
from utils.training.run_configs import get_experimental_test_managers
from utils.training.run_configs import PADDINGS

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Set TF logging verbosity level
#   Note: allows to disable optimizer's state warnings since the
#   `expect_partial()` load status is not available from `model.load_weights()`
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Shadow potential GPUs to enforce CPU processing
# Note: using CPU-based inference may guarantee exact reproducibility.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -----------------------------------------------------------------------------
# Paths and configs
# -----------------------------------------------------------------------------
# Predictions "suffix"
pred_suffix = 'predictions'

# Datasets to perform predictions on
datasets_dir = Path('./data/datasets/test')
dset_fname_seq = [
    '20200304-ge9ld-numerical-test-phantom.hdf5',
    '20200304-ge9ld-random-phantom-test-set.hdf5',
    '20200527-ge9ld-experimental-test-set-carotid-long.hdf5',
    '20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.hdf5',
]
dset_path_seq = [
    datasets_dir.joinpath(fn).resolve(strict=True) for fn in dset_fname_seq
]

# Additional scale factor
scale_factor_seq = [
    3,  # Calibration scale factor for the numerical test phantom
    1,
    1,
    1,
]

# Trained model managers for each dataset
trained_models_dir = Path('./data/trained-models')
model_managers_seq = [
    get_numerical_test_phantom_managers(trained_models_dir),
    get_test_samples_managers(trained_models_dir),
    get_experimental_test_managers(trained_models_dir),
    get_experimental_test_managers(trained_models_dir),
]

# Predictions directory
predictions_dir = datasets_dir

# -----------------------------------------------------------------------------
# Check export prediction paths
# -----------------------------------------------------------------------------
predictions_dir = predictions_dir.resolve()
if not predictions_dir.is_dir():
    predictions_dir.mkdir()
    print(f"Successfully created directory '{predictions_dir}'")

pred_path_seq = []
pred_exist_seq = []
for dset_path in dset_path_seq:
    pred_path = predictions_dir.joinpath(dset_path.stem + '-' + pred_suffix)
    pred_path = pred_path.with_suffix(dset_path.suffix)
    pred_path = pred_path.resolve(strict=False)
    pred_path_seq.append(pred_path)
    if pred_path.is_file():
        pred_exist_seq.append(pred_path)

input_str_seq = [
    f"The following {len(pred_path_seq)} HDF5 file(s) will be created:"
]
for fp in pred_path_seq:
    input_str_seq.append(f"  '{fp}'")
input_str_seq.append("Do you want to continue (y / [n]): ")
input_str = '\n'.join(input_str_seq)
usr_answer = input(input_str).lower() or 'n'
if usr_answer != 'y':
    raise InterruptedError()

if pred_exist_seq:
    input_str_seq = [
        f"The following {len(pred_exist_seq)} HDF5 file(s) already exist "
        f"and will be overwritten:"
    ]
    for fp in pred_exist_seq:
        input_str_seq.append(f"  '{fp}'")
    input_str_seq.append("Do you want to continue (y / [n]): ")
    input_str = '\n'.join(input_str_seq)
    usr_answer = input(input_str).lower() or 'n'
    if usr_answer != 'y':
        raise InterruptedError()
    else:
        # Explicitly remove files
        for fp in pred_exist_seq:
            fp.unlink()

# Create (or truncate) all predictions files
for fp in pred_path_seq:
    with h5py.File(fp, 'w'):
        pass

# -----------------------------------------------------------------------------
# Compute predictions
# -----------------------------------------------------------------------------
# Note: loop through datasets performed first to maximize cache efficacy
log_sep = 80 * '#'
sub_log_sep = 80 * '-'

# Paddings and corresponding post-inference slicer
#   Note: keeps batch and channel axes
image_crop_slicer = tuple(
    [slice(p[0], -p[1]) if tuple(p) != (0, 0) else slice(None)
     for p in PADDINGS]
)
pred_batch_slicer = slice(None), *image_crop_slicer


# Loop through datasets and corresponding model managers
for dset_path, pred_path, model_managers, scale_factor in zip(
        dset_path_seq, pred_path_seq, model_managers_seq, scale_factor_seq):

    # Info
    print(log_sep)
    print(f"Processing test dataset '{dset_path}'")

    # Loop through trained models
    for model_name, run_mgr in model_managers.items():

        # Info
        print(sub_log_sep)
        print(f"Processing trained model '{model_name}'")

        # Make sure to reset the session
        tf.keras.backend.clear_session()

        # Load trained model (best validation)
        model = run_mgr.load_trained_model()

        # Extract model properties
        input_signal = run_mgr.run_config.mapping_config.input_signal
        output_signal = run_mgr.run_config.mapping_config.output_signal
        input_name = run_mgr.run_config.mapping_config.input_name

        # Dataset
        dset_grp = 'images'
        dset_name = '/'.join([dset_grp, input_name])
        with h5py.File(dset_path, 'r') as h5r:
            # HDF5 Dataset
            dset = h5r[dset_name]
            dset_attrs = dict(dset.attrs)
            # Image axes
            inp_xaxis = dset_attrs['xaxis']
            inp_zaxis = dset_attrs['zaxis']
        #   Image slicer
        if input_signal in ('rf', 'iq'):
            inp_image_slicer = slice(None), slice(None)
        elif input_signal in ('env', 'bm'):
            inp_image_slicer = slice(None), slice(None, None, 2)
        else:
            valid_st = 'rf', 'iq', 'env', 'bm'
            valid_st_str = ", ".join(f"'{s}'" for s in valid_st)
            err_msg = (
                f"Unsupported signal type '{input_signal}'. "
                f"Supported signal types: {valid_st_str}"
            )
            raise ValueError(err_msg)

        # Create TF Dataset
        batch_size = 1
        tf_dataset = create_image_dataset(
            path=dset_path,
            name=dset_name,
            signal_type=input_signal,
            slicer=inp_image_slicer,
            paddings=PADDINGS,
            batch_size=batch_size,
        )

        # Perform inferences
        pred_seq = []
        for inp in tf_dataset:
            # Scale factor
            inp *= scale_factor

            # Predict
            pred = model.predict(inp)

            # Convert back to complex IQ representation
            if output_signal == 'iq':
                pred = pred[..., [0]] + 1j * pred[..., [1]]

            # Post inference cropping
            pred_crop = pred[pred_batch_slicer]

            # Remove channel axis and store
            pred_seq.append(pred_crop[..., 0])

        # Concatenate predictions
        predictions = np.concatenate(pred_seq, axis=0)

        # Export HDF5 dataset settings
        #   Export shape and image axes (keeping the input 3D convention)
        inp_x_slicer, inp_z_slicer = inp_image_slicer
        exp_xaxis = np.atleast_1d(inp_xaxis[inp_x_slicer])
        exp_zaxis = np.atleast_1d(inp_zaxis[inp_z_slicer])
        exp_image_axes = exp_xaxis, exp_zaxis
        exp_image_shape = tuple(ax.size for ax in exp_image_axes)
        exp_dset_shape = len(predictions), *exp_image_shape
        #   Export '0db' factor (1 since no need to scale after inference)
        out_fct = type(dset_attrs['0db'])(1)  # same dtype
        #   Dataset attributes kwargs
        exp_dset_kwargs = {
            'name': '/'.join([dset_grp, model_name]),
            'shape': exp_dset_shape,
            'dtype': predictions.dtype,
            'chunks': (1, *exp_image_shape)
        }
        exp_dset_attrs = {
            'xaxis': exp_xaxis,
            'zaxis': exp_zaxis,
            '0db': out_fct,
        }
        # Note: do not copy image axes at group level as they may differ

        # Save predictions
        print(f"Dumping '{model_name}' results to '{pred_path}'")
        with h5py.File(pred_path, 'r+') as h5w:
            # Create h5py.Dataset
            dset = h5w.create_dataset(**exp_dset_kwargs)
            # Dump data by assignment by pre-reshaping predictions
            # Note: much faster than slicing (apparently an h5py issue)
            dset[:] = np.reshape(predictions, newshape=exp_dset_shape)
            # Dump attributes
            for k, v in exp_dset_attrs.items():
                dset.attrs.create(name=k, data=v)
            print(f"Successfully saved '{dset.name}' in '{h5w.filename}'")

print(log_sep)
print('Done')
print(log_sep)
