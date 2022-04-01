#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
import tensorflow as tf
from pathlib import Path

from dui.models.gunet_functional import create_gunet_model
from dui.layers.utils import get_channel_axis
from dui.utils.configs import get_module_params, save_config
from dui.utils.signal import bmode_from_rf, compress_db, bmode_from_iq_chan
from dui.metrics import ClippedPSNR, ClippedSSIM
from dui.metrics import MappedClippedPSNR, MappedClippedSSIM
from dui.datasets.utils import create_image_dataset

from utils.training.manager import RunManager
from utils.training.run_configs import get_all_unique_managers
from utils.training.run_configs import TRAIN_START
from utils.training.run_configs import VALID_START, VALID_STOP
from utils.training.run_configs import TRAIN_REPEAT
from utils.training.run_configs import TRAIN_SHUFFLE
from utils.training.run_configs import TRAIN_VERBOSE
from utils.training.run_configs import VALID_SHUFFLE
from utils.training.run_configs import PREFETCH_SIZE, NUM_PARALLEL_CALLS
from utils.training.run_configs import SLICER, STEP, PADDINGS

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
DATA_PATH = './data/datasets/train/20200304-ge9ld-random-phantom.hdf5'
BASE_PATH = './data/trained-models'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-id', type=int, default=None, help='CUDA DEVICE ID')
parser.add_argument(
    '--base-path', type=str, default=BASE_PATH,
    help=f"Base path to save trained models (Default: '{BASE_PATH}')")
parser.add_argument(
    '--data-path', type=str, default=DATA_PATH,
    help=f"Path to training dataset (Default: '{DATA_PATH}')")


# -----------------------------------------------------------------------------
# Main training method
# -----------------------------------------------------------------------------
def main_training_run(
        run_manager: RunManager, data_path: Path, skip_existing: bool = True):

    # Logs
    log_sep = 80 * '#'

    # Check data file exists
    if not data_path.is_file():
        raise FileNotFoundError(f"File '{data_path}' does not exist.")

    # Skip if existing
    if not run_manager.exists():
        run_manager.path.mkdir(parents=True)
        print(f"Successfully created directory '{run_manager.path}'")
    else:
        dir_exist = f'Model directory already exists: {run_manager.path}'
        if skip_existing:
            print(dir_exist)
            print(log_sep)
            print("SKIPPING TRAINING (NOT A ROBUST CHECK)")
            print(log_sep)
            return
        else:
            inp_msg = dir_exist + " Do you want to continue? (y / [n]): "
            usr_answer = input(inp_msg).lower() or 'n'
            if usr_answer != 'y':
                raise InterruptedError()

    # Make sure to reset the session
    tf.keras.backend.clear_session()

    # Extract global run configuration and sub configurations
    run_config = run_manager.run_config
    training_config = run_config.training_config
    network_config = run_config.network_config
    mapping_config = run_config.mapping_config

    # Set graph-level seed (used by initializers + dataset API)
    graph_seed = training_config.seed
    tf.compat.v1.random.set_random_seed(seed=graph_seed)

    # -------------------------------------------------------------------------
    # Training set
    # -------------------------------------------------------------------------
    # Extract mapping properties
    inp_signal = mapping_config.input_signal
    ref_signal = mapping_config.output_signal
    inp_name = mapping_config.input_name
    ref_name = mapping_config.output_name

    # Create datasets using factory
    inp_trainset = create_image_dataset(
        path=data_path,
        name='images/' + inp_name,
        factor='0db',
        signal_type=inp_signal,
        data_format='channels_last',
        paddings=PADDINGS,
        start=TRAIN_START,
        stop=TRAIN_START + training_config.train_size,
        step=STEP,
        slicer=SLICER,
        batch_size=training_config.batch_size,
        shuffle=TRAIN_SHUFFLE,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        seed=training_config.seed,
    )
    ref_trainset = create_image_dataset(
        path=data_path,
        name='images/' + ref_name,
        factor='0db',
        signal_type=ref_signal,
        data_format='channels_last',
        paddings=PADDINGS,
        start=TRAIN_START,
        stop=TRAIN_START + training_config.train_size,
        step=STEP,
        slicer=SLICER,
        batch_size=training_config.batch_size,
        shuffle=TRAIN_SHUFFLE,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        seed=training_config.seed,
    )

    # Store as immutable sequence (tuple)
    trainset_seq = tuple([inp_trainset, ref_trainset])

    # Get network input shape
    inp_out_shape = inp_trainset.output_shapes.as_list()
    ref_out_shape = ref_trainset.output_shapes.as_list()
    out_shape_seq = tuple([inp_out_shape, ref_out_shape])
    if out_shape_seq.count(out_shape_seq[0]) != len(out_shape_seq):
        raise ValueError(f'Incompatible shapes: {out_shape_seq}')
    input_shape = out_shape_seq[0]

    # Zip datasets
    train_dataset = tf.data.Dataset.zip(trainset_seq)
    train_dataset = train_dataset.repeat(count=TRAIN_REPEAT)

    # Set prefetching
    train_dataset = train_dataset.prefetch(buffer_size=PREFETCH_SIZE)

    # -------------------------------------------------------------------------
    # Validation set
    # -------------------------------------------------------------------------
    # Create datasets using factory
    inp_validset = create_image_dataset(
        path=data_path,
        name='images/' + inp_name,
        factor='0db',
        signal_type=inp_signal,
        data_format='channels_last',
        paddings=PADDINGS,
        start=VALID_START,
        stop=VALID_STOP,
        step=STEP,
        slicer=SLICER,
        batch_size=training_config.batch_size,
        shuffle=VALID_SHUFFLE,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        seed=training_config.seed,
    )
    ref_validset = create_image_dataset(
        path=data_path,
        name='images/uq',
        factor='0db',
        signal_type=ref_signal,
        data_format='channels_last',
        paddings=PADDINGS,
        start=VALID_START,
        stop=VALID_STOP,
        step=STEP,
        slicer=SLICER,
        batch_size=training_config.batch_size,
        shuffle=VALID_SHUFFLE,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        seed=training_config.seed,
    )

    # Store as immutable sequence (tuple)
    validset_seq = tuple([inp_validset, ref_validset])

    # Zip datasets
    valid_dataset = tf.data.Dataset.zip(validset_seq)

    # Set prefetching
    valid_dataset = valid_dataset.prefetch(buffer_size=PREFETCH_SIZE)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    # Create complete model config (with input shape)
    model_input_shape = input_shape[1:]  # remove batch dimension
    model_config = {'input_shape': model_input_shape}
    model_config.update(network_config.as_dict())

    # Create (functional) model
    model = create_gunet_model(**model_config)

    # -------------------------------------------------------------------------
    # Create training directory structure and initial dumps
    # -------------------------------------------------------------------------
    # Save model configurations
    run_manager.save_configs(model=model)

    # Save model factory configurations
    model_factory_config = get_module_params(create_gunet_model, model_config)
    save_config(
        config=model_factory_config,
        path=run_manager.model_factory_config_path.as_posix()
    )

    # Log model creation
    print(log_sep)
    print(f'Successfully created {model.name}:')
    print(json.dumps(model_factory_config, indent=4))
    print(f'   Config directory: {os.path.abspath(run_manager.config_dir)}')
    print(f'   Log directory: {os.path.abspath(run_manager.path)}')

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    signal_type = mapping_config.output_signal
    data_format = run_config.network_config.data_format
    # dB-range considered
    vmin_db, vmax_db = -62, +36

    # Create B-mode transformation mapping function
    if signal_type == 'rf':
        channel_axis = get_channel_axis(data_format=data_format)
        filter_size = 33
        beta = 8

        def map_func(tensor: tf.Tensor) -> tf.Tensor:
            return bmode_from_rf(
                tensor=tensor,
                filter_size=filter_size,
                beta=beta,
                data_format=data_format,
                axis=channel_axis,
            )
    elif signal_type == 'iq':
        def map_func(tensor: tf.Tensor) -> tf.Tensor:
            return bmode_from_iq_chan(tensor=tensor, data_format=data_format)
    elif signal_type == 'env':
        def map_func(tensor: tf.Tensor) -> tf.Tensor:
            return compress_db(tensor=tensor)
    elif signal_type == 'bm':
        def map_func(tensor: tf.Tensor) -> tf.Tensor:
            return tensor
    else:
        raise NotImplementedError()

    # B-mode metrics (after transformation)
    mapped_metrics_kwargs = {
        'map_func': map_func,
        'vmin': vmin_db,
        'vmax': vmax_db,
    }
    bmode_psnr_metric = MappedClippedPSNR(**mapped_metrics_kwargs)
    bmode_ssim_metric = MappedClippedSSIM(**mapped_metrics_kwargs)

    # Signal metrics (before transformation)
    vmax_lin = np.power(10, vmax_db / 20)
    if signal_type == 'bm':
        vmax_sig, vmin_sig = vmax_db, vmin_db
    elif signal_type == 'env':
        vmax_sig, vmin_sig = vmax_lin, 0
    else:  # rf or iq
        vmax_sig, vmin_sig = vmax_lin, -vmax_lin
    sig_metrics_config = {'vmin': vmin_sig, 'vmax': vmax_sig}
    sig_psnr_metric = ClippedPSNR(**sig_metrics_config)
    sig_ssim_metric = ClippedSSIM(**sig_metrics_config)

    # Metrics list
    metrics = [
        'mse',
        'mae',
        sig_ssim_metric,
        sig_psnr_metric,
        bmode_psnr_metric,
        bmode_ssim_metric,
    ]

    # -------------------------------------------------------------------------
    # Create callbacks
    # -------------------------------------------------------------------------
    # Create checkpoint callbacks
    #   Step checkpoint
    ckpt_step_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=run_manager.ckpt_epoch,
        verbose=TRAIN_VERBOSE,
        save_weights_only=True,
        save_freq='epoch'
    )
    #   Best checkpoint w.r.t. metrics
    ckpt_best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=run_manager.ckpt_best_valid,
        verbose=TRAIN_VERBOSE,
        save_weights_only=True,  # Note: overwrites
        monitor='val_loss',
        mode='min',  # since monitoring a loss
        save_best_only=True,  # needs `monitor` argument
    )

    # CSV logger
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=run_manager.train_log_path,
        separator=',',
        append=True
    )

    # Callback list
    ckpt_callbacks = [
        ckpt_step_callback,
        ckpt_best_callback,
    ]
    train_callbacks = ckpt_callbacks + [csv_logger]

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    # Create optimizer
    opt_identifier = training_config.optimizer_identifier
    optimizer = tf.keras.optimizers.get(identifier=opt_identifier)

    # Create loss
    loss = training_config.get_loss()

    # Compute model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Log
    print(log_sep)
    print(f"ALL SET RUNNING FOR {training_config.iteration_number} iterations")
    print(log_sep)

    model.fit(
        train_dataset,
        epochs=training_config.epochs,
        steps_per_epoch=training_config.steps_per_epoch,
        validation_data=valid_dataset,
        callbacks=train_callbacks,
        verbose=TRAIN_VERBOSE
    )

    print(log_sep)
    print("TRAINING DONE")
    print(log_sep)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parse arguments
    args = parser.parse_args()
    gpu_id = args.gpu_id
    data_path = Path(args.data_path)
    base_path = Path(args.base_path)

    # Shadow GPUs by setting CUDA_VISIBLE_DEVICES w.r.t. `gpu_id`
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get all run managers
    all_mgrs = get_all_unique_managers(base_path=base_path)

    # Filter out fully trained managers
    run_mgr_seq = [mgr for mgr in all_mgrs if not mgr.is_trained()]

    # Loop over run managers
    for run_mgr in run_mgr_seq:
        main_training_run(run_manager=run_mgr, data_path=data_path)

    print("ALL TRAININGS PERFORMED")
