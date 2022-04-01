import numpy as np
from typing import Dict, List
from collections import Counter

from utils.types import TPath
from utils.training.manager import create_run_manager, RunManager
from utils.training.configs import RunConfig
from utils.training.configs import MappingConfig, TrainingConfig, NetworkConfig

SLICER = None
TRAIN_START, TRAIN_STOP = 0, 30000
VALID_START, VALID_STOP = TRAIN_STOP, 30500
STEP = 1
NUM_PARALLEL_CALLS = 2
TRAIN_REPEAT = -1
PREFETCH_SIZE = 1
TRAIN_SHUFFLE = True
VALID_SHUFFLE = True
TRAIN_VERBOSE = 1
PADDINGS = [[6, 6], [0, 0]]


def get_numerical_test_phantom_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    loss_seq = ['mse', 'mae', 'mslae', 'mslae']
    chan_seq = [16, 16, 16, 32]

    # Create managers
    model_managers = {}

    for loss, chan in zip(loss_seq, chan_seq):
        # Create short key for this comparison
        k = f"{loss}-{chan}"

        # Create RunConfig
        run_cfg = _create_run_config(loss=loss, channel_number=chan)

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_experimental_test_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    loss_seq = ['mslae']
    chan_seq = [16]

    # Create managers
    model_managers = {}

    for loss, chan in zip(loss_seq, chan_seq):
        # Create short key for this comparison
        k = f"{loss}-{chan}"

        # Create RunConfig
        run_cfg = _create_run_config(loss=loss, channel_number=chan)

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_test_samples_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    inp_sig_seq = ['iq', 'iq', 'env', 'bm']
    out_name_seq = ['hq', 'uq', 'uq', 'uq']
    loss_seq = ['mslae', 'mslae', 'mslae', 'mae']

    # Create managers
    model_managers = {}

    for inp_sig, out_name, loss in zip(inp_sig_seq, out_name_seq, loss_seq):
        # Create short key for this comparison
        k = f"{inp_sig}-{out_name}-{loss}"

        # Create RunConfig
        run_cfg = _create_run_config(
            input_signal=inp_sig,
            output_name=out_name,
            output_signal=inp_sig,
            loss=loss,
        )

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_image_representation_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_image_representation_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        inp_sig = run_cfg.mapping_config.input_signal
        out_name = run_cfg.mapping_config.output_name
        loss = run_cfg.training_config.loss
        loss = 'mslae' if loss.startswith('mslae') else loss
        loss = 'mmuae' if loss.startswith('mmuae') else loss
        k = f"{inp_sig}-{out_name}-{loss}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_reference_image_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_reference_image_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        out_name = run_cfg.mapping_config.output_name
        k = f"{out_name}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_training_loss_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_training_loss_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        loss = run_cfg.training_config.loss
        loss = 'mslae' if loss.startswith('mslae') else loss
        loss = 'mmuae' if loss.startswith('mmuae') else loss
        k = f"{loss}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_conv_block_and_skip_conn_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_conv_block_and_skip_conn_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        rb = run_cfg.network_config.residual_block
        skip = run_cfg.network_config.skip_connection
        k = f"{'rcb' if rb else 'fcb'}-{skip}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_channel_number_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_channel_number_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        chan = run_cfg.network_config.channel_number
        k = f"ch{chan}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_train_size_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_train_size_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        ts = run_cfg.training_config.train_size
        k = f"{ts}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_kernel_init_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_kernel_init_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        bki = run_cfg.network_config.block_kernel_initializer
        k = f"{bki}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_learning_rate_managers(
        base_path: TPath
) -> Dict[str, RunManager]:

    # Get corresponding run configurations
    run_cfg_seq = _get_learning_rate_configs()

    # Create managers
    model_managers = {}

    for run_cfg in run_cfg_seq:
        # Create short key for this comparison
        lr = run_cfg.training_config.learning_rate
        k = f"{lr:.1e}"

        # Create RunManager
        run_mgr = create_run_manager(base_path=base_path, config=run_cfg)
        _assert_trained(manager=run_mgr)

        # Store
        model_managers[k] = run_mgr

    return model_managers


def get_all_unique_managers(base_path: TPath) -> List[RunManager]:

    run_cfg_seq = _get_all_unique_configs()
    run_mgr_seq = [
        create_run_manager(base_path=base_path, config=run_cfg)
        for run_cfg in run_cfg_seq]

    return run_mgr_seq


def _get_all_unique_configs() -> List[RunConfig]:

    # Get all run managers
    all_configs = []
    all_configs += _get_image_representation_configs()
    all_configs += _get_reference_image_configs()
    all_configs += _get_training_loss_configs()
    all_configs += _get_conv_block_and_skip_conn_configs()
    all_configs += _get_channel_number_configs()
    all_configs += _get_train_size_configs()
    all_configs += _get_kernel_init_configs()
    all_configs += _get_learning_rate_configs()

    # Make unique (RunConfig is hashable)
    #  Set (does not preserve the initial order (hash randomization))
    # run_cfg_seq = list(set(all_configs))
    #  Counter (preserves the initial order)
    run_cfg_seq = list(Counter(all_configs))

    return run_cfg_seq


def _get_image_representation_configs() -> List[RunConfig]:

    inp_sig_seq = ['rf', 'iq', 'env', 'bm']
    out_name_seq = 4 * ['uq']
    loss_seq = ['mslae', 'mslae', 'mslae', 'mae']

    run_cfg_seq = []
    for inp_sig, out_name, loss in zip(inp_sig_seq, out_name_seq, loss_seq):
        run_cfg = _create_run_config(
            input_signal=inp_sig,
            output_name=out_name,
            output_signal=inp_sig,
            loss=loss,
        )
        run_cfg_seq.append(run_cfg)

    return run_cfg_seq


def _get_reference_image_configs() -> List[RunConfig]:

    out_name_seq = ['uq', 'hq']
    run_cfg_seq = [
        _create_run_config(output_name=out_name) for out_name in out_name_seq]

    return run_cfg_seq


def _get_training_loss_configs() -> List[RunConfig]:

    loss_seq = ['mslae', 'mae', 'mse']
    run_cfg_seq = [_create_run_config(loss=loss) for loss in loss_seq]

    return run_cfg_seq


def _get_conv_block_and_skip_conn_configs() -> List[RunConfig]:

    rb_seq = [True, False, False]
    skip_seq = ['add', 'add', 'concat']
    run_cfg_seq = [
        _create_run_config(residual_block=rb, skip_connection=skip)
        for rb, skip in zip(rb_seq, skip_seq)]

    return run_cfg_seq


def _get_channel_number_configs() -> List[RunConfig]:

    chan_seq = [8, 16, 32]
    run_cfg_seq = [
        _create_run_config(channel_number=chan) for chan in chan_seq]

    return run_cfg_seq


def _get_train_size_configs() -> List[RunConfig]:

    ts_seq = np.round(np.logspace(np.log10(200), np.log10(30000), 8))
    ts_seq = [int(ts) for ts in ts_seq]
    run_cfg_seq = [_create_run_config(train_size=ts) for ts in ts_seq]

    return run_cfg_seq


def _get_kernel_init_configs() -> List[RunConfig]:

    bki_seq = ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']
    run_cfg_seq = [
        _create_run_config(block_kernel_initializer=bki) for bki in bki_seq]

    return run_cfg_seq


def _get_learning_rate_configs() -> List[RunConfig]:

    lr_seq = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    lr_seq.reverse()
    run_cfg_seq = [_create_run_config(learning_rate=lr) for lr in lr_seq]

    return run_cfg_seq


def _create_run_config(
        # Mapping
        input_name: str = 'lq',
        input_signal: str = 'iq',
        output_name: str = 'uq',
        output_signal: str = 'iq',
        # Training
        loss: str = 'mslae',
        learning_rate: float = 5e-5,
        train_size: int = 30000,
        # Network
        channel_number: int = 16,
        residual_block: bool = True,
        skip_connection: str = 'add',
        block_kernel_initializer: str = 'glorot_uniform',
        channel_kernel_initializer: str = 'glorot_uniform',
        block_size: int = 2,
        level_number: int = 5,
        kernel_size: int = 3,

) -> RunConfig:

    map_cfg = MappingConfig(
        input_name=input_name,
        input_signal=input_signal,
        output_name=output_name,
        output_signal=output_signal,
    )
    train_cfg = TrainingConfig(
        loss=loss,
        learning_rate=learning_rate,
        train_size=train_size,
    )
    ntwrk_cfg = NetworkConfig(
        channel_number=channel_number,
        residual_block=residual_block,
        skip_connection=skip_connection,
        block_kernel_initializer=block_kernel_initializer,
        channel_kernel_initializer=channel_kernel_initializer,
        block_size=block_size,
        level_number=level_number,
        kernel_size=kernel_size,
    )
    run_cfg = RunConfig(
        mapping_config=map_cfg,
        training_config=train_cfg,
        network_config=ntwrk_cfg
    )

    return run_cfg


def _assert_trained(manager: RunManager) -> None:
    if not manager.exists():
        raise RuntimeError(
            f"The following run configuration does not exist:"
            f"\n{manager.run_config}"
        )
    if not manager.is_trained():
        raise RuntimeError(
            f"The following run configuration is not trained:"
            f"\n{manager.run_config}"
        )
