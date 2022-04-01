import tensorflow as tf
from tensorflow.python.keras import Model
import glob
import json
import pickle
import pandas
from pathlib import Path
from typing import Tuple, Union

from dui.models.utils import get_custom_objects

from utils.training.configs import MappingConfig, TrainingConfig, NetworkConfig
from utils.training.configs import RunConfig

from utils.types import TPath


class RunManager:
    # TODO: instead of metaclass maybe simply defining class
    #  class properties directly
    # _some_prop  # would allow to call cls._some_prop for instance

    def __init__(
            self,
            path: TPath,
            run_config: RunConfig,
    ):
        # Make sure absolute path
        path = Path(path).resolve()
        self._path = path

        # Check run configuration (set)
        if not isinstance(run_config, RunConfig):
            raise TypeError("Must be a '{}'.".format(RunConfig.__name__))
        self._run_config = run_config

        # Configuration file structure
        #   Configuration directory
        config_dir = path.joinpath('configs')
        self._config_dir = config_dir
        #   Run configuration
        run_config_path = config_dir.joinpath('run_config.json')
        self._run_config_path = run_config_path
        #   Model configuration
        model_config_path = config_dir.joinpath('model_config.json')
        self._model_config_path = model_config_path
        #   Model custom objects
        custom_objects_path = config_dir.joinpath('model_custom_objects.pickle')
        self._model_custom_objects_path = custom_objects_path
        #   Model factory configuration
        model_fctry_cfg_name = 'model_factory_config.json'
        model_fctry_cfg_path = config_dir.joinpath(model_fctry_cfg_name)
        self._model_factory_config_path = model_fctry_cfg_path
        #   Model summary
        summary_path = config_dir.joinpath('model.summary')
        self._model_summary_path = summary_path

        # Checkpoints structure
        ckpt_dir = path.joinpath('checkpoints')
        self._ckpt_dir = ckpt_dir
        ckpt_epochs_dir = ckpt_dir.joinpath('epochs')
        self._ckpt_epochs_dir = ckpt_epochs_dir
        ckpt_bests_dir = ckpt_dir.joinpath('bests')
        self._ckpt_bests_dir = ckpt_bests_dir
        ckpt_epoch_name = 'cp-epoch{epoch:06d}.ckpt'
        ckpt_epoch = ckpt_epochs_dir.joinpath(ckpt_epoch_name)
        self._ckpt_epoch = ckpt_epoch.as_posix()
        ckpt_best_valid_name = 'cp-best-valid.ckpt'
        ckpt_best_valid = ckpt_bests_dir.joinpath(ckpt_best_valid_name)
        self._ckpt_best_valid = ckpt_best_valid.as_posix()

        # Training log
        self._train_log_path = path.joinpath('training.log')

    # Properties
    @property
    def path(self):
        return self._path

    @property
    def run_config(self):
        return self._run_config

    @property
    def config_dir(self):
        return self._config_dir

    @property
    def run_config_path(self):
        return self._run_config_path

    @property
    def model_config_path(self):
        return self._model_config_path

    @property
    def model_custom_objects_path(self):
        return self._model_custom_objects_path

    @property
    def model_factory_config_path(self):
        return self._model_factory_config_path

    @property
    def model_summary_path(self):
        return self._model_summary_path

    @property
    def ckpt_dir(self):
        return self._ckpt_dir

    @property
    def ckpt_bests_dir(self):
        return self._ckpt_bests_dir

    @property
    def ckpt_epochs_dir(self):
        return self._ckpt_epochs_dir

    @property
    def ckpt_epoch(self):
        return self._ckpt_epoch

    @property
    def ckpt_best_valid(self):
        return self._ckpt_best_valid

    @property
    def train_log_path(self):
        return self._train_log_path

    # Methods
    def is_trained(self):

        # Init to false
        is_trained = False

        # Check that log file exists and is non-empty before loading
        log_fp = self.train_log_path
        if log_fp.is_file() and log_fp.stat().st_size != 0:
            train_log = self.load_training_log()
            tot_epoch_number = self.run_config.training_config.epochs
            cur_epoch_number = train_log.epoch.values[-1]

            # Consider fully trained model only if all epochs performed
            if tot_epoch_number == cur_epoch_number:
                is_trained = True

        return is_trained

    def exists(self):
        return self.path.exists()

    def save_configs(self, model: Model) -> None:

        # TODO: merge with or use dui.utils.configs.save.save_model_configs?
        # TODO: sub-methods for each config type?

        if not isinstance(model, Model):
            raise ValueError("Must be a 'tensorflow.keras.models.Model'")

        config_dir = self.config_dir
        if not config_dir.is_dir():
            # TODO: check if parent already exists?
            config_dir.mkdir()

        def assert_file_exists(path: TPath) -> None:
            path = Path(path)
            if path.is_file():
                raise FileExistsError(f"'{path}' already exists.")

        json_indent = 4  # follow Python indent convention

        # Save model config as JSON
        model_config_path = self.model_config_path
        assert_file_exists(path=model_config_path)
        json_str = model.to_json(indent=json_indent)
        with open(model_config_path, 'w') as fh:
            fh.write(json_str)

        # Save custom model objects
        model_custom_objects_path = self.model_custom_objects_path
        assert_file_exists(path=model_custom_objects_path)
        custom_objects = get_custom_objects(model=model)
        with open(model_custom_objects_path, 'wb') as fh:
            pickle.dump(custom_objects, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Save model summary
        model_summary_path = self.model_summary_path
        assert_file_exists(path=model_summary_path)
        with open(model_summary_path, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Save run config
        run_config = self.run_config
        run_config_path = self.run_config_path
        assert_file_exists(path=run_config_path)
        with open(run_config_path, 'w') as fh:
            json.dump(obj=run_config.as_dict(), fp=fh, indent=json_indent)

    def load_training_log(self) -> pandas.DataFrame:

        train_log_path = self.train_log_path
        if not train_log_path.is_file():
            raise FileNotFoundError(f"'{train_log_path}' does not exist.")

        df = pandas.read_csv(train_log_path)

        # Update epoch to start at 1
        df.epoch += 1

        # Compute corresponding iteration and add to DataFrame
        steps_per_epoch = self.run_config.training_config.steps_per_epoch
        iteration = df.epoch * steps_per_epoch
        df.insert(loc=1, column='iteration', value=iteration)

        return df

    def load_trained_model(self, ckpt: Union[str, int] = 'best') -> Model:

        # TODO: wrap model loading in dedicated load_model method?

        # Load model config from JSON
        # TODO: merge with or use dui.utils.configs.load_config?
        model_config_path = self.model_config_path
        if not model_config_path.is_file():
            file_err = f"'{model_config_path}' does not exist."
            raise FileNotFoundError(file_err)
        with open(model_config_path, 'r') as fp:
            model_config = json.load(fp=fp)

        # Load custom objects
        custom_objects_path = self.model_custom_objects_path
        # TODO: merge with or use dui.models.utils.load_custom_objects?
        if not custom_objects_path.is_file():
            file_err = f"'{custom_objects_path}' does not exist."
            raise FileNotFoundError(file_err)
        with open(custom_objects_path, 'rb') as fp:
            custom_objects = pickle.load(file=fp)

        # Create model from config and custom objects
        model = tf.keras.models.model_from_config(
            config=model_config, custom_objects=custom_objects
        )

        # Case checkpoint type
        if isinstance(ckpt, str):
            if ckpt == 'best':
                ckpt = self.ckpt_best_valid
            elif ckpt == 'last':
                ckpt_epochs_dir = self.ckpt_epochs_dir.as_posix()
                ckpt = tf.train.latest_checkpoint(ckpt_epochs_dir)
            else:
                raise NotImplementedError()
        elif isinstance(ckpt, int):
            ckpt = self.ckpt_epoch.format(epoch=ckpt)
        else:
            raise TypeError("Must be a 'str' or 'int'")

        # Load weights
        model.load_weights(ckpt)

        return model

    @classmethod
    def from_json(cls, path: TPath):

        # Check path
        path = Path(path)

        # Deduce base path from JSON configuration path
        # TODO: use some relative path from RunManager to avoid
        #  "hard-coded" path.parent.parent. May require defining a metaclass.
        run_mgr_path = path.parent.parent
        with open(path, 'r') as fp:
            run_cfg_dict = json.load(fp=fp)

        # TODO: RunConfig.from_json? .from_dict? .from_config?
        # Create RunConfig
        #   Get sub-configuration dicts
        map_cfg_dict = run_cfg_dict['mapping_config']
        ntwrk_cfg_dict = run_cfg_dict['network_config']
        train_cfg_dict = run_cfg_dict['training_config']
        #   Create sub-configs
        map_cfg = MappingConfig(**map_cfg_dict)
        train_cfg = TrainingConfig(**train_cfg_dict)
        ntwrk_cfg = NetworkConfig(**ntwrk_cfg_dict)
        #   Create run config
        run_cfg = RunConfig(
            mapping_config=map_cfg,
            training_config=train_cfg,
            network_config=ntwrk_cfg
        )

        # Create run manager
        run_mgr = RunManager(path=run_mgr_path, run_config=run_cfg)

        return run_mgr


def create_run_manager(
        base_path: TPath,
        config: RunConfig,
) -> RunManager:

    # Make sure absolute path
    base_path = Path(base_path).resolve()

    # Generate path
    # TODO: maybe some option for uuid-based paths instead
    #  of named ones which are always fragile
    path = _gen_save_path(base_path=base_path, config=config)

    # Create run manager
    return RunManager(path=path, run_config=config)


def find_run_managers(
        base_path: TPath,
        followlinks: bool = False,
) -> Tuple[RunManager]:

    # Make sure absolute path (and exist)
    base_path = Path(base_path).resolve(strict=True)

    # Find configurations and create corresponding run managers
    run_mgr_seq = []
    # TODO: use automatized relative glob pattern (may require
    #  defining a metaclass)
    run_cfg_rel_path = "configs/run_config.json"
    if not followlinks:
        # Note: Path.rglob does not follow symlinks to prevent potential
        #  infinite loops does not provide an optional flag
        rglob_list = list(base_path.rglob(run_cfg_rel_path))
    else:
        rglob_pattern = base_path.joinpath("**", run_cfg_rel_path).as_posix()
        rglob_list = glob.glob(rglob_pattern, recursive=True)
        # Convert list of str to list of Path for consistency with Path.rglob
        rglob_list = [Path(p) for p in rglob_list]
    rglob_list.sort()
    for path in rglob_list:
        run_mgr = RunManager.from_json(path=path)
        run_mgr_seq.append(run_mgr)

    return tuple(run_mgr_seq)


def find_trained_model_run_managers(
        base_path: TPath,
        followlinks: bool = False,
) -> Tuple[RunManager]:

    # Find all run managers
    all_run_mgr_seq = find_run_managers(
        base_path=base_path, followlinks=followlinks
    )

    # Filter trained managers
    run_mgr_seq = [mgr for mgr in all_run_mgr_seq if mgr.is_trained()]

    return tuple(run_mgr_seq)


def _gen_save_path(
        base_path: TPath,
        config: RunConfig,
) -> Path:

    # Check path
    base_path = Path(base_path)

    # Unpack run configuration
    run_cfg = config
    map_cfg = run_cfg.mapping_config
    network_cfg = run_cfg.network_config
    train_cfg = run_cfg.training_config

    # Generate sub-directory structure
    #   Mapping config
    map_name_fmt = 'inp-{in:}-{it:}-out-{on:}-{ot:}'
    map_name_fmt_cfg = {
        'in': map_cfg.input_name,
        'it': map_cfg.input_signal,
        'on': map_cfg.output_name,
        'ot': map_cfg.output_signal,
    }
    map_name = map_name_fmt.format(**map_name_fmt_cfg).lower()
    #   Training config
    train_name_fmt = (
        '{loss:s}-bs-{bs:d}-{opt:s}-lr-{lr:.1e}-seed-{seed:d}-ts-{ts:05d}'
    )
    train_name_fmt_cfg = {
        'loss': train_cfg.loss,
        'bs': train_cfg.batch_size,
        'opt': train_cfg.optimizer,
        'lr': train_cfg.learning_rate,
        'seed': train_cfg.seed,
        'ts': train_cfg.train_size
    }
    train_name = train_name_fmt.format(**train_name_fmt_cfg).lower()
    #   Network config
    network_name_fmt = (
        'gunet-ch-{ch:d}-ln-{ln:d}-bs-{bs:d}-ks-{ks:d}-'
        'mf-{mf:d}-af-{af:s}-sc-{sc:s}-rb-{rb:}-'
        'bki-{bki:}-bbi-{bbi:}-cki-{cki:}-cbi-{cbi:}'
    )
    network_name_fmt_cfg = {
        'ch': network_cfg.channel_number,
        'ln': network_cfg.level_number,
        'bs': network_cfg.block_size,
        'ks': network_cfg.kernel_size,
        'mf': network_cfg.multiscale_factor,
        'af': network_cfg.activation,
        'sc': network_cfg.skip_connection,
        'rb': network_cfg.residual_block,
        'bki': network_cfg.block_kernel_initializer.replace('_', ''),
        'bbi': network_cfg.block_bias_initializer.replace('_', ''),
        'cki': network_cfg.channel_kernel_initializer.replace('_', ''),
        'cbi': network_cfg.channel_bias_initializer.replace('_', '')
    }
    network_name = network_name_fmt.format(**network_name_fmt_cfg).lower()

    # Build path
    path = base_path.resolve()
    path = path.joinpath(map_name)
    path = path.joinpath(train_name)
    path = path.joinpath(network_name)

    return path.resolve()
