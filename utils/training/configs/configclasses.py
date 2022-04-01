import dataclasses
from dataclasses import dataclass
from typing import Union, Sequence
import json
import tensorflow.python.keras as K

from dui.losses import MSLAE, MMUAE


@dataclass(frozen=True)
class _BaseConfig:

    def __post_init__(self):

        # Check types of arguments passed to constructor
        obj_fields = dataclasses.fields(self)
        for obj_field in obj_fields:
            attr_name = obj_field.name
            attr_type = obj_field.type
            if attr_type.__module__ == 'typing':
                raise NotImplementedError()
            attr_type_name = attr_type.__name__
            attr_val = self.__getattribute__(attr_name)
            if not isinstance(attr_val, attr_type):
                cls_name = self.__class__.__name__
                err = (
                    f"Argument '{attr_name}' to instantiate a '{cls_name}' "
                    f"must be a '{attr_type_name}'"
                )
                raise TypeError(err)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        cfg_dict = self.as_dict()
        return json.dumps(obj=cfg_dict, **kwargs)

    # TODO: from_dict? attention with RunConfig
    # TODO: from_config?


@dataclass(frozen=True)
class MappingConfig(_BaseConfig):
    """Signal mapping configurations"""
    input_name: str
    input_signal: str
    output_name: str
    output_signal: str


@dataclass(frozen=True)
class NetworkConfig(_BaseConfig):
    """Network configuration with defaults

    Note: no `input_shape` as not a "full" keras Model config
    """
    channel_number: int = 16
    level_number: int = 5
    block_size: int = 2
    # TODO: check with Sequence of int for kernel size
    kernel_size: int = 3
    # TODO: check with Sequence of int for multiscale factor
    multiscale_factor: int = 2
    channel_factor: int = 2
    residual: bool = True
    residual_block: bool = True
    skip_connection: str = 'add'
    padding: str = 'same'
    data_format: str = 'channels_last'
    activation: str = 'relu'
    use_bias: bool = True
    block_kernel_initializer: str = 'glorot_uniform'
    block_bias_initializer: str = 'zeros'
    channel_kernel_initializer: str = 'glorot_uniform'
    channel_bias_initializer: str = 'zeros'
    dtype: str = 'float32'

    # TODO: create dui utility function to check all input parameters
    #  that could be called when checking network parameters before launching
    #  large training loops
    # def __post_init__(self):
    #
    #     # Useful check to prevent trying to build with an error but prevents
    #     # from using a frozen set...
    #     if self.skip_connection == 'concat' and self.residual_block:
    #         raise ValueError('Incompatible skip connection and residual block')

    def get_model_config(self, input_shape: Sequence[int]) -> dict:
        model_config = self.as_dict()
        model_config['input_shape'] = input_shape
        return model_config


# TODO: valid config?
@dataclass(frozen=True)
class TrainingConfig(_BaseConfig):
    """Training configuration with defaults"""
    loss: str = 'mse'
    optimizer: str = 'adam'
    learning_rate: float = 5e-5
    batch_size: int = 2
    step_number: int = 500000
    steps_per_epoch: int = 1000
    train_size: int = 30000
    seed: int = 5250  # you know why

    def __post_init__(self):

        # Little hack to have a default value on specific loss cases
        # while keeping the frozen=True property of the dataclass
        loss = self.loss
        if loss.startswith('mslae'):
            if loss == 'mslae':
                self.__dict__['loss'] = 'mslae62'  # default for mslae
        if loss.startswith('mmuae'):
            if loss == 'mmuae':
                self.__dict__['loss'] = 'mmuae62'  # default for mmuae

    @property
    def optimizer_identifier(self):
        # To be used with tf.keras.optimizers.get(identifier)
        identifier = {
            'class_name': self.optimizer,
            'config': {
                'learning_rate': self.learning_rate
            }
        }
        return identifier

    @property
    def epochs(self):
        # Compute total number of "epochs" in the Keras sense
        step_number = self.step_number
        steps_per_epoch = self.steps_per_epoch
        epochs = (step_number - 1) // steps_per_epoch + 1
        return epochs

    @property
    def iteration_number(self):
        # Compute exact iteration number (as epochs may be "floored")
        return self.epochs * self.steps_per_epoch

    @property
    def iteration_sample_number(self):
        # Compute sample iteration number (i.e. samples seen during training)
        return self.iteration_number * self.batch_size

    def get_loss(self) -> Union[str, K.losses.Loss]:

        # Get loss property
        # loss = self._get_checked_loss()
        loss = self.loss

        # Specific losses provided
        # TODO: should use proper serialization here...
        if 'mslae' in loss:
            min_value_db_str = loss.replace('mslae', '')
            min_value_db = -float(min_value_db_str)
            min_value = 10 ** (min_value_db / 20)
            loss = MSLAE(min_value=min_value)
        elif 'mmuae' in loss:
            min_value_db_str = loss.replace('mmuae', '')
            min_value_db = -float(min_value_db_str)
            mu = 10 ** (-min_value_db / 20)
            loss = MMUAE(mu=mu)
        # TODO: Use tf loss identifier otherwise
        #   Note: uses deserialization then...
        # else:
        #     loss = K.losses.get(identifier)

        return loss


@dataclass(frozen=True)
class RunConfig(_BaseConfig):
    mapping_config: MappingConfig
    training_config: TrainingConfig
    network_config: NetworkConfig
