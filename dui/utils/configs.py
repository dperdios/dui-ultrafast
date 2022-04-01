import os
import inspect
import json
from typing import Optional
from tensorflow.python.keras.models import Model

try:
    import yaml
except ImportError:
    yaml = None


def save_model_config(model: Model, path: str = 'config.json') -> None:

    if not isinstance(model, Model):
        raise ValueError('Must be a `tensorflow.keras.models.Model`')

    if not isinstance(path, str):
        raise ValueError('Must be a `str`')

    # Get extension
    _, ext = os.path.splitext(path)

    if ext == '.json':
        # Get JSON string
        dump_str = model.to_json(indent=4)  # follow Python indent convention
    elif ext == '.yaml':
        dump_str = model.to_yaml()
    else:
        raise ValueError('Unsupported extension type {}'.format(ext))

    # Dump
    with open(path, 'w') as file:
        file.write(dump_str)


def save_model_summary(
        model: Model, path: str = 'model.summary', line_length: int = 120
) -> None:

    if not isinstance(model, Model):
        raise ValueError('Must be a `tensorflow.keras.models.Model`')

    if not isinstance(path, str):
        raise ValueError('Must be a `str`')

    # Dump
    with open(path, 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))


def get_default_module_params(module) -> dict:

    # TODO: test module type

    # Get module signature
    sig = inspect.signature(module)

    # Inspect module parameters (and default value if any)
    params = {}
    for pk, pv in sig.parameters.items():
        params[pk] = pv.default

    return params


def get_module_params(module, params: Optional[dict] = None):

    params = params or dict()
    if not isinstance(params, dict):
        raise ValueError('Must be a `dict`')

    default_params = get_default_module_params(module=module)

    # Note for merging dicts {**x, **y}: values from y replace those from x
    return {**default_params, **params}


def save_config(config: dict, path: str = 'config.json') -> None:

    if not isinstance(config, dict):
        raise ValueError('Must be a `dict`')

    if not isinstance(path, str):
        raise ValueError('Must be a `str`')

    # Get extension
    _, ext = os.path.splitext(path)

    if ext == '.json':
        # Get JSON string (follow Python indent convention)
        dump_str = json.dumps(config, indent=4)
    elif ext == '.yaml':
        # Get YAML string
        _assert_yaml_installed()
        dump_str = yaml.safe_dump(config)
    else:
        raise ValueError('Unsupported extension type {}'.format(ext))

    # Dump
    with open(path, 'w') as file:
        file.write(dump_str)


def load_config(path: str = 'config.json') -> dict:

    if not os.path.isfile(path=path):
        raise FileNotFoundError('{}'.format(path))

    # Read file (i.e. config str)
    with open(path, 'r') as file:
        config_str = file.read()

    # Get extension
    _, ext = os.path.splitext(path)

    # Check extension
    if ext == '.json':
        config = json.loads(config_str)
    elif ext == '.yaml':
        _assert_yaml_installed()
        config = yaml.safe_load(config_str)
    else:
        raise ValueError('Unsupported extension type {}'.format(ext))

    return config


def _assert_yaml_installed() -> None:
    if yaml is None:
        raise ImportError(
            'Requires yaml module installed (`pip install pyyaml`).'
        )
