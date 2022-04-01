import os
import pickle
from tensorflow.python.keras.models import Model


def get_custom_objects(model: Model) -> dict:

    if not isinstance(model, Model):
        raise ValueError('Must be a keras Model')

    custom_objects = {}
    for layer in model.layers:
        layer_cls = layer.__class__
        layer_cls_name = layer_cls.__name__
        if 'keras' not in layer_cls.__module__:
            custom_objects[layer_cls_name] = layer_cls

    return custom_objects


def save_custom_objects(model: Model, path='custom_objects.pickle') -> None:

    # TODO: not very safe, maybe other possibility with model sub-classing
    custom_objects = get_custom_objects(model=model)

    # Dump config dict as pickle
    with open(path, 'wb') as handle:
        pickle.dump(custom_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_custom_objects(path='custom_objects.pickle') -> dict:

    if not os.path.isfile(path):
        raise FileNotFoundError

    with open(path, 'rb') as handle:
        custom_objects = pickle.load(handle)

    return custom_objects
