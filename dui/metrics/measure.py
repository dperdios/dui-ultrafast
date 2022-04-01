import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


class PSNR(MeanMetricWrapper):

    def __init__(
            self,
            max_val: float,  # see tf.image.psnr `max_value` definition
            name: str = 'psnr',
            dtype=None,
    ):
        # Call super constructor
        fn = tf.image.psnr
        kwargs = {'max_val': max_val}
        super(PSNR, self).__init__(fn=fn, name=name, dtype=dtype, **kwargs)

    # Note: no need to serialize specifically `max_value` as it is taken care
    #   of by MeanMetricWrapper through the `kwargs` passed to it (super
    #   constructor)


class SSIM(MeanMetricWrapper):

    def __init__(
            self,
            max_val: float,  # see `tf.image.ssim` `max_value` definition
            filter_size: int = 11,  # `tf.image.ssim` default
            filter_sigma: float = 1.5,  # `tf.image.ssim` default
            k1: float = 0.01,  # `tf.image.ssim` default
            k2: float = 0.03,  # `tf.image.ssim` default
            name: str = 'ssim',
            dtype=None,
    ):
        # Note: argument defaults are the same as `tf.image.ssim` defaults

        # Call super constructor
        fn = tf.image.ssim
        kwargs = {
            'max_val': max_val,
            'filter_size': filter_size,
            'filter_sigma': filter_sigma,
            'k1': k1,
            'k2': k2,
        }
        super(SSIM, self).__init__(fn=fn, name=name, dtype=dtype, **kwargs)

    # Note: no need to serialize specifically the arguments as it is taken care
    #   off by MeanMetricWrapper through the `kwargs` passed to it (super
    #   constructor)


class ClippedPSNR(PSNR):

    def __init__(
            self,
            vmin: float,
            vmax: float,
            name: str = 'psnr_clipped',
            dtype=None,
    ):
        # Call super constructor
        max_val = vmax - vmin
        super(ClippedPSNR, self).__init__(
            max_val=max_val, name=name, dtype=dtype
        )

        # Add properties
        self._vmin = vmin
        self._vmax = vmax

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract properties
        vmin = self._vmin
        vmax = self._vmax

        # Clip w.r.t. vmin and vmax
        clip_kwargs = {
            'clip_value_min': vmin, 'clip_value_max': vmax
        }
        y_true = tf.clip_by_value(y_true, **clip_kwargs)
        y_pred = tf.clip_by_value(y_pred, **clip_kwargs)

        # Call "super" update
        return super(ClippedPSNR, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    # TODO: serialization


# TODO: mother-class to avoid copy-pasted code
class ClippedSSIM(SSIM):

    def __init__(
            self,
            vmin: float,
            vmax: float,
            filter_size: int = 11,  # `tf.image.ssim` default
            filter_sigma: float = 1.5,  # `tf.image.ssim` default
            k1: float = 0.01,  # `tf.image.ssim` default
            k2: float = 0.03,  # `tf.image.ssim` default
            name: str = 'ssim_clipped',
            dtype=None,
    ):
        # Call super constructor
        max_val = vmax - vmin
        super(ClippedSSIM, self).__init__(
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
            name=name,
            dtype=dtype
        )

        # Add properties
        self._vmin = vmin
        self._vmax = vmax

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract properties
        vmin = self._vmin
        vmax = self._vmax

        # Clip w.r.t. vmin and vmax
        clip_kwargs = {
            'clip_value_min': vmin, 'clip_value_max': vmax
        }
        y_true = tf.clip_by_value(y_true, **clip_kwargs)
        y_pred = tf.clip_by_value(y_pred, **clip_kwargs)

        # Call "super" update
        return super(ClippedSSIM, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    # TODO: serialization


class MappedClippedPSNR(ClippedPSNR):

    def __init__(
            self,
            map_func,
            vmin: float,
            vmax: float,
            name: str = 'mapped_clipped_psnr',
            dtype=None,
    ):

        # Call super constructor
        super(MappedClippedPSNR, self).__init__(
            vmin=vmin, vmax=vmax, name=name, dtype=dtype
        )

        # Add properties
        self._map_func = map_func

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Extract properties
        map_func = self._map_func

        # Apply mapping function before evaluating PSNR
        # TODO: could add clipping too...
        y_true = map_func(y_true)
        y_pred = map_func(y_pred)

        # Call "super" update
        return super(MappedClippedPSNR, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    # TODO: serialization


class MappedClippedSSIM(ClippedSSIM):

    def __init__(
            self,
            map_func,
            vmin: float,
            vmax: float,
            name: str = 'mapped_clipped_ssim',
            dtype=None,
    ):

        # Call super constructor
        super(MappedClippedSSIM, self).__init__(
            vmin=vmin, vmax=vmax, name=name, dtype=dtype
        )

        # Add properties
        self._map_func = map_func

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Extract properties
        map_func = self._map_func

        # Apply mapping function before evaluating SSIM
        # TODO: could add clipping too...
        y_true = map_func(y_true)
        y_pred = map_func(y_pred)

        # Call "super" update
        return super(MappedClippedSSIM, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    # TODO: serialization