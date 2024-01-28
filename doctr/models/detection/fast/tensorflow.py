# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

from doctr.file_utils import CLASS_NAME
from doctr.models.utils import IntermediateLayerGetter, _bf16_to_float32, load_pretrained_params
from doctr.utils.repr import NestedObject

from ...classification import textnet_base, textnet_small, textnet_tiny
from ...modules.layers import FASTConvLayer
from .base import _FAST, FASTPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_small": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_base": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
}


class FastNeck(layers.Layer, NestedObject):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layer.

    Args:
    ----
        in_channels: number of input channels
        out_channels: number of output channels
        upsample_size: size of the upsampling layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        upsample_size: int = 256,
    ) -> None:
        super().__init__()
        self.reduction = [FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]]
        self.upsample = [layers.UpSampling2D(size=scale, interpolation="bilinear") for scale in [2, 4, 8]]

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f, **kwargs) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [upsample(f) for upsample, f in zip(self.upsample, (f2, f3, f4))]
        f = tf.concat((f1, f2, f3, f4), axis=-1)
        return f


class FastHead(Sequential):
    """Head of the FAST architecture

    Args:
    ----
        in_channels: number of input channels
        num_classes: number of output classes
        out_channels: number of output channels
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        _layers = [
            FASTConvLayer(in_channels, out_channels, kernel_size=3),
            layers.Dropout(dropout),
            layers.Conv2D(num_classes, kernel_size=1, use_bias=False),
        ]
        super().__init__(_layers)


class FAST(_FAST, keras.Model, NestedObject):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
    ----
        feature extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization
        dropout_prob: dropout probability
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    _children_names: List[str] = ["feat_extractor", "neck", "head", "postprocessor"]

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.3,
        dropout_prob: float = 0.1,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        class_names: List[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.feat_extractor = feature_extractor
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        # Identify the number of channels for the neck & head initialization
        feat_out_channels = [
            layers.Input(shape=in_shape[1:]).shape[-1] for in_shape in self.feat_extractor.output_shape
        ]
        # Initialize neck & head
        self.neck = FastNeck(feat_out_channels[0], feat_out_channels[1], feat_out_channels[0] * 4)
        self.head = FastHead(feat_out_channels[-1], num_classes, feat_out_channels[1], dropout_prob)

        self.postprocessor = FASTPostProcessor(assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh)

    def _upsample(self, x: tf.Tensor, scale: int = 1) -> tf.Tensor:
        return layers.UpSampling2D(size=scale, interpolation="bilinear")(x)

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: List[Dict[str, np.ndarray]],
    ) -> tf.Tensor:
        # TODO: same as pytorch

        return tf.constant(0.0)

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[Dict[str, np.ndarray]]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        feat_maps = self.feat_extractor(x, **kwargs)
        # Pass through the Neck & Head
        feat_concat = self.neck(feat_maps, **kwargs)
        logits = self.head(feat_concat, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(tf.math.sigmoid(self._upsample(logits, scale=4)))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes (keep only text predictions)
            out["preds"] = [dict(zip(self.class_names, preds)) for preds in self.postprocessor(prob_map.numpy())]

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn,
    feat_layers: List[str],
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = _cfg.get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(
            input_shape=_cfg["input_shape"],
            include_top=False,
            pretrained=pretrained_backbone,
        ),
        feat_layers,
    )

    # Build the model
    model = FAST(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg["url"])

    return model


def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )
