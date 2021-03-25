# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, Any
from .core import OCRPredictor
from .detection.zoo import detection_predictor
from .recognition.zoo import recognition_predictor


__all__ = ["ocr_predictor"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'db_sar_vgg': {'detection': 'db_resnet50_predictor', 'recognition': 'sar_vgg16_bn_predictor'},
    'db_sar_resnet': {'detection': 'db_resnet50_predictor', 'recognition': 'sar_resnet31_predictor'},
    'db_crnn_vgg': {'detection': 'db_resnet50_predictor', 'recognition': 'crnn_vgg16_bn_predictor'},
    'db_crnn_resnet': {'detection': 'db_resnet50_predictor', 'recognition': 'crnn_resnet31_predictor'},
}


def _predictor(det_arch: str, reco_arch: str, pretrained: bool, det_bs=2, reco_bs=32) -> OCRPredictor:

    # Detection
    det_predictor = detection_predictor(det_arch, pretrained=pretrained, batch_size=det_bs)

    # Recognition
    reco_predictor = recognition_predictor(reco_arch, pretrained=pretrained, batch_size=reco_bs)

    return OCRPredictor(det_predictor, reco_predictor)


def ocr_predictor(
    det_arch: str = 'db_resnet50',
    reco_arch: str = 'crnn_vgg16_bn',
    pretrained: bool = False,
    **kwargs: Any
) -> OCRPredictor:
    """End-to-end OCR architecture using one model for localization, and another for text recognition.

    Example::
        >>> import numpy as np
        >>> from doctr.models import ocr_predictor
        >>> model = ocr_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([[input_page]])

    Args:
        arch: name of the architecture to use ('db_sar_vgg', 'db_sar_resnet', 'db_crnn_vgg', 'db_crnn_resnet')
        pretrained: If True, returns a model pre-trained on our OCR dataset

    Returns:
        OCR predictor
    """

    return _predictor(det_arch, reco_arch, pretrained, **kwargs)
