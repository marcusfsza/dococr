# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
from typing import Any, List, Optional

import torch
from onnxruntime.quantization import quantize_dynamic
from torch import nn

from doctr.utils.data import download_from_url

__all__ = ['load_pretrained_params', 'conv_sequence_pt', 'export_classification_model_to_onnx']


def load_pretrained_params(
    model: nn.Module,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    Example::
        >>> from doctr.models import load_pretrained_params
        >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the keras model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        overwrite: should the zip extraction be enforced if the archive has already been extracted
    """

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir='models', **kwargs)

        # Read state_dict
        state_dict = torch.load(archive_path, map_location='cpu')

        # Load weights
        model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    relu: bool = False,
    bn: bool = False,
    **kwargs: Any,
) -> List[nn.Module]:
    """Builds a convolutional-based layer sequence

    Example::
        >>> from doctr.models import conv_sequence
        >>> from torch.nn import Sequential
        >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs['bias'] = kwargs.get('bias', not(bn))
    # Add activation directly to the conv if there is no BN
    conv_seq: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, **kwargs)
    ]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if relu:
        conv_seq.append(nn.ReLU(inplace=True))

    return conv_seq


def export_classification_model_to_onnx(model: nn.Module, exp_name: str, dummy_input: torch.Tensor) -> List[str]:
    """Export classification model to ONNX format.

    >>> from doctr.models.utils import export_classification_model_to_onnx
    >>> export_classification_model_to_onnx(model, "my_model", dummy_input=torch.randn(1, 3, 32, 32))

    Args:
        model: the pytorch model to be exported
        exp_name: the name of the exported model
        dummy_input: the dummy input to the model

    Returns:
        list of exported model files
    """

    torch.onnx.export(
        model,
        dummy_input,
        f"{exp_name}.onnx",
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        export_params=True, opset_version=13, verbose=False
    )
    logging.info(f"Model exported to {exp_name}.onnx")
    quantize_dynamic(f"{exp_name}.onnx", f'{exp_name}.quant.onnx')
    logging.info(f"Quantized model saved to {exp_name}.quant.onnx")
    return [f"{exp_name}.onnx", f"{exp_name}.quant.onnx"]
