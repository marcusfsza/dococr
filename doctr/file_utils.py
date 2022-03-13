# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

import importlib.util
import logging
import os
import sys

from .version import __version__

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


__all__ = ['is_tf_available', 'is_torch_available']

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

# config root logger
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

_tf_available = False
_torch_available = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES:
    logger.info("\033[1mBoth PyTorch and TensorFlow are available. Will use TensorFlow by default.\n"
                "To use PyTorch, set `USE_TORCH=1` and `USE_TF=0` in your environment variables.\033[0m")
    USE_TF, USE_TORCH = "1", "0"


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"\n\033[1mDocTR version {__version__}:\tPyTorch {_torch_version} is enabled.\033[0m\n")
            _torch_available = True
        except importlib_metadata.PackageNotFoundError:
            raise ModuleNotFoundError("PyTorch is installed but not available. Please check your installation.")

elif USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if int(_tf_version.split('.')[0]) < 2:  # type: ignore[union-attr]
            logger.info(f"TensorFlow found but with version {_tf_version}. DocTR requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"\n\033[1mDocTR version {__version__}:\tTensorFlow {_tf_version} is enabled.\033[0m\n")
            _tf_available = True


if not _torch_available and not _tf_available:
    raise ModuleNotFoundError("DocTR requires either TensorFlow or PyTorch to be installed. Please ensure one of them"
                              " is installed and that either USE_TF or USE_TORCH is enabled.")


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available
