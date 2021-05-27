# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
import tensorflow as tf
from typing import List, Union

__all__ = ['extract_crops', 'extract_rcrops']


def extract_crops(img: Union[np.ndarray, tf.Tensor], boxes: np.ndarray) -> List[Union[np.ndarray, tf.Tensor]]:
    """Created cropped images from list of bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)

    Returns:
        list of cropped images
    """
    if isinstance(img, tf.Tensor):
        img = img.numpy().astype(np.uint8)

    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    return [img[box[1]: box[3], box[0]: box[2]] for box in _boxes]


def extract_rcrops(img: Union[np.ndarray, tf.Tensor], boxes: np.ndarray) -> List[Union[np.ndarray, tf.Tensor]]:
    """Created cropped images from list of rotated bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 5) where N is the number of boxes, and the relative
            coordinates (x, y, w, h, alpha)

    Returns:
        list of cropped images
    """
    if isinstance(img, tf.Tensor):
        img = img.numpy().astype(np.uint8)

    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 5:
        raise AssertionError("boxes are expected to be relative and in order (x, y, w, h, alpha)")

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]
        _boxes = _boxes.round().astype(int)

    crops = []
    for box in _boxes:
        x, y, w, h, alpha = box.astype(np.float32)
        src_pts = cv2.boxPoints(((x, y), (w, h), alpha))
        # Preserve size
        dst_pts = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
        # The transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Warp the rotated rectangle
        crop = cv2.warpPerspective(img, M, (int(w), int(h)))
        crops.append(crop)

    return crops
