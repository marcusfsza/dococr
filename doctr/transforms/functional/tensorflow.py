# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
from copy import deepcopy
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from doctr.utils.geometry import compute_expanded_shape, rotate_abs_boxes
from .base import crop_boxes

__all__ = ["invert_colors", "rotate", "crop_detection"]


def invert_colors(img: tf.Tensor, min_val: float = 0.6) -> tf.Tensor:
    out = tf.image.rgb_to_grayscale(img)  # Convert to gray
    # Random RGB shift
    shift_shape = [img.shape[0], 1, 1, 3] if img.ndim == 4 else [1, 1, 3]
    rgb_shift = tf.random.uniform(shape=shift_shape, minval=min_val, maxval=1)
    # Inverse the color
    if out.dtype == tf.uint8:
        out = tf.cast(tf.cast(out, dtype=rgb_shift.dtype) * rgb_shift, dtype=tf.uint8)
    else:
        out *= tf.cast(rgb_shift, dtype=out.dtype)
    # Inverse the color
    out = 255 - out if out.dtype == tf.uint8 else 1 - out
    return out


def rotate(
    img: tf.Tensor,
    boxes: np.ndarray,
    angle: float,
    expand: bool = False,
) -> Tuple[tf.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        boxes: array of boxes to rotate as well
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
        A tuple of rotated img (tensor), rotated boxes (np array)
    """
    # Compute the expanded padding
    if expand:
        exp_shape = compute_expanded_shape(img.shape[:-1], angle)
        h_pad, w_pad = int(math.ceil(exp_shape[0] - img.shape[0])), int(math.ceil(exp_shape[1] - img.shape[1]))
        if min(h_pad, w_pad) < 0:
            h_pad, w_pad = int(math.ceil(exp_shape[1] - img.shape[0])), int(math.ceil(exp_shape[0] - img.shape[1]))
        exp_img = tf.pad(img, tf.constant([[h_pad // 2, h_pad - h_pad // 2], [w_pad // 2, w_pad - w_pad // 2], [0, 0]]))
    else:
        exp_img = img
    # Rotate the padded image
    rotated_img = tfa.image.rotate(exp_img, angle * math.pi / 180)  # Interpolation NEAREST by default

    # Get absolute coords
    _boxes = deepcopy(boxes)
    if np.max(_boxes) <= 1:
        _boxes[:, [0, 2]] = _boxes[:, [0, 2]] * img.shape[1]
        _boxes[:, [1, 3]] = _boxes[:, [1, 3]] * img.shape[0]

    # Rotate the boxes: xmin, ymin, xmax, ymax or polygons --> (4, 2) polygon
    r_boxes = rotate_abs_boxes(_boxes, angle, img.shape[:-1], expand).astype(np.float32)

    # Always return relative boxes to avoid label confusions when resizing is performed aferwards
    r_boxes[..., 0] = r_boxes[..., 0] / rotated_img.shape[1]
    r_boxes[..., 1] = r_boxes[..., 1] / rotated_img.shape[0]

    return rotated_img, r_boxes


def crop_detection(
    img: tf.Tensor,
    boxes: np.ndarray,
    crop_box: Tuple[float, float, float, float]
) -> Tuple[tf.Tensor, np.ndarray]:
    """Crop and image and associated bboxes

    Args:
        img: image to crop
        boxes: array of boxes to clip, absolute (int) or relative (float)
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image. Relative coords.

    Returns:
        A tuple of cropped image, cropped boxes, where the image is not resized.
    """
    if any(val < 0 or val > 1 for val in crop_box):
        raise AssertionError("coordinates of arg `crop_box` should be relative")
    h, w = img.shape[:2]
    xmin, ymin = int(round(crop_box[0] * (w - 1))), int(round(crop_box[1] * (h - 1)))
    xmax, ymax = int(round(crop_box[2] * (w - 1))), int(round(crop_box[3] * (h - 1)))
    cropped_img = tf.image.crop_to_bounding_box(
        img, ymin, xmin, ymax - ymin, xmax - xmin
    )
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes
