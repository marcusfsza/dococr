# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import List, Any, Tuple
import tensorflow as tf

from .base import _AbstractDataset, _VisionDataset


__all__ = ['AbstractDataset', 'VisionDataset']


class AbstractDataset(_AbstractDataset):

    @staticmethod
    def _get_img_shape(img: Any) -> Tuple[int, int]:
        return img.shape[:2]

    def _read_sample(self, index: int) -> Tuple[tf.Tensor, Any]:
        img_name, target = self.data[index]
        # Read image
        img = tf.io.read_file(os.path.join(self.root, img_name))
        img = tf.image.decode_jpeg(img, channels=3)
        if self.fp16:
            img = tf.image.convert_image_dtype(img, dtype=tf.float16)
        else:
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        img = tf.clip_by_value(img, 0, 1)

        return img, target

    @staticmethod
    def collate_fn(samples: List[Tuple[tf.Tensor, Any]]) -> Tuple[tf.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, list(targets)


class VisionDataset(AbstractDataset, _VisionDataset):
    pass
