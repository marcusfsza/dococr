# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np

from .datasets import AbstractDataset

__all__ = ["IC13"]


class IC13(AbstractDataset):
    """IC13 dataset from
    `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.
    Example::
        >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
        >>> # structure: - img_folder: all images (part1 and part2) | - label_folder: all labels (part1 and part2)
        >>> from doctr.datasets import IC13
        >>> train_set = IC13(img_folder="/path/to/images", label_folder="/path/to/labels", train=True)
        >>> img, target = train_set[0]
    Args:
        img_folder: folder with all the images of the dataset
        label_folder: folder with all annotation files for the images
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
    ) -> None:
        super().__init__(img_folder)
        self.sample_transforms = sample_transforms

        # File existence check
        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_folder if not os.path.exists(label_folder) else img_folder}")

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float32

        img_names = os.listdir(img_folder)
        split = int(len(img_names) * 0.9)
        img_names = img_names[:split] if train else img_names[split:]

        for img_name in img_names:

            img_path = Path(img_folder, img_name)
            label_path = Path(label_folder, Path(img_name).stem + ".txt")

            with open(label_path, "r") as f:
                _lines = [[x for x in line.split(',')] for line in f.readlines()]
                labels = [line[4].strip() for line in _lines]
                coords = np.array([np.array(list(map(int, line[:8]))).reshape((4, 2))
                                  for line in _lines], dtype=np_dtype)
                if rotated_bbox:
                    # x_center, y_center, w, h, alpha = 0
                    mins = coords.min(axis=1)
                    maxs = coords.max(axis=1)
                    box_targets = np.concatenate(
                        ((mins + maxs) / 2, maxs - mins, np.zeros((coords.shape[0], 1))), axis=1)
                else:
                    # xmin, ymin, xmax, ymax
                    box_targets = np.concatenate((coords.min(axis=1), coords.max(axis=1)), axis=1)

                self.data.append((img_path, dict(boxes=box_targets, languages=vocab_lang, labels=labels)))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        img, target = self._read_sample(index)
        h, w = self._get_img_shape(img)
        if self.sample_transforms is not None:
            img = self.sample_transforms(img)

        # Boxes
        boxes = target['boxes'].copy()
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
        boxes = boxes.clip(0, 1)
        target['boxes'] = boxes

        return img, target