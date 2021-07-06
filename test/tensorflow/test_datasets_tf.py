import pytest
import numpy as np
import tensorflow as tf

from doctr import datasets
from doctr.transforms import Resize
from doctr.datasets import DataLoader


@pytest.mark.parametrize(
    "dataset_name, train, input_size, size, rotate",
    [
        ['FUNSD', True, [512, 512], 149, False],
        ['FUNSD', False, [512, 512], 50, True],
        ['SROIE', True, [512, 512], 626, False],
        ['SROIE', False, [512, 512], 360, False],
        ['CORD', True, [512, 512], 800, True],
        ['CORD', False, [512, 512], 100, False],
    ],
)
def test_dataset(dataset_name, train, input_size, size, rotate):

    ds = datasets.__dict__[dataset_name](
        train=train, download=True, sample_transforms=Resize(input_size), rotated_bbox=rotate
    )

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}(train={train})"
    img, target = ds[0]
    assert isinstance(img, tf.Tensor) and img.shape == (*input_size, 3)
    assert isinstance(target, dict)

    loader = datasets.DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)


def test_detection_dataset(mock_image_folder, mock_detection_label):

    input_size = (1024, 1024)

    ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_folder=mock_detection_label,
        sample_transforms=Resize(input_size),
    )

    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 4
    # Flags
    assert isinstance(target['flags'], np.ndarray) and target['flags'].dtype == np.bool
    # Cardinality consistency
    assert target['boxes'].shape[0] == target['flags'].shape[0]

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)

    # Rotated DS
    rotated_ds = datasets.DetectionDataset(
        img_folder=mock_image_folder,
        label_folder=mock_detection_label,
        sample_transforms=Resize(input_size),
        rotated_bbox=True
    )
    _, r_target = rotated_ds[0]
    assert r_target['boxes'].shape[1] == 5


def test_recognition_dataset(mock_image_folder, mock_recognition_label):
    input_size = (32, 128)
    ds = datasets.RecognitionDataset(
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label,
        sample_transforms=Resize(input_size),
    )
    assert ds.__len__() == 5
    image, label = ds[0]
    assert isinstance(image, tf.Tensor)
    assert image.shape[:2] == input_size
    assert isinstance(label, str)

    loader = DataLoader(ds, batch_size=2)
    images, labels = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(labels, list) and all(isinstance(elt, str) for elt in labels)


def test_ocrdataset(mock_ocrdataset):

    input_size = (512, 512)

    ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
    )
    rotated_ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
        rotated_bbox=True
    )
    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'][:, :4] >= 0, target['boxes'][:, :4] <= 1))
    assert target['boxes'].shape[1] == 4
    _, r_target = rotated_ds[0]
    assert r_target['boxes'].shape[1] == 5
    # Flags
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    # Cardinality consistency
    assert target['boxes'].shape[0] == len(target['labels'])

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
