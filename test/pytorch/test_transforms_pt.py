import pytest

import math
import torch
import numpy as np
from doctr.transforms import Resize, ColorInversion, RandomRotate
from doctr.transforms.functional import rotate, crop_detection


def test_resize():
    output_size = (32, 32)
    transfo = Resize(output_size)
    input_t = torch.ones((3, 64, 64), dtype=torch.float32)
    out = transfo(input_t)

    assert torch.all(out == 1)
    assert out.shape[-2:] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, interpolation='bilinear')"

    transfo = Resize(output_size, preserve_aspect_ratio=True)
    input_t = torch.ones((3, 32, 64), dtype=torch.float32)
    out = transfo(input_t)

    assert out.shape[-2:] == output_size
    assert not torch.all(out == 1)
    # Asymetric padding
    assert torch.all(out[:, -1] == 0) and torch.all(out[:, 0] == 1)

    # Symetric padding
    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (f"Resize(output_size={output_size}, interpolation='bilinear', "
                             f"preserve_aspect_ratio=True, symmetric_pad=True)")
    out = transfo(input_t)
    assert out.shape[-2:] == output_size
    # symetric padding
    assert torch.all(out[:, -1] == 0) and torch.all(out[:, 0] == 0)

    # Inverse aspect ratio
    input_t = torch.ones((3, 64, 32), dtype=torch.float32)
    out = transfo(input_t)

    assert not torch.all(out == 1)
    assert out.shape[-2:] == output_size

    # Same aspect ratio
    output_size = (32, 128)
    transfo = Resize(output_size, preserve_aspect_ratio=True)
    out = transfo(torch.ones((3, 16, 64), dtype=torch.float32))
    assert out.shape[-2:] == output_size


@pytest.mark.parametrize(
    "rgb_min",
    [
        0.2,
        0.4,
        0.6,
    ],
)
def test_invert_colorize(rgb_min):

    transfo = ColorInversion(min_val=rgb_min)
    input_t = torch.ones((8, 3, 32, 32), dtype=torch.float32)
    out = transfo(input_t)
    assert torch.all(out <= 1 - rgb_min + 1e-4)
    assert torch.all(out >= 0)

    input_t = torch.full((8, 3, 32, 32), 255, dtype=torch.uint8)
    out = transfo(input_t)
    assert torch.all(out <= int(math.ceil(255 * (1 - rgb_min + 1e-4))))
    assert torch.all(out >= 0)


def test_rotate():
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, r_boxes = rotate(input_t, boxes, angle=12.)
    assert r_img.shape == (3, 50, 50)
    assert r_img[0, 0, 0] == 0.
    assert r_boxes.all() == np.array([[25., 25., 20., 10., 12.]]).all()
    rel_boxes = np.array([
        [.3, .4, .7, .6]
    ])
    r_img, r_boxes = rotate(input_t, rel_boxes, angle=12.)
    assert r_boxes.all() == np.array([[.5, .5, .4, .2, 12.]]).all()


def test_random_rotate():
    rotator = RandomRotate(max_angle=10.)
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, target = rotator(input_t, dict(boxes=boxes))
    assert r_img.shape == input_t.shape
    assert abs(target["boxes"][-1, -1]) <= 10.


def test_crop_detection():
    img = torch.ones((3, 50, 50), dtype=torch.float32)
    abs_boxes = np.array([
        [15, 20, 35, 30],
        [5, 10, 10, 20],
    ])
    crop_box = (12, 23, 50, 50)
    c_img, c_boxes = crop_detection(img, abs_boxes, crop_box)
    assert c_img.shape == (3, 27, 38)
    assert c_boxes.shape == (1, 4)
    rel_boxes = np.array([
        [.3, .4, .7, .6],
        [.1, .2, .2, .4],
    ])
    c_img, c_boxes = crop_detection(img, rel_boxes, crop_box)
    assert c_img.shape == (3, 27, 38)
    assert c_boxes.shape == (1, 4)
