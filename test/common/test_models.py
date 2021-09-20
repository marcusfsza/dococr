import pytest
import numpy as np
from io import BytesIO
import requests
import cv2

from doctr.io import reader, DocumentFile
from doctr.models._utils import extract_crops, extract_rcrops, get_bitmap_angle, estimate_orientation


def test_extract_crops(mock_pdf):  # noqa: F811
    doc_img = DocumentFile.from_pdf(mock_pdf).as_images()[0]
    num_crops = 2
    rel_boxes = np.array([[idx / num_crops, idx / num_crops, (idx + 1) / num_crops, (idx + 1) / num_crops]
                          for idx in range(num_crops)], dtype=np.float32)
    abs_boxes = np.array([[int(idx * doc_img.shape[1] / num_crops),
                           int(idx * doc_img.shape[0]) / num_crops,
                           int((idx + 1) * doc_img.shape[1] / num_crops),
                           int((idx + 1) * doc_img.shape[0] / num_crops)]
                          for idx in range(num_crops)], dtype=np.float32)

    with pytest.raises(AssertionError):
        extract_crops(doc_img, np.zeros((1, 5)))

    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = extract_crops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # Identity
    assert np.all(doc_img == extract_crops(doc_img, np.array([[0, 0, 1, 1]], dtype=np.float32))[0])

    # No box
    assert extract_crops(doc_img, np.zeros((0, 4))) == []


def test_extract_rcrops(mock_pdf):  # noqa: F811
    doc_img = DocumentFile.from_pdf(mock_pdf).as_images()[0]
    num_crops = 2
    rel_boxes = np.array([[idx / num_crops + .1, idx / num_crops + .1, .1, .1, 0]
                          for idx in range(num_crops)], dtype=np.float32)
    abs_boxes = np.array([[int((idx / num_crops + .1) * doc_img.shape[1]),
                           int((idx / num_crops + .1) * doc_img.shape[0]),
                           int(.1 * doc_img.shape[1]),
                           int(.1 * doc_img.shape[0]), 0]
                          for idx in range(num_crops)], dtype=np.int)

    with pytest.raises(AssertionError):
        extract_rcrops(doc_img, np.zeros((1, 8)))
    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = extract_rcrops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # No box
    assert extract_rcrops(doc_img, np.zeros((0, 5))) == []


@pytest.fixture(scope="function")
def mock_image(tmpdir_factory):
    url = 'https://github.com/mindee/doctr/releases/download/v0.2.1/bitmap30.png'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_bitmap.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())
    image = reader.read_img_as_numpy(tmp_path)
    return image


@pytest.fixture(scope="function")
def mock_bitmap(mock_image):
    bitmap = np.squeeze(cv2.cvtColor(mock_image, cv2.COLOR_BGR2GRAY) / 255.)
    return bitmap


def test_get_bitmap_angle(mock_bitmap):
    angle = get_bitmap_angle(mock_bitmap)
    assert abs(angle - 30.) < 1.


def test_estimate_orientation(mock_image):
    angle = estimate_orientation(mock_image)
    assert abs(angle - 30.) < 1.
