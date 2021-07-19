import pytest
import numpy as np
import tensorflow as tf

from doctr.models import detection, PreProcessor
from doctr.documents import DocumentFile


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet50", (1024, 1024, 3), (1024, 1024, 1), True],
        ["linknet16", (1024, 1024, 3), (1024, 1024, 1), False],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = detection.__dict__[arch_name](pretrained=True)
    assert isinstance(model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    target = [
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .8]], dtype=np.float32), flags=[True, False]),
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .9]], dtype=np.float32), flags=[True, False])
    ]
    # test training model
    out = model(input_tensor, target, return_model_output=True, return_boxes=True, training=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    # Check proba map
    seg_map = out['out_map'].numpy()
    assert seg_map.shape == (batch_size, *output_size)
    if out_prob:
        assert np.all(np.logical_and(seg_map >= 0, seg_map <= 1))
    # Check boxes
    for boxes in out['preds'][0]:
        assert boxes.shape[1] == 5
        assert np.all(boxes[:, :2] < boxes[:, 2:4])
        assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)
    # Check loss
    assert isinstance(out['loss'], tf.Tensor)
    # Target checks
    target = [
        dict(boxes=np.array([[0, 0, 1, 1]], dtype=np.uint8), flags=[True, False]),
        dict(boxes=np.array([[0, 0, 1, 1]], dtype=np.uint8), flags=[True, False])
    ]
    with pytest.raises(AssertionError):
        out = model(input_tensor, target, training=True)

    target = [
        dict(boxes=np.array([[0, 0, 1.5, 1.5]], dtype=np.float32), flags=[True, False]),
        dict(boxes=np.array([[-.2, -.3, 1, 1]], dtype=np.float32), flags=[True, False])
    ]
    with pytest.raises(ValueError):
        out = model(input_tensor, target, training=True)


@pytest.fixture(scope="session")
def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = detection.DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(input_shape=(512, 512, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(pages)
    out, _ = zip(*out)
    # The input PDF has 8 pages
    assert len(out) == 8

    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    return predictor


@pytest.fixture(scope="session")
def test_rotated_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = detection.DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(rotated_bbox=True, input_shape=(512, 512, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(pages)

    # The input PDF has 8 pages
    assert len(out) == 8

    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    return predictor


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet50",
        "linknet16",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, detection.DetectionPredictor)
    input_tensor = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert all(isinstance(out_img, tuple) for out_img in out)
    all_boxes, _ = zip(*out)
    assert all(isinstance(boxes, np.ndarray) and boxes.shape[1] == 5 for boxes in all_boxes)


def test_detection_zoo_error():
    with pytest.raises(ValueError):
        _ = detection.zoo.detection_predictor("my_fancy_model", pretrained=False)


def test_linknet_focal_loss():
    batch_size = 2
    input_shape = (1024, 1024, 3)
    model = detection.linknet16(pretrained=True)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    target = [
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .8]], dtype=np.float32), flags=[True, False]),
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .9]], dtype=np.float32), flags=[True, False])
    ]
    # test focal loss
    out = model(input_tensor, target, return_model_output=True, return_boxes=True, training=True, focal_loss=True)
    assert isinstance(out['loss'], tf.Tensor)
