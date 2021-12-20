import pytest
import tensorflow as tf

from doctr.models import classification
from doctr.models.classification.predictor import CropOrientationPredictor


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn", (224, 224, 3), (7, 56, 512)],
        ["resnet31", (32, 128, 3), (4, 32, 512)],
        ["magc_resnet31", (32, 128, 3), (4, 32, 512)],
        ["mobilenet_v3_small", (512, 512, 3), (16, 16, 576)],
        ["mobilenet_v3_large", (512, 512, 3), (16, 16, 960)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = classification.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.numpy().shape == (batch_size, *output_size)


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_orientation", (128, 128, 3)],
    ],
)
def test_classification_models(arch_name, input_shape):
    batch_size = 8
    reco_model = classification.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    assert isinstance(reco_model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)

    out = reco_model(input_tensor)
    assert isinstance(out, tf.Tensor)
    assert out.shape.as_list() == [8, 4]


@pytest.mark.parametrize(
    "arch_name",
    [
        "mobilenet_v3_small_orientation",
    ],
)
def test_classification_zoo(arch_name):
    batch_size = 16
    # Model
    predictor = classification.zoo.crop_orientation_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, CropOrientationPredictor)
    input_tensor = tf.random.uniform(shape=[batch_size, 128, 128, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(pred, int) for pred in out)
