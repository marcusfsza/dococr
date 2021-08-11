import pytest

from doctr import datasets


def test_visiondataset():
    url = 'https://data.deepai.org/mnist.zip'
    with pytest.raises(ValueError):
        datasets.datasets.VisionDataset(url, download=False)

    dataset = datasets.datasets.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == 'VisionDataset()'


def test_abstractdataset():
    with pytest.raises(ValueError):
        datasets.datasets.AbstractDataset('my/fantasy/folder')


def test_character_generator():
    img, char = datasets.generate_character('a')
    assert img.shape == (32, 32, 3)
    assert char == 'a'
    img, char = datasets.generate_character('!', angle=10)
