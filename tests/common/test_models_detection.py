import numpy as np

from doctr.models import detection


def test_dbpostprocessor():
    postprocessor = detection.DBPostProcessor(rotated_bbox=False)
    r_postprocessor = detection.DBPostProcessor(rotated_bbox=True)
    mock_batch = np.random.rand(2, 512, 512).astype(np.float32)
    out, _ = postprocessor(mock_batch)
    r_out, _ = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    assert all(sample.shape[1] == 6 for sample in r_out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:, :4] >= 0, sample[:, :4] <= 1)) for sample in out)
    assert all(np.all(np.logical_and(sample[:, :4] >= 0, sample[:, :4] <= 1)) for sample in r_out)
    # Repr
    assert repr(postprocessor) == 'DBPostProcessor(box_thresh=0.1)'
    # Edge case when the expanded points of the polygon has two lists
    issue_points = np.array([
        [869, 561],
        [923, 581],
        [925, 595],
        [915, 583],
        [889, 583],
        [905, 593],
        [882, 601],
        [901, 595],
        [904, 604],
        [876, 608],
        [915, 614],
        [911, 605],
        [925, 601],
        [930, 616],
        [911, 617],
        [900, 636],
        [931, 637],
        [904, 649],
        [932, 649],
        [932, 628],
        [918, 627],
        [934, 624],
        [935, 573],
        [909, 569],
        [934, 562]], dtype=np.int32)
    out = postprocessor.polygon_to_box(issue_points)
    r_out = r_postprocessor.polygon_to_box(issue_points)
    assert isinstance(out, tuple) and len(out) == 4
    assert isinstance(r_out, tuple) and len(r_out) == 5


def test_linknet_postprocessor():
    postprocessor = detection.LinkNetPostProcessor()
    r_postprocessor = detection.LinkNetPostProcessor(rotated_bbox=True)
    mock_batch = np.random.rand(2, 512, 512).astype(np.float32)
    out, _ = postprocessor(mock_batch)
    r_out, _ = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    assert all(sample.shape[1] == 6 for sample in r_out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)
