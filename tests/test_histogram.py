import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pygbm.histogram import Histogram


def test_build_histogram():
    binned_features = np.array(
        [0, 2, 0, 1, 2, 0, 2, 1],
        dtype=np.uint8)
    sample_indices = np.array([0, 2, 3], dtype=np.uint32)
    ordered_gradients = np.array([0, 1, 3], dtype=np.float32)
    ordered_hessians = np.array([1, 1, 2], dtype=np.float32)
    hist = Histogram(3).build(sample_indices, binned_features,
                              ordered_gradients, ordered_hessians)
    assert_array_equal(hist.count, np.array([2, 1, 0]))
    assert_allclose(hist.sum_gradients, np.array([1, 3, 0], dtype=np.float32))
    assert_allclose(hist.sum_hessians, np.array([2, 2, 0], dtype=np.float32))
