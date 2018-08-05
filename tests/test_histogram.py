import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pygbm.histogram import build_histogram, build_histogram_unrolled
from pygbm.histogram import find_split


@pytest.mark.parametrize(
    'build_func', [build_histogram, build_histogram_unrolled])
def test_build_histogram(build_func):
    binned_feature = np.array([0, 2, 0, 1, 2, 0, 2, 1], dtype=np.uint8)

    # Small sample_indices (below unrolling threshold)
    ordered_gradients = np.array([0, 1, 3], dtype=np.float32)
    ordered_hessians = np.array([1, 1, 2], dtype=np.float32)

    sample_indices = np.array([0, 2, 3], dtype=np.uint32)
    hist = build_func(3, sample_indices, binned_feature,
                      ordered_gradients, ordered_hessians)
    assert_array_equal(hist['count'], [2, 1, 0])
    assert_allclose(hist['sum_gradients'], [1, 3, 0])
    assert_allclose(hist['sum_hessians'], [2, 2, 0])

    # Larger sample_indices (above unrolling threshold)
    sample_indices = np.array([0, 2, 3, 6, 7], dtype=np.uint32)
    ordered_gradients = np.array([0, 1, 3, 0, 1], dtype=np.float32)
    ordered_hessians = np.array([1, 1, 2, 1, 0], dtype=np.float32)

    hist = build_func(3, sample_indices, binned_feature,
                      ordered_gradients, ordered_hessians)
    assert_array_equal(hist['count'], [2, 2, 1])
    assert_allclose(hist['sum_gradients'], [1, 4, 0])
    assert_allclose(hist['sum_hessians'], [2, 2, 1])


def test_find_split():
    binned_feature = np.array([0, 2, 0, 1, 2, 0, 2, 1], dtype=np.uint8)
    ordered_gradients = np.array([-2, 1, -2, -1, 1, -2, 1, -1],
                                 dtype=np.float32)
    ordered_hessians = np.ones_like(binned_feature, dtype=np.float32)
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)
    histogram = build_histogram(3, sample_indices, binned_feature,
                                ordered_gradients, ordered_hessians)
    l2_regularization = 0.0001
    bin_idx, gain = find_split(histogram, ordered_gradients.sum(),
                               ordered_gradients.sum(), l2_regularization)
    assert bin_idx == 1
    assert gain > 0.
