import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pygbm.histogram import _build_histogram_naive, _build_histogram_unrolled


@pytest.mark.parametrize(
    'build_func', [_build_histogram_naive, _build_histogram_unrolled])
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
