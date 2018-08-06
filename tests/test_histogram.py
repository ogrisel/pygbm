import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pygbm.histogram import _build_histogram_naive, _build_histogram_unrolled
from pygbm.histogram import find_split


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


@pytest.mark.parametrize('n_bins', [3, 32, 256])
def test_find_split(n_bins):
    rng = np.random.RandomState(42)
    feature_idx = 12
    l2_regularization = 1e-3
    binned_feature = rng.randint(0, n_bins, size=int(1e4)).astype(np.uint8)
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)
    ordered_hessians = np.ones_like(binned_feature, dtype=np.float32)

    for true_bin in range(1, n_bins - 1):
        for sign in [-1, 1]:
            ordered_gradients = np.full_like(binned_feature, sign,
                                             dtype=np.float32)
            ordered_gradients[binned_feature <= true_bin] *= -1

            histogram = _build_histogram_unrolled(
                n_bins, sample_indices, binned_feature,
                ordered_gradients, ordered_hessians)

            split_info = find_split(histogram, feature_idx,
                                    ordered_gradients.sum(),
                                    ordered_hessians.sum(),
                                    l2_regularization)

            assert split_info.bin_idx == true_bin
            assert split_info.gain >= 0
            assert split_info.feature_idx == feature_idx
