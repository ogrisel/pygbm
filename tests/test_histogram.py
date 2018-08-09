import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pygbm.histogram import _build_ghc_histogram_naive
from pygbm.histogram import _build_ghc_histogram_unrolled
from pygbm.histogram import _build_gc_histogram_unrolled
from pygbm.histogram import _build_gc_root_histogram_unrolled
from pygbm.histogram import _build_ghc_root_histogram_unrolled


@pytest.mark.parametrize(
    'build_func', [_build_ghc_histogram_naive, _build_ghc_histogram_unrolled])
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


def test_compare_histograms_optimized_vs_general():
    # Check that optimized histograms are consistent with equivalent
    # non-optimized more general versions
    rng = np.random.RandomState(42)
    n_samples = 10
    n_bins = 5
    sample_indices = np.arange(n_samples).astype(np.uint32)
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    ordered_gradients = rng.randn(n_samples).astype(np.float32)
    ordered_hessians = np.ones(n_samples, dtype=np.float32)

    hist_gc_root = _build_gc_root_histogram_unrolled(n_bins, binned_feature,
                                                     ordered_gradients)
    hist_ghc_root = _build_ghc_root_histogram_unrolled(n_bins, binned_feature,
                                                       ordered_gradients,
                                                       ordered_hessians)
    hist_gc = _build_gc_histogram_unrolled(n_bins, sample_indices,
                                           binned_feature,
                                           ordered_gradients)
    hist_ghc = _build_ghc_histogram_unrolled(n_bins, sample_indices,
                                             binned_feature,
                                             ordered_gradients,
                                             ordered_hessians)

    assert_array_equal(hist_gc_root['count'], hist_gc['count'])
    assert_allclose(hist_gc_root['sum_gradients'], hist_gc['sum_gradients'])

    assert_array_equal(hist_gc_root['count'], hist_ghc_root['count'])
    assert_allclose(hist_gc_root['sum_gradients'],
                    hist_ghc_root['sum_gradients'])

    assert_array_equal(hist_gc_root['count'], hist_ghc['count'])
    assert_allclose(hist_gc_root['sum_gradients'], hist_ghc['sum_gradients'])

    assert_allclose(hist_gc_root['sum_hessians'], np.zeros(n_bins))  # unused
    assert_allclose(hist_ghc_root['sum_hessians'], hist_ghc_root['count'])
    assert_allclose(hist_gc['sum_hessians'], np.zeros(n_bins))  # unused
    assert_allclose(hist_ghc['sum_hessians'], hist_ghc['count'])
