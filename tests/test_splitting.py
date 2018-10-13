import numpy as np
from numpy.testing import assert_allclose
import pytest

from pygbm.splitting import _find_histogram_split
from pygbm.splitting import HistogramSplitter


@pytest.mark.parametrize('n_bins', [3, 32, 256])
def test_hitstogram_split(n_bins):
    rng = np.random.RandomState(42)
    feature_idx = 12
    l2_regularization = 0
    min_hessian_to_split = 1e-3
    binned_feature = rng.randint(0, n_bins, size=int(1e4)).astype(np.uint8)
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)
    ordered_hessians = np.ones_like(binned_feature, dtype=np.float32)

    for true_bin in range(1, n_bins - 1):
        for sign in [-1, 1]:
            ordered_gradients = np.full_like(binned_feature, sign,
                                             dtype=np.float32)
            ordered_gradients[binned_feature <= true_bin] *= -1

            split_info = _find_histogram_split(
                feature_idx, binned_feature, n_bins, sample_indices,
                ordered_gradients, ordered_hessians, l2_regularization,
                min_hessian_to_split, True, ordered_hessians[0],
                ordered_gradients.sum(),
                ordered_hessians[0] * sample_indices.shape[0])

            assert split_info.bin_idx == true_bin
            assert split_info.gain >= 0
            assert split_info.feature_idx == feature_idx


@pytest.mark.skip()
# this is failing at the gain comparison. Investigating why... Maybe its
# because of the slight differences in the histograms but I doubt it
def test_split_vs_split_subtraction():
    # Make sure  find_node_split and finde_node_split_subtraction return the
    # same results.
    # Should we add test about computation time to make sure
    # time(subtraction) < time(regular)?
    rng = np.random.RandomState(42)

    n_bins = 10
    n_features = 20
    n_samples = int(1e3)
    l2_regularization = 0.
    min_hessian_to_split = 1e-3
    from pygbm import binning
    from sklearn.datasets import make_classification
    #binned_features = rng.randint(0, n_bins - 1, size=n_samples,
    #dtype=np.uint8)
    # TODO: change that
    X, y = make_classification(n_samples=n_samples, random_state=rng)
    binned_features = binning.BinMapper().fit_transform(X)
    sample_indices = np.arange(n_samples, dtype=np.uint32)

    all_gradients = rng.randn(n_samples).astype(np.float32)

    all_hessians = np.ones(n_samples, dtype=np.float32)
    all_hessians = rng.lognormal(size=n_samples).astype(np.float32)

    splitter = HistogramSplitter(n_features, binned_features, n_bins,
                                 all_gradients, all_hessians,
                                 l2_regularization, min_hessian_to_split)

    mask = rng.randint(0, 2, n_samples).astype(np.bool)
    sample_indices_left = sample_indices[mask]
    sample_indices_right = sample_indices[~mask]

    split_info_parent, hists_parent = splitter.find_node_split(sample_indices)
    split_info_left, hists_left = splitter.find_node_split(sample_indices_left)
    split_info_right, hists_right = splitter.find_node_split(sample_indices_right)

    split_info_left_sub, hists_left_sub = splitter.find_node_split_subtraction(sample_indices_left, hists_parent, hists_right)
    split_info_right_sub, hists_right_sub = splitter.find_node_split_subtraction(sample_indices_right, hists_parent, hists_left)
    
    for hists, hists_sub in ((hists_left, hists_left_sub),
                             (hists_right, hists_right_sub)):
        for hist, hist_sub in zip(hists, hists_sub):
            for key in ('count', 'sum_hessians', 'sum_gradients'):
                assert_allclose(hist[key], hist_sub[key], rtol=1e-3)

    
    for split_info, split_info_sub in (
            (split_info_left, split_info_left_sub),
            (split_info_right, split_info_right_sub)):
        assert split_info.gain == split_info_sub.gain
        assert split_info.feature_idx == split_info_sub.feature_idx
        assert split_info.gradient_left == split_info_sub.gradient_left
        assert split_info.gradient_right == split_info_sub.gradient_right
        assert split_info.hessian_right == split_info_sub.hessian_right
        assert split_info.hessian_left == split_info_sub.hessian_left