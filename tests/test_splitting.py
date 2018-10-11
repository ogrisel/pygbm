import numpy as np
import pytest

from pygbm.splitting import _find_histogram_split


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
                min_hessian_to_split, True, ordered_hessians[0])

            assert split_info.bin_idx == true_bin
            assert split_info.gain >= 0
            assert split_info.feature_idx == feature_idx
