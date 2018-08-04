import numpy as np
import pytest

from pygbm.binning import find_bins, map_to_bins


DATA = np.random.RandomState(42).normal(loc=[0, 10],
                                        scale=[1, 0.01],
                                        size=(int(1e6), 2)).astype(np.float32)


def test_find_bins():
    bins = find_bins(DATA, random_state=0)
    assert bins.shape == (2, 256)
    assert bins.dtype == DATA.dtype

    assert np.allclose(bins[0][[64, 128, 192]],
                       np.array([-0.7, 0.0, 0.7]), atol=1e-1)

    assert np.allclose(bins[1][[64, 128, 192]],
                       np.array([9.99, 10.00, 10.01]), atol=1e-2)


def test_find_bins_low_n_bins():
    bins = find_bins(DATA, n_bins=128, random_state=0)
    assert bins.shape == (2, 128)
    assert bins.dtype == DATA.dtype


def test_find_bins_invalid_n_bins():
    with pytest.raises(ValueError):
        find_bins(DATA, n_bins=1024)


@pytest.mark.parametrize('n_bins', [16, 128, 256])
def test_map_to_bins(n_bins):
    bins = find_bins(DATA, n_bins=n_bins, random_state=0)
    binned = map_to_bins(DATA, bins)
    assert binned.shape == DATA.shape
    assert binned.dtype == np.uint8
    assert binned.flags.f_contiguous

    min_indices = DATA.argmin(axis=0)
    max_indices = DATA.argmax(axis=0)

    for feature_idx, min_idx in enumerate(min_indices):
        assert binned[min_idx, feature_idx] == 0
    for feature_idx, max_idx in enumerate(max_indices):
        assert binned[max_idx, feature_idx] == n_bins - 1
