import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin


def find_bins(data, n_bins=256, subsample=int(2e5), random_state=None):
    """Extract feature-wise equally-spaced quantiles from numerical data

    Subsample the dataset if too large as the feature-wise quantiles
    should be stable.

    Parameters
    ----------
    data: array-like (n_samples, n_features)
        The numerical dataset to analyse.

    n_bins: int
        The number of bins to extract for each feature. As we code the binned
        values as 8-bit integers, n_bins should be no larger than 256.

    subsample: int
        Number of random subsamples to consider to compute the quantiles.

    random_state: int or numpy.random.RandomState or None
        Pseudo-random number generator to control the random sub-sampling.

    Return
    ------
    binning_thresholds: array (n_features, n_bins)
        For each feature, store the increasing numeric values that can
        be used to separate the bins.
    """
    if n_bins > 256:
        raise ValueError(f'n_bins should no larger than 256, got {n_bins}')
    rng = check_random_state(random_state)
    if data.shape[0] > subsample:
        subset = rng.choice(np.arange(data.shape[0]), subsample)
        data = data[subset]
    data = np.asfortranarray(data, dtype=np.float32)
    binning_thresholds = np.percentile(data, np.linspace(0, 100, num=n_bins),
                                       axis=0).T
    return np.ascontiguousarray(binning_thresholds, dtype=data.dtype)


def map_to_bins(data, binning_thresholds=None, out=None):
    """Bin numerical values to discrete integer-coded levels.

    # TODO: write doc for params and returned value.
    """
    # TODO: add support for categorical data encoded as integers
    # TODO: add support for sparse data (numerical or categorical)
    if out is not None:
        assert out.shape == data.shape
        assert out.dtype == np.uint8
        assert out.flags.f_contiguous
        binned = out
    else:
        binned = np.zeros_like(data, dtype=np.uint8, order='F')

    binning_thresholds = np.ascontiguousarray(binning_thresholds,
                                              dtype=np.float32)
    for feature_idx in range(data.shape[1]):
        _map_num_col_to_bins(data[:, feature_idx],
                             binning_thresholds[feature_idx],
                             binned[:, feature_idx])
    return binned


@njit(parallel=True)
def _map_num_col_to_bins(data, binning_thresholds, binned):
    """Binary search to the find the bin index for each value in data."""
    for i in prange(data.shape[0]):
        # TODO: add support for missing values (NaN or custom marker)
        left, right = 0, binning_thresholds.shape[0] - 1
        while left < right:
            middle = (right + left - 1) // 2
            if data[i] <= binning_thresholds[middle]:
                right = middle
            else:
                left = middle + 1
        binned[i] = left


class BinMapper(BaseEstimator, TransformerMixin):

    def __init__(self, n_bins=256, subsample=int(1e5), random_state=None):
        self.n_bins = n_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        self.bin_thresholds_ = find_bins(
            X, self.n_bins, subsample=self.subsample,
            random_state=self.random_state)
        return self

    def transform(self, X):
        return map_to_bins(X, binning_thresholds=self.bin_thresholds_)
