import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin


def find_binning_thresholds(data, max_bins=256, subsample=int(2e5),
                            random_state=None):
    """Extract feature-wise equally-spaced quantiles from numerical data

    Subsample the dataset if too large as the feature-wise quantiles
    should be stable.

    Parameters
    ----------
    data: array-like (n_samples, n_features)
        The numerical dataset to analyse.

    max_bins: int
        The number of bins to extract for each feature. As we code the binned
        values as 8-bit integers, max_bins should be no larger than 256.

    subsample: int
        Number of random subsamples to consider to compute the quantiles.

    random_state: int or numpy.random.RandomState or None
        Pseudo-random number generator to control the random sub-sampling.

    Return
    ------
    binning_thresholds: tuple of arrays
        For each feature, store the increasing numeric values that can
        be used to separate the bins.
        len(binning_thresholds) == n_features
        Each array has size (n_bins - 1) where:
            n_bins == min(max_bins, len(np.unique(data[:, feature_idx])))
    """
    if not (2 <= max_bins <= 256):
        raise ValueError(f'max_bins={max_bins} should be no smaller than 2 '
                         f'and no larger than 256.')
    rng = check_random_state(random_state)
    if data.shape[0] > subsample:
        subset = rng.choice(np.arange(data.shape[0]), subsample)
        data = data[subset]
    dtype = data.dtype
    if dtype.kind != 'f':
        dtype = np.float32

    percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
    binning_thresholds = []
    for f_idx in range(data.shape[1]):
        col_data = np.ascontiguousarray(data[:, f_idx], dtype=dtype)
        distinct_values = np.unique(col_data)
        if len(distinct_values) <= max_bins:
            midpoints = (distinct_values[:-1] + distinct_values[1:])
            midpoints *= .5
        else:
            # We sort again the data in this case. We could compute
            # approximate midpoint percentiles using the output of
            # np.unique(col_data, return_counts) instead but this is more
            # work and the performance benefit will be limited because we
            # work on a fixed-size subsample of the full data.
            midpoints = np.percentile(col_data, percentiles,
                                      interpolation='midpoint').astype(dtype)
        binning_thresholds.append(midpoints)
    return tuple(binning_thresholds)


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

    binning_thresholds = tuple(np.ascontiguousarray(bt, dtype=np.float32)
                               for bt in binning_thresholds)

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
        left, right = 0, binning_thresholds.shape[0]
        while left < right:
            middle = (right + left - 1) // 2
            if data[i] <= binning_thresholds[middle]:
                right = middle
            else:
                left = middle + 1
        binned[i] = left


class BinMapper(BaseEstimator, TransformerMixin):
    # TODO: write docstrings

    def __init__(self, max_bins=256, subsample=int(1e5), random_state=None):
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        self.bin_thresholds_ = find_binning_thresholds(
            X, self.max_bins, subsample=self.subsample,
            random_state=self.random_state)

        self.n_bins_per_feature_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32)

        return self

    def transform(self, X):
        return map_to_bins(X, binning_thresholds=self.bin_thresholds_)
