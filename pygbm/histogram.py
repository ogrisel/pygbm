import numpy as np
from numba import njit


HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', np.float32),
    ('sum_hessians', np.float32),
    ('count', np.uint32),
])


def build_histogram(n_bins, sample_indices, binned_feature,
                    ordered_gradients, ordered_hessians):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _build_histogram(histogram, sample_indices, binned_feature,
                     ordered_gradients, ordered_hessians)
    return histogram


@njit(fastmath=True)
def _build_histogram(histogram, sample_indices, binned_feature,
                     ordered_gradients, ordered_hessians):
    for i, sample_idx in enumerate(sample_indices):
        bin_idx = binned_feature[sample_idx]
        histogram[bin_idx].count += 1
        histogram[bin_idx].sum_gradients += ordered_gradients[i]
        histogram[bin_idx].sum_hessians += ordered_hessians[i]
    return histogram
