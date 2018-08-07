import numpy as np
from numba import njit


HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', np.float32),
    ('sum_hessians', np.float32),
    ('count', np.uint32),
])


@njit(fastmath=True)
def _build_histogram_naive(n_bins, sample_indices, binned_feature,
                           ordered_gradients, ordered_hessians):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    for i, sample_idx in enumerate(sample_indices):
        bin_idx = binned_feature[sample_idx]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_idx]['count'] += 1
    return histogram


@njit(fastmath=True)
def _build_histogram_unrolled(n_bins, sample_indices, binned_feature,
                              ordered_gradients, ordered_hessians):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = sample_indices.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        histogram[bin_0]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_1]['sum_gradients'] += ordered_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += ordered_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += ordered_gradients[i + 3]

        histogram[bin_0]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_1]['sum_hessians'] += ordered_hessians[i + 1]
        histogram[bin_2]['sum_hessians'] += ordered_hessians[i + 2]
        histogram[bin_3]['sum_hessians'] += ordered_hessians[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_idx]['count'] += 1

    return histogram
