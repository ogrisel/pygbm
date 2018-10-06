import numpy as np
from numba import njit

HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', np.float32),
    ('sum_hessians', np.float32),
    ('count', np.uint32),
])


@njit
def _build_ghc_histogram_naive(n_bins, sample_indices, binned_feature,
                               ordered_gradients, ordered_hessians):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    for i, sample_idx in enumerate(sample_indices):
        bin_idx = binned_feature[sample_idx]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_idx]['count'] += 1
    return histogram


@njit
def _build_ghc_histogram_unrolled_fast(n_bins, parent_histogram,
                                       sibling_histogram):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    unrolled_upper = (n_bins // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = i
        bin_1 = i + 1
        bin_2 = i + 2
        bin_3 = i + 3
        histogram[bin_0]['sum_gradients'] = parent_histogram[bin_0]['sum_gradients'] - sibling_histogram[bin_0]['sum_gradients']
        histogram[bin_1]['sum_gradients'] = parent_histogram[bin_1]['sum_gradients'] - sibling_histogram[bin_1]['sum_gradients']
        histogram[bin_2]['sum_gradients'] = parent_histogram[bin_2]['sum_gradients'] - sibling_histogram[bin_2]['sum_gradients']
        histogram[bin_3]['sum_gradients'] = parent_histogram[bin_3]['sum_gradients'] - sibling_histogram[bin_3]['sum_gradients']

        histogram[bin_0]['sum_hessians'] = parent_histogram[bin_0]['sum_hessians'] - sibling_histogram[bin_0]['sum_hessians']
        histogram[bin_1]['sum_hessians'] = parent_histogram[bin_1]['sum_hessians'] - sibling_histogram[bin_1]['sum_hessians']
        histogram[bin_2]['sum_hessians'] = parent_histogram[bin_2]['sum_hessians'] - sibling_histogram[bin_2]['sum_hessians']
        histogram[bin_3]['sum_hessians'] = parent_histogram[bin_3]['sum_hessians'] - sibling_histogram[bin_3]['sum_hessians']

        histogram[bin_0]['count'] = parent_histogram[bin_0]['count'] - sibling_histogram[bin_0]['count']
        histogram[bin_1]['count'] = parent_histogram[bin_1]['count'] - sibling_histogram[bin_1]['count']
        histogram[bin_2]['count'] = parent_histogram[bin_2]['count'] - sibling_histogram[bin_2]['count']
        histogram[bin_3]['count'] = parent_histogram[bin_3]['count'] - sibling_histogram[bin_3]['count']

    for i in range(unrolled_upper, n_bins):
        histogram[i]['sum_gradients'] = parent_histogram[i]['sum_gradients'] - sibling_histogram[i]['sum_gradients']
        histogram[i]['sum_hessians'] = parent_histogram[i]['sum_hessians'] - sibling_histogram[i]['sum_hessians']
        histogram[i]['count'] = parent_histogram[i]['count'] - sibling_histogram[i]['count']

    return histogram


@njit
def _build_ghc_histogram_unrolled(n_bins, sample_indices, binned_feature,
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


@njit
def _build_gc_histogram_unrolled(n_bins, sample_indices, binned_feature,
                                 ordered_gradients):
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

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['count'] += 1

    return histogram


@njit
def _build_gc_root_histogram_unrolled(n_bins, binned_feature, all_gradients):
    """Special case for the root node

    The root node has to find the a split among all the samples from the
    training set. binned_feature and all_gradients already have a consistent
    ordering.
    """
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = binned_feature.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        histogram[bin_0]['sum_gradients'] += all_gradients[i]
        histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        histogram[bin_idx]['sum_gradients'] += all_gradients[i]
        histogram[bin_idx]['count'] += 1

    return histogram


@njit
def _build_ghc_root_histogram_unrolled(n_bins, binned_feature, all_gradients,
                                       all_hessians):
    """Special case for the root node

    The root node has to find the a split among all the samples from the
    training set. binned_feature and all_gradients already have a consistent
    ordering.
    """
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = binned_feature.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        histogram[bin_0]['sum_gradients'] += all_gradients[i]
        histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]

        histogram[bin_0]['sum_hessians'] += all_hessians[i]
        histogram[bin_1]['sum_hessians'] += all_hessians[i + 1]
        histogram[bin_2]['sum_hessians'] += all_hessians[i + 2]
        histogram[bin_3]['sum_hessians'] += all_hessians[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        histogram[bin_idx]['sum_gradients'] += all_gradients[i]
        histogram[bin_idx]['sum_hessians'] += all_hessians[i]
        histogram[bin_idx]['count'] += 1

    return histogram
