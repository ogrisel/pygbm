# from collections import namedtuple
import numpy as np
from numba import njit, jitclass, prange, float32, uint8, uint32
from .histogram import _build_ghc_histogram_unrolled
from .histogram import _build_gc_histogram_unrolled


@jitclass([
    ('gain', float32),
    ('feature_idx', uint32),
    ('bin_idx', uint8),
    ('gradient_left', float32),
    ('hessian_left', float32),
    ('gradient_right', float32),
    ('hessian_right', float32),
])
class SplitInfo:
    def __init__(self, gain=0, feature_idx=0, bin_idx=0,
                 gradient_left=0., hessian_left=0.,
                 gradient_right=0., hessian_right=0.):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.gradient_left = gradient_left
        self.hessian_left = hessian_left
        self.gradient_right = gradient_right
        self.hessian_right = hessian_right


@njit(parallel=True)
def _parallel_find_splits(sample_indices, ordered_gradients, ordered_hessians,
                          n_features, binned_features, n_bins,
                          l2_regularization):
    # Pre-allocate the results datastructure to be able to use prange
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(n_features)]
    for feature_idx in prange(n_features):
        binned_feature = binned_features.T[feature_idx]
        split_info = _find_histogram_split(
            feature_idx, binned_feature, n_bins, sample_indices,
            ordered_gradients, ordered_hessians, l2_regularization)
        split_infos[feature_idx] = split_info
    return split_infos


@jitclass([
    ('n_features', uint32),
    ('binned_features', uint8[::1, :]),
    ('n_bins', uint32),
    ('all_gradients', float32[::1]),
    ('all_hessians', float32[::1]),
    ('constant_hessian', uint8),
    ('l2_regularization', float32),
])
class HistogramSplitter:
    def __init__(self, n_features, binned_features, n_bins,
                 all_gradients, all_hessians, l2_regularization):
        self.n_features = n_features
        self.binned_features = binned_features
        self.n_bins = n_bins
        self.all_gradients = all_gradients
        self.all_hessians = all_hessians
        self.constant_hessian = all_hessians.shape[0] == 1
        # XXX: To avoid dividing by 0 (but why?)
        self.l2_regularization = max(l2_regularization, 1e-8)

    def split_indices(self, sample_indices, split_info):
        binned_feature = self.binned_features.T[split_info.feature_idx]
        sample_indices_left, sample_indices_right = [], []
        for sample_idx in sample_indices:
            if binned_feature[sample_idx] <= split_info.bin_idx:
                sample_indices_left.append(sample_idx)
            else:
                sample_indices_right.append(sample_idx)

        return (np.array(sample_indices_left, dtype=np.uint32),
                np.array(sample_indices_right, dtype=np.uint32))

    def find_node_split(self, sample_indices):
        loss_dtype = self.all_gradients.dtype
        ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)

        if self.constant_hessian:
            ordered_hessians = self.all_hessians
            for i, sample_idx in enumerate(sample_indices):
                ordered_gradients[i] = self.all_gradients[sample_idx]
        else:
            ordered_hessians = np.empty_like(sample_indices, dtype=loss_dtype)
            for i, sample_idx in enumerate(sample_indices):
                ordered_gradients[i] = self.all_gradients[sample_idx]
                ordered_hessians[i] = self.all_hessians[sample_idx]

        split_infos = _parallel_find_splits(sample_indices, ordered_gradients,
                                            ordered_hessians, self.n_features,
                                            self.binned_features,
                                            self.n_bins,
                                            self.l2_regularization)
        best_gain = None
        for split_info in split_infos:
            gain = split_info.gain
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_split_info = split_info
        return best_split_info


@njit(fastmath=False)
def _split_gain(gradient_left, hessian_left, gradient_right, hessian_right,
                gradient_parent, hessian_parent, l2_regularization):
    """Loss reduction

    Compute the reduction in loss after taking a split compared to keeping
    the node a leaf of the tree.

    See Equation 7 of:
    XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
    https://arxiv.org/abs/1603.02754
    """
    def negative_loss(gradient, hessian):
        return (gradient ** 2) / (hessian + l2_regularization)

    gain = negative_loss(gradient_left, hessian_left)
    gain += negative_loss(gradient_right, hessian_right)
    gain -= negative_loss(gradient_parent, hessian_parent)
    return gain


@njit(locals={'gradient_left': float32, 'hessian_left': float32,
              'best_gain': float32, 'best_bin_idx': uint8,
              'constant_hessian': uint8},
      fastmath=True)
def _find_histogram_split(feature_idx, binned_feature, n_bins, sample_indices,
                          ordered_gradients, ordered_hessians,
                          l2_regularization):

    gradient_parent = ordered_gradients.sum()
    constant_hessian = ordered_hessians.shape[0] == 1
    if constant_hessian:
        hessian_parent = ordered_hessians[0] * sample_indices.shape[0]
        histogram = _build_gc_histogram_unrolled(
            n_bins, sample_indices, binned_feature,
            ordered_gradients)
    else:
        hessian_parent = ordered_hessians.sum()
        histogram = _build_ghc_histogram_unrolled(
            n_bins, sample_indices, binned_feature,
            ordered_gradients, ordered_hessians)

    gradient_left, hessian_left = 0., 0.
    best_gain = -1.
    for bin_idx in range(histogram.shape[0]):
        gradient_left += histogram[bin_idx]['sum_gradients']
        if constant_hessian:
            hessian_left += histogram[bin_idx]['count'] * ordered_hessians[0]
        else:
            hessian_left += histogram[bin_idx]['sum_hessians']
        gradient_right = gradient_parent - gradient_left
        hessian_right = hessian_parent - hessian_left
        gain = _split_gain(gradient_left, hessian_left,
                           gradient_right, hessian_right,
                           gradient_parent, hessian_parent,
                           l2_regularization)
        if gain >= best_gain:
            best_gain = gain
            best_bin_idx = bin_idx
            best_gradient_left = gradient_left
            best_hessian_left = hessian_left
            best_gradient_right = gradient_right
            best_hessian_right = hessian_right

    return SplitInfo(best_gain,
                     feature_idx,
                     best_bin_idx,
                     best_gradient_left,
                     best_hessian_left,
                     best_gradient_right,
                     best_hessian_right)
