from collections import namedtuple
import numpy as np
from numba import njit, jitclass, prange, float32, uint8, uint32

from .histogram import _build_histogram_unrolled


SplitContext = namedtuple('SplitContext', [
    'n_features',
    'binned_features',
    'n_bins',
    'all_gradients',
    'all_hessians',
    'l2_regularization',
])


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


@njit
def split_indices(sample_indices, split_info, context):
    binned_feature = context.binned_features[:, split_info.feature_idx]
    sample_indices_left, sample_indices_right = [], []
    for sample_idx in sample_indices:
        if binned_feature[sample_idx] <= split_info.bin_idx:
            sample_indices_left.append(sample_idx)
        else:
            sample_indices_right.append(sample_idx)

    return (np.array(sample_indices_left, dtype=np.uint32),
            np.array(sample_indices_right, dtype=np.uint32))


@njit(parallel=True)
def _parallel_find_splits(sample_indices, ordered_gradients, ordered_hessians,
                          n_features, binned_features, n_bins,
                          l2_regularization):
    # Pre-allocate the results datastructure to be able to use prange
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(n_features)]
    for feature_idx in prange(n_features):
        binned_feature = binned_features[:, feature_idx]
        split_info = _find_histogram_split(
            feature_idx, binned_feature, n_bins, sample_indices,
            ordered_gradients, ordered_hessians, l2_regularization)
        split_infos[feature_idx] = split_info
    return split_infos


@njit(locals={'l2_regularization': float32})
def find_node_split(sample_indices, context):
    loss_dtype = context.all_gradients.dtype
    ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)
    ordered_hessians = np.empty_like(sample_indices, dtype=loss_dtype)

    for i, sample_idx in enumerate(sample_indices):
        ordered_gradients[i] = context.all_gradients[sample_idx]
        ordered_hessians[i] = context.all_hessians[sample_idx]

    # XXX: To avoid dividing by 0 (but why?)
    l2_regularization = max(context.l2_regularization, 1e-8)

    split_infos = _parallel_find_splits(sample_indices, ordered_gradients,
                                        ordered_hessians, context.n_features,
                                        context.binned_features,
                                        context.n_bins,
                                        l2_regularization)
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
              'best_gain': float32, 'best_bin_idx': uint8},
      fastmath=True)
def _find_histogram_split(feature_idx, binned_feature, n_bins, sample_indices,
                          ordered_gradients, ordered_hessians,
                          l2_regularization):

    gradient_parent = ordered_gradients.sum()
    hessian_parent = ordered_hessians.sum()

    histogram = _build_histogram_unrolled(
        n_bins, sample_indices, binned_feature,
        ordered_gradients, ordered_hessians)

    gradient_left, hessian_left = 0., 0.
    best_gain = -1.
    for bin_idx in range(histogram.shape[0]):
        gradient_left += histogram[bin_idx]['sum_gradients']
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
