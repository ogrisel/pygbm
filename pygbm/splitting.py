from collections import namedtuple
import numpy as np
from numba import njit, float32, uint8

from .histogram import _build_histogram_unrolled


SplitContext = namedtuple('SplitContext', [
    'n_features',
    'binned_features',
    'n_bins',
    'all_gradients',
    'all_hessians',
    'l2_regularization',
])

SplitInfo = namedtuple('SplitInfo', [
    'gain', 'feature_idx', 'bin_idx',
    'gradient_left', 'hessian_left',
    'gradient_right', 'hessian_right',
])


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


@njit(parallel=False)
def find_node_split(sample_indices, grower_context):
    loss_dtype = grower_context.all_gradients.dtype
    ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)
    ordered_hessians = np.empty_like(sample_indices, dtype=loss_dtype)

    for i, sample_idx in enumerate(sample_indices):
        ordered_gradients[i] = grower_context.all_gradients[sample_idx]
        ordered_hessians[i] = grower_context.all_hessians[sample_idx]

    # TODO: use prange to parallelize this loop: we need to properly
    # type SplitInfo (probably using a jitclass) to be able to preallocate
    # the result data structure.
    split_infos = []
    for feature_idx in range(grower_context.n_features):
        binned_feature = grower_context.binned_features[:, feature_idx]
        split_info = _find_histogram_split(
            feature_idx, binned_feature, grower_context.n_bins, sample_indices,
            ordered_gradients, ordered_hessians,
            grower_context.l2_regularization)
        split_infos.append(split_info)

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
