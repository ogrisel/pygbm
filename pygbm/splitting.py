# from collections import namedtuple
import numpy as np
from numba import (njit, jitclass, prange, float32, uint8, uint32, optional,
                   typeof)
from .histogram import _build_ghc_histogram_unrolled
from .histogram import _subtract_ghc_histograms_unrolled
from .histogram import _build_gc_histogram_unrolled
from .histogram import _build_ghc_root_histogram_unrolled
from .histogram import _build_gc_root_histogram_unrolled
from .histogram import HISTOGRAM_DTYPE


@jitclass([
    ('gain', float32),
    ('feature_idx', uint32),
    ('bin_idx', uint8),
    ('gradient_left', float32),
    ('hessian_left', float32),
    ('gradient_right', float32),
    ('hessian_right', float32),
    ('histogram', typeof(HISTOGRAM_DTYPE)[:]),  # array of size n_bins
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


@jitclass([
    ('n_features', uint32),
    ('binned_features', uint8[::1, :]),
    ('n_bins', uint32),
    ('all_gradients', float32[::1]),
    ('all_hessians', float32[::1]),
    ('constant_hessian', uint8),
    ('l2_regularization', float32),
    ('min_hessian_to_split', float32),
])
class HistogramSplitter:
    def __init__(self, n_features, binned_features, n_bins,
                 all_gradients, all_hessians, l2_regularization,
                 min_hessian_to_split=1e-3):
        self.n_features = n_features
        self.binned_features = binned_features
        self.n_bins = n_bins
        self.all_gradients = all_gradients
        self.all_hessians = all_hessians
        self.constant_hessian = all_hessians.shape[0] == 1
        self.l2_regularization = l2_regularization
        self.min_hessian_to_split = min_hessian_to_split

    def split_indices(self, sample_indices, split_info):
        binned_feature = self.binned_features.T[split_info.feature_idx]
        left_idx, right_idx = 0, 0
        sample_indices_left = np.empty_like(sample_indices)
        sample_indices_right = np.empty_like(sample_indices)
        # TODO: parallelize this loop by blocks (e.g. 1e6-sized chunks)?
        # That would involve some overhead with concatenating the blocks at
        # the end. Alternatively we could make sample_indices a list of numpy
        # arrays in the grower tree node datastructure to ease
        # parallelization.
        for sample_idx in sample_indices:
            if binned_feature[sample_idx] <= split_info.bin_idx:
                sample_indices_left[left_idx] = sample_idx
                left_idx += 1
            else:
                sample_indices_right[right_idx] = sample_idx
                right_idx += 1

        return sample_indices_left[:left_idx], sample_indices_right[:right_idx]

    def find_node_split(self, sample_indices):
        loss_dtype = self.all_gradients.dtype

        if sample_indices.shape[0] == self.all_gradients.shape[0]:
            # Root node: the ordering of sample_indices and all_gradients
            # are expected to be consistent in this case.
            ordered_gradients = self.all_gradients
            ordered_hessians = self.all_hessians
        else:
            ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)
            if self.constant_hessian:
                ordered_hessians = self.all_hessians
                for i, sample_idx in enumerate(sample_indices):
                    ordered_gradients[i] = self.all_gradients[sample_idx]
            else:
                ordered_hessians = np.empty_like(sample_indices,
                                                 dtype=loss_dtype)
                for i, sample_idx in enumerate(sample_indices):
                    ordered_gradients[i] = self.all_gradients[sample_idx]
                    ordered_hessians[i] = self.all_hessians[sample_idx]

        if self.constant_hessian:
            constant_hessian_val = self.all_hessians[0]  # h for 1 sample
            hessian = constant_hessian_val * sample_indices.shape[0]
        else:
            constant_hessian_val = 0.  # won't be used anyway
            hessian = ordered_hessians.sum()

        gradient = ordered_gradients.sum()

        return _parallel_find_split(
            sample_indices,
            ordered_gradients,
            ordered_hessians,
            self.n_features,
            self.binned_features,
            self.n_bins,
            self.l2_regularization,
            self.min_hessian_to_split,
            self.constant_hessian,
            constant_hessian_val,
            gradient,
            hessian)

    def find_node_split_subtraction(self, sample_indices, parent_histograms,
                                    sibling_histograms):

        # hessian of a single sample, if constant (else, it's ignored)
        if self.constant_hessian:
            constant_hessian_val = self.all_hessians[0]
        else:
            constant_hessian_val = 0.

        return _parallel_find_split_subtraction(
            sample_indices,
            self.n_features,
            self.binned_features,
            self.n_bins,
            self.l2_regularization,
            self.min_hessian_to_split,
            parent_histograms,
            sibling_histograms,
            self.constant_hessian,
            constant_hessian_val)


@njit()
def _find_best_feature_to_split_helper(n_features, n_bins, split_infos):

    best_gain = None
    # need to convert to int64, it's a numba bug. See issue #2756
    histograms = np.empty(
        shape=(np.int64(n_features), np.int64(n_bins)),
        dtype=HISTOGRAM_DTYPE
    )
    for i, split_info in enumerate(split_infos):
        histograms[i, :] = split_info.histogram
        gain = split_info.gain
        if best_gain is None or gain > best_gain:
            best_gain = gain
            best_split_info = split_info
    return best_split_info, histograms
   

@njit(parallel=True)
def _parallel_find_split(sample_indices, ordered_gradients, ordered_hessians,
                         n_features, binned_features, n_bins,
                         l2_regularization, min_hessian_to_split,
                         constant_hessian, constant_hessian_val, gradient,
                         hessian):
    """For each feature, find the best bin to split on with
    _find_histogram_split. Returns the best SplitInfo among all features,
    along with all the feature histograms."""
    # Pre-allocate the results datastructure to be able to use prange
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(n_features)]
    for feature_idx in prange(n_features):
        binned_feature = binned_features.T[feature_idx]
        split_info = _find_histogram_split(
            feature_idx, binned_feature, n_bins, sample_indices,
            ordered_gradients, ordered_hessians, l2_regularization,
            min_hessian_to_split, constant_hessian, constant_hessian_val,
            gradient, hessian)
        split_infos[feature_idx] = split_info

    return _find_best_feature_to_split_helper(n_features, n_bins, split_infos)


@njit(parallel=True)
def _parallel_find_split_subtraction(sample_indices,
                                     n_features, binned_features, n_bins,
                                     l2_regularization, min_hessian_to_split,
                                     parent_histograms, sibling_histograms,
                                     constant_hessian, constant_hessian_val):
    """For each feature, find the best bin to split on with
    _find_histogram_split. Returns the best SplitInfo among all features,
    along with all the feature histograms."""
    # Pre-allocate the results datastructure to be able to use prange
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(n_features)]
    for feature_idx in prange(n_features):
        binned_feature = binned_features.T[feature_idx]
        split_info = _find_histogram_split_subtraction(
            feature_idx, binned_feature, n_bins, sample_indices,
            l2_regularization, min_hessian_to_split, parent_histograms,
            sibling_histograms, constant_hessian, constant_hessian_val)
        split_infos[feature_idx] = split_info

    return _find_best_feature_to_split_helper(n_features, n_bins, split_infos)


@njit(locals={'histogram': typeof(HISTOGRAM_DTYPE)[:]},
      fastmath=True)
def _find_histogram_split(feature_idx, binned_feature, n_bins, sample_indices,
                          ordered_gradients, ordered_hessians,
                          l2_regularization, min_hessian_to_split,
                          constant_hessian, constant_hessian_val, gradient,
                          hessian):
    """Compute the histogram for a given feature and return the best bin to
    split on."""

    #gradient = ordered_gradients.sum()
    root_node = binned_feature.shape[0] == sample_indices.shape[0]

    if root_node:
        if constant_hessian:
            histogram = _build_gc_root_histogram_unrolled(
                n_bins, binned_feature, ordered_gradients)
        else:
            histogram = _build_ghc_root_histogram_unrolled(
                n_bins, binned_feature, ordered_gradients,
                ordered_hessians)
    else:
        if constant_hessian:
            histogram = _build_gc_histogram_unrolled(
                n_bins, sample_indices, binned_feature, ordered_gradients)
        else:
            histogram = _build_ghc_histogram_unrolled(
                n_bins, sample_indices, binned_feature, ordered_gradients,
                ordered_hessians)

    return _find_best_bin_to_split_helper(
        feature_idx, n_bins, histogram, min_hessian_to_split,
        l2_regularization, gradient, hessian,
        constant_hessian, constant_hessian_val
    )


@njit(locals={'histogram': typeof(HISTOGRAM_DTYPE)[:],
              'gradient': float32,
              'hessian': float32},
      fastmath=True)
def _find_histogram_split_subtraction(feature_idx, binned_feature, n_bins,
                                      sample_indices, l2_regularization,
                                      min_hessian_to_split, parent_histograms,
                                      sibling_histograms, constant_hessian,
                                      constant_hessian_val):
    """Compute the histogram for a given feature and return the best bin to
    split on. Uses hist(parent) = hist(left) + hist(right)"""

    histogram = _subtract_ghc_histograms_unrolled(
        n_bins, parent_histograms[feature_idx],
        sibling_histograms[feature_idx]
    )

    gradient = np.sum(histogram[:]['sum_gradients'])
    if constant_hessian:
        hessian = constant_hessian_val * sample_indices.shape[0]
    else:
        hessian = np.sum(histogram[:]['sum_hessians'])

    return _find_best_bin_to_split_helper(
        feature_idx, n_bins, histogram, min_hessian_to_split,
        l2_regularization, gradient, hessian,
        constant_hessian, constant_hessian_val
    )


@njit(locals={'gradient_left': float32, 'hessian_left': float32},
      fastmath=True)
def _find_best_bin_to_split_helper(feature_idx, n_bins, histogram,
                                   min_hessian_to_split, l2_regularization,
                                   gradient, hessian, constant_hessian,
                                   constant_hessian_val):
    """For a given feature and the corresponding histogram, find best bin to
    split on and return the corresponding SplitInfo"""
    # Allocate the structure for the best split information. It can be
    # returned as such (with a negative gain) if the min_hessian_to_split
    # condition is not satisfied. Such invalid splits are later discarded by
    # the TreeGrower.
    best_split = SplitInfo(-1., 0, 0, 0., 0., 0., 0.)

    gradient_left, hessian_left = 0., 0.
    for bin_idx in range(n_bins):
        gradient_left += histogram[bin_idx]['sum_gradients']
        if constant_hessian:
            hessian_left += histogram[bin_idx]['count'] * constant_hessian_val
        else:
            hessian_left += histogram[bin_idx]['sum_hessians']
        if hessian_left < min_hessian_to_split:
            continue
        hessian_right = hessian- hessian_left
        if hessian_right < min_hessian_to_split:
            continue
        gradient_right = gradient - gradient_left
        gain = _split_gain(gradient_left, hessian_left,
                           gradient_right, hessian_right,
                           gradient, hessian,
                           l2_regularization)
        if gain > best_split.gain:
            best_split.gain = gain
            best_split.feature_idx = feature_idx
            best_split.bin_idx = bin_idx
            best_split.gradient_left = gradient_left
            best_split.hessian_left = hessian_left
            best_split.gradient_right = gradient_right
            best_split.hessian_right = hessian_right

    best_split.histogram = histogram
    return best_split


@njit(fastmath=False)
def _split_gain(gradient_left, hessian_left, gradient_right, hessian_right,
                gradient, hessian, l2_regularization):
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
    gain -= negative_loss(gradient, hessian)
    return gain
