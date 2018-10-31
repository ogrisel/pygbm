# from collections import namedtuple
import numpy as np
from numba import njit, jitclass, prange, float32, uint8, uint32, typeof
import numba
from .histogram import _build_histogram
from .histogram import _subtract_histograms
from .histogram import _build_histogram_no_hessian
from .histogram import _build_histogram_root
from .histogram import _build_histogram_root_no_hessian
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
    ('constant_hessian_value', float32),
    ('l2_regularization', float32),
    ('min_hessian_to_split', float32),
])
class SplittingContext:
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
        if self.constant_hessian:
            self.constant_hessian_value = self.all_hessians[0]  # 1 scalar
        else:
            self.constant_hessian_value = float32(1.)  # won't be used anyway


@njit(parallel=True,
      locals={'sample_idx': uint32,
              'left_count': uint32,
              'right_count': uint32})
def split_indices(context, sample_indices, split_info):
    """Split samples into left and right arrays.

    This is a multi-threaded implementation inspired by lightgbm.
    Quick break-down:
    - samples indices is devided into n_threads regions. Each thread will
      be responsible for its own region. Here is an example with 4
      threads and 24 samples:
      samples_indices = [abcdef|ghijkl|mnopqr|stuvwx]
    - Each thread processes 6 = 24 // 4 entries and maps them
      into tmp_left or tmp_right. For example,
      we could have the following mapping:
      - tmp_left =  [abef..|il....|mnopqr|tux...]
      - tmp_right = [cd....|ghjk..|......|svw...]
    - We keep track of the start positions of the regions (the '|') as
      well as the size of each region. We also keep track of the number
      of samples put into the left/right node by each thread. Concretely:
      - left_cnt =  [4, 2, 6, 3]
      - right_cnt = [2, 4, 0, 3]
    - Finally, we put tmp_left and tmp_right into the returned arrays
      sample_indices_left and sample_indices_right. This is done with the
      help of left_write_offset and right_write_offset, whose entries are
      here denoted by '^':
      - sample_indices_left =   [abefilmnopqrtux.........]
                                 ^   ^ ^     ^
      - samples_indices_right = [cdghjksvw...............]
                                 ^ ^   ^
                                       ^
    """

    binned_feature = context.binned_features.T[split_info.feature_idx]

    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    n_samples = sample_indices.shape[0]

    sample_indices_left = np.empty_like(sample_indices)
    sample_indices_right = np.empty_like(sample_indices)

    tmp_left = np.empty_like(sample_indices)
    tmp_right = np.empty_like(sample_indices)

    sizes = np.full(n_threads, n_samples // n_threads, dtype=np.int32)
    sizes[:n_samples % n_threads] += 1
    starts = np.zeros(n_threads, dtype=np.int32)
    starts[1:] = np.cumsum(sizes[:-1])

    left_counts = np.empty(n_threads, dtype=np.int32)
    right_counts = np.empty(n_threads, dtype=np.int32)

    # map indices to tmp_left and tmp_right
    for thread_idx in prange(n_threads):
        left_count = 0
        right_count = 0

        start = starts[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            sample_idx = sample_indices[i]
            if binned_feature[sample_idx] <= split_info.bin_idx:
                tmp_left[start + left_count] = sample_idx
                left_count += 1
            else:
                tmp_right[start + right_count] = sample_idx
                right_count += 1

        left_counts[thread_idx] = left_count
        right_counts[thread_idx] = right_count

    left_write_offset = np.zeros(n_threads, dtype=np.int32)
    right_write_offset = np.zeros(n_threads, dtype=np.int32)
    left_write_offset[1:] = np.cumsum(left_counts[:-1])
    right_write_offset[1:] = np.cumsum(right_counts[:-1])

    # put tmp_left and tmp_right into samples_indices_left/right
    for thread_idx in prange(n_threads):

        start = starts[thread_idx]
        stop = starts[thread_idx] + left_counts[thread_idx]
        for i, j in enumerate(range(start, stop)):
            sample_indices_left[left_write_offset[thread_idx] + i] = \
                tmp_left[j]

        start = starts[thread_idx]
        stop = starts[thread_idx] + right_counts[thread_idx]
        for i, j in enumerate(range(start, stop)):
            sample_indices_right[right_write_offset[thread_idx] + i] = \
                tmp_right[j]

    return (sample_indices_left[:left_counts.sum()],
            sample_indices_right[:right_counts.sum()])


@njit()
def find_node_split(context, sample_indices):
    loss_dtype = context.all_gradients.dtype

    if sample_indices.shape[0] == context.all_gradients.shape[0]:
        # Root node: the ordering of sample_indices and all_gradients
        # are expected to be consistent in this case.
        ordered_gradients = context.all_gradients
        ordered_hessians = context.all_hessians
    else:
        ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)
        if context.constant_hessian:
            ordered_hessians = context.all_hessians
            for i, sample_idx in enumerate(sample_indices):
                ordered_gradients[i] = context.all_gradients[sample_idx]
        else:
            ordered_hessians = np.empty_like(sample_indices,
                                             dtype=loss_dtype)
            for i, sample_idx in enumerate(sample_indices):
                ordered_gradients[i] = context.all_gradients[sample_idx]
                ordered_hessians[i] = context.all_hessians[sample_idx]

    sum_gradients = ordered_gradients.sum()
    if context.constant_hessian:
        n_samples = sample_indices.shape[0]
        sum_hessians = context.constant_hessian_value * float32(n_samples)
    else:
        sum_hessians = ordered_hessians.sum()

    return _parallel_find_split(
        context, sample_indices, ordered_gradients, ordered_hessians,
        sum_gradients, sum_hessians)


@njit()
def find_node_split_subtraction(context, sample_indices, parent_histograms,
                                sibling_histograms):

    # We can pick any feature (here the first) in the histograms to
    # compute the gradients: they must be the same across all features
    # anyway, we have tests ensuring this. Maybe a more robust way would
    # be to compute an average but it's probably not worth it.
    sum_gradients = (parent_histograms[0]['sum_gradients'].sum() -
                     sibling_histograms[0]['sum_gradients'].sum())

    if context.constant_hessian:
        n_samples = sample_indices.shape[0]
        sum_hessians = context.constant_hessian_value * float32(n_samples)
    else:
        sum_hessians = (parent_histograms[0]['sum_hessians'].sum() -
                        sibling_histograms[0]['sum_hessians'].sum())

    return _parallel_find_split_subtraction(
        context, parent_histograms, sibling_histograms,
        sum_gradients, sum_hessians)


@njit
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
def _parallel_find_split(splitter, sample_indices, ordered_gradients,
                         ordered_hessians, sum_gradients, sum_hessians):
    """For each feature, find the best bin to split on by scanning data.

    This is done by calling _find_histogram_split that compute histograms
    for the samples that reached this node.

    Returns the best SplitInfo among all features, along with all the feature
    histograms that can be latter used to compute the sibling or children
    histograms by substraction.
    """
    # Pre-allocate the results datastructure to be able to use prange:
    # numba jitclass do not seem to properly support default values for kwargs.
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(splitter.n_features)]
    for feature_idx in prange(splitter.n_features):
        split_info = _find_histogram_split(
            splitter, feature_idx, sample_indices,
            ordered_gradients, ordered_hessians, sum_gradients, sum_hessians)
        split_infos[feature_idx] = split_info

    return _find_best_feature_to_split_helper(
        splitter.n_features, splitter.n_bins, split_infos)


@njit(parallel=True)
def _parallel_find_split_subtraction(context, parent_histograms,
                                     sibling_histograms,
                                     sum_gradients, sum_hessians):
    """For each feature, find the best bin to split by histogram substraction

    This in turn calls _find_histogram_split_subtraction that does not need
    to scan the samples from this node and can therefore be significantly
    faster than computing the histograms from data.

    Returns the best SplitInfo among all features, along with all the feature
    histograms that can be latter used to compute the sibling or children
    histograms by substraction.
    """
    # Pre-allocate the results datastructure to be able to use prange
    split_infos = [SplitInfo(0, 0, 0, 0., 0., 0., 0.)
                   for i in range(context.n_features)]
    for feature_idx in prange(context.n_features):
        split_info = _find_histogram_split_subtraction(
            context, feature_idx, parent_histograms, sibling_histograms,
            sum_gradients, sum_hessians)
        split_infos[feature_idx] = split_info

    return _find_best_feature_to_split_helper(
        context.n_features, context.n_bins, split_infos)


@njit(fastmath=True)
def _find_histogram_split(context, feature_idx, sample_indices,
                          ordered_gradients, ordered_hessians,
                          sum_gradients, sum_hessians):
    """Compute the histogram for a given feature and return the best bin."""
    binned_feature = context.binned_features.T[feature_idx]
    root_node = binned_feature.shape[0] == sample_indices.shape[0]

    if root_node:
        if context.constant_hessian:
            histogram = _build_histogram_root_no_hessian(
                context.n_bins, binned_feature, ordered_gradients)
        else:
            histogram = _build_histogram_root(
                context.n_bins, binned_feature, ordered_gradients,
                ordered_hessians)
    else:
        if context.constant_hessian:
            histogram = _build_histogram_no_hessian(
                context.n_bins, sample_indices, binned_feature,
                ordered_gradients)
        else:
            histogram = _build_histogram(
                context.n_bins, sample_indices, binned_feature,
                ordered_gradients, ordered_hessians)

    return _find_best_bin_to_split_helper(
        context, feature_idx, histogram, sum_gradients, sum_hessians)


@njit(fastmath=True)
def _find_histogram_split_subtraction(context, feature_idx,
                                      parent_histograms, sibling_histograms,
                                      sum_gradients, sum_hessians):
    """Compute the histogram by substraction of parent and sibling

    Uses the identity: hist(parent) = hist(left) + hist(right)
    """
    histogram = _subtract_histograms(
        context.n_bins, parent_histograms[feature_idx],
        sibling_histograms[feature_idx])

    return _find_best_bin_to_split_helper(
        context, feature_idx, histogram, sum_gradients, sum_hessians)


@njit(locals={'gradient_left': float32, 'hessian_left': float32},
      fastmath=True)
def _find_best_bin_to_split_helper(context, feature_idx, histogram,
                                   sum_gradients, sum_hessians):
    """Find best bin to split on and return the corresponding SplitInfo"""
    # Allocate the structure for the best split information. It can be
    # returned as such (with a negative gain) if the min_hessian_to_split
    # condition is not satisfied. Such invalid splits are later discarded by
    # the TreeGrower.
    best_split = SplitInfo(-1., 0, 0, 0., 0., 0., 0.)

    gradient_left, hessian_left = 0., 0.
    for bin_idx in range(context.n_bins):
        gradient_left += histogram[bin_idx]['sum_gradients']
        if context.constant_hessian:
            hessian_left += (histogram[bin_idx]['count'] *
                             context.constant_hessian_value)
        else:
            hessian_left += histogram[bin_idx]['sum_hessians']
        if hessian_left < context.min_hessian_to_split:
            continue
        hessian_right = sum_hessians - hessian_left
        if hessian_right < context.min_hessian_to_split:
            continue
        gradient_right = sum_gradients - gradient_left
        gain = _split_gain(gradient_left, hessian_left,
                           gradient_right, hessian_right,
                           sum_gradients, sum_hessians,
                           context.l2_regularization)
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
                sum_gradients, sum_hessians, l2_regularization):
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
    gain -= negative_loss(sum_gradients, sum_hessians)
    return gain
