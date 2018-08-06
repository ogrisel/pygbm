import numpy as np
from heapq import heappush, heappop
from .histogram import _build_histogram_unrolled, find_split


class TreeNode:
    split_info = None
    left_child = None
    right_child = None

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians):
        self.depth = depth
        self.sample_indices = sample_indices
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians


class TreeGrower:
    def __init__(self, n_bins, features_data, all_gradients, all_hessians,
                 max_leaf_nodes=None, max_depth=None, min_gain_split=1e-7,
                 l2_regularization=1e-7):
        if features_data.dtype != np.uint8:
            raise NotImplementedError(
                "Explicit feature binning required for now")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError(f'max_leaf_nodes={max_leaf_nodes} should not be'
                             f' smaller than 1')
        if max_depth is not None and max_depth < 1:
            raise ValueError(f'max_depth={max_depth} should not be'
                             f' smaller than 1')
        self.n_bins = n_bins
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.features_data = features_data
        self.all_gradients = all_gradients
        self.all_hessians = all_hessians
        self.min_gain_split = min_gain_split
        self.l2_regularization = l2_regularization
        self.splittable_nodes = []
        self.finalized_leaves = []

        # Initialize the root node
        n_samples = features_data.shape[0]
        depth = 0
        self.root = TreeNode(depth, np.arange(n_samples, dtype=np.uint32),
                             all_gradients.sum(), all_hessians.sum())
        if self.max_leaf_nodes is not None and self.max_leaf_nodes == 1:
            self.finalize_leaf(self.root)
            return
        self._add_node(self.root)

    def _add_node(self, node, parent=None, side=None):
        if parent is not None:
            if side == 'left':
                parent.left_child = node
            elif side == 'right':
                parent.right_child = node
            else:
                raise ValueError(f'side={side} should be "left" or "right"')
        split_gain = self.find_best_split(node)
        if split_gain < self.min_gain_split:
            self.finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, (split_gain, node))

    def split_next(self):
        """Split the node with highest potential gain.

        Return False if there are no remaining splittable nodes after this
        split.
        """
        if len(self.splittable_nodes) == 0:
            raise StopIteration("No more splittable nodes")

        _, node = heappop(self.splittable_nodes)

        split_bin_idx = node.split_info.bin_idx
        split_feature_idx = node.split_info.feature_idx
        sample_indices_left, sample_indices_right = [], []
        binned_feature = self.features_data[:, split_feature_idx]

        # TODO: Benchmarl / profile and use numba to speed up this leep
        for sample_idx in node.sample_indices:
            if binned_feature[sample_idx] <= split_bin_idx:
                sample_indices_left.append(sample_idx)
            else:
                sample_indices_right.append(sample_idx)

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(depth, np.array(sample_indices_left),
                                   node.split_info.gradient_left,
                                   node.split_info.hessian_left)
        right_child_node = TreeNode(depth, np.array(sample_indices_right),
                                    node.split_info.gradient_right,
                                    node.split_info.hessian_right)
        if self.max_depth is not None and depth == self.max_depth:
            self.finalize_leaf(left_child_node)
            self.finalize_leaf(right_child_node, parent=node, side='right')

        elif (self.max_leaf_nodes is not None
                and n_leaf_nodes == self.max_leaf_nodes):
            self.finalize_leaf(left_child_node)
            self.finalize_leaf(right_child_node)
            self.finalize_splittable_nodes()

        else:
            self._add_node(left_child_node, parent=node, side='left')
            self._add_node(right_child_node, parent=node, side='right')
        return left_child_node, right_child_node

    def can_split_further(self):
        return len(self.splittable_nodes) >= 1

    def finalize_leaf(self, node):
        # TODO: store preduction score on the node
        self.finalized_leaves.append(node)

    def finalize_splittable_nodes(self):
        while len(self.splittable_nodes) > 0:
            _, node = self.splittable_nodes.pop()
            self.finalize_leaf(node)

    def find_best_split(self, node):
        # TODO: extract this logic as a HistogramSplitter that returns
        # the best split info for a given node.
        all_gradients = self.all_gradients
        all_hessians = self.all_hessians
        sample_indices = node.sample_indices
        loss_dtype = all_gradients.dtype
        ordered_gradients = np.empty_like(sample_indices, dtype=loss_dtype)
        ordered_hessians = np.empty_like(sample_indices, dtype=loss_dtype)

        for i, sample_idx in enumerate(sample_indices):
            ordered_gradients[i] = all_gradients[sample_idx]
            ordered_hessians[i] = all_hessians[sample_idx]

        # TODO: parallelize this loop using numba (or dask)
        split_infos = []
        for feature_idx in range(self.features_data.shape[1]):
            binned_feature = self.features_data[:, feature_idx]
            histogram = _build_histogram_unrolled(
                self.n_bins, sample_indices, binned_feature,
                ordered_gradients, ordered_hessians)

            split_infos.append(find_split(histogram, feature_idx,
                                          node.sum_gradients,
                                          node.sum_hessians,
                                          self.l2_regularization))

        best_gain = None
        for split_info in split_infos:
            gain = split_info.gain
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_split_info = split_info
        node.split_info = best_split_info
        return best_gain
