import warnings
from heapq import heappush, heappop
import numpy as np

from .splitting import HistogramSplitter
from .predictor import TreePredictor, PREDICTOR_RECORD_DTYPE


class TreeNode:
    split_info = None  # Result of the split evaluation
    left_child = None  # Link to left node (only for non-leaf nodes)
    right_child = None  # Link to right node (only for non-leaf nodes)
    value = None  # Prediction value (only for leaf nodes)
    histograms = None  # List of histogram (1 per feature)
    sibling = None  # Link to sibling node, None for root
    parent = None  # Link to parent node, None for root

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians, parent=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians
        self.parent = parent

    def __repr__(self):
        # To help with debugging
        out = f"TreeNode: depth={self.depth}, "
        out += f"samples={len(self.sample_indices)}"
        if self.split_info is not None:
            out += f", feature_idx={self.split_info.feature_idx}"
            out += f", bin_idx={self.split_info.bin_idx}"
        return out

    def __lt__(self, other_node):
        """Comparison for priority queue

        Nodes with high gain are higher priority than nodes with node gain.

        heapq.heappush only need the '<' operator.
        heapq.heappop take the smallest item first (smaller ishigher priority).
        """
        if self.split_info is None or other_node.split_info is None:
            raise ValueError("Cannot compare nodes with split_info")
        return self.split_info.gain > other_node.split_info.gain


class TreeGrower:
    def __init__(self, features_data, all_gradients, all_hessians,
                 max_leaf_nodes=None, max_depth=None, min_samples_leaf=20,
                 min_gain_to_split=0., n_bins=256, l2_regularization=0.,
                 min_hessian_to_split=1e-3, shrinkage=1.):
        if features_data.dtype != np.uint8:
            raise NotImplementedError(
                "Explicit feature binning required for now")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError(f'max_leaf_nodes={max_leaf_nodes} should not be'
                             f' smaller than 1')
        if max_depth is not None and max_depth < 1:
            raise ValueError(f'max_depth={max_depth} should not be'
                             f' smaller than 1')
        if not features_data.flags.f_contiguous:
            warnings.warn("Binned data should be passed as Fortran contiguous"
                          "array for maximum efficiency.")
        self.splitter = HistogramSplitter(
            features_data.shape[1], features_data, n_bins,
            all_gradients, all_hessians, l2_regularization,
            min_hessian_to_split)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.features_data = features_data
        if min_gain_to_split < 0:
            raise ValueError(f"min_gain_to_split should be positive, got: "
                             f"{min_gain_to_split}")
        self.min_gain_to_split = min_gain_to_split
        self.shrinkage = shrinkage
        self.splittable_nodes = []
        self.finalized_leaves = []
        self._intilialize_root()
        self.n_nodes = 1

    def grow(self):
        while self.can_split_further():
            self.split_next()

    def _intilialize_root(self):
        n_samples = self.features_data.shape[0]
        depth = 0
        self.root = TreeNode(depth, np.arange(n_samples, dtype=np.uint32),
                             self.splitter.all_gradients.sum(),
                             self.splitter.all_hessians.sum())
        if (self.max_leaf_nodes is not None and self.max_leaf_nodes == 1):
            self._finalize_leaf(self.root)
            return
        self._compute_spittability(self.root)

    def _compute_spittability(self, node):
        parent_histograms, sibling_histograms = None, None
        if node.parent is not None and node.sibling.histograms is not None:
            parent_histograms = node.parent.histograms
            sibling_histograms = node.sibling.histograms

        split_info, histograms = self.splitter.find_node_split(
            node.sample_indices, parent_histograms, sibling_histograms)
        node.split_info = split_info
        node.histograms = histograms
        if split_info.gain < self.min_gain_to_split:
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    def split_next(self):
        """Split the node with highest potential gain.

        Return the two resulting nodes created by the split.
        """
        if len(self.splittable_nodes) == 0:
            raise StopIteration("No more splittable nodes")

        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)

        sample_indices_left, sample_indices_right = \
            self.splitter.split_indices(node.sample_indices, node.split_info)

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(depth, sample_indices_left,
                                   node.split_info.gradient_left,
                                   node.split_info.hessian_left, parent=node)
        right_child_node = TreeNode(depth, sample_indices_right,
                                    node.split_info.gradient_right,
                                    node.split_info.hessian_right, parent=node)
        left_child_node.sibling = right_child_node
        right_child_node.sibling = left_child_node
        node.right_child = right_child_node
        node.left_child = left_child_node
        self.n_nodes += 2

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)

        elif (self.max_leaf_nodes is not None
                and n_leaf_nodes == self.max_leaf_nodes):
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()

        else:
            if (self.min_samples_leaf is not None
                    and len(sample_indices_left) < self.min_samples_leaf):
                self._finalize_leaf(left_child_node)
            else:
                self._compute_spittability(left_child_node)
            if (self.min_samples_leaf is not None
                    and len(sample_indices_right) < self.min_samples_leaf):
                self._finalize_leaf(right_child_node)
            else:
                self._compute_spittability(right_child_node)
        return left_child_node, right_child_node

    def can_split_further(self):
        return len(self.splittable_nodes) >= 1

    def _finalize_leaf(self, node):
        """Compute the prediction value that minimizes the objective function

        See Equation 5 of:
        XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
        https://arxiv.org/abs/1603.02754
        """
        node.value = self.shrinkage * node.sum_gradients / (
            node.sum_hessians + self.splitter.l2_regularization)
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self, bin_thresholds=None):
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        self._fill_predictor_node_array(predictor_nodes, self.root,
                                        bin_thresholds=bin_thresholds)
        return TreePredictor(predictor_nodes)

    def _fill_predictor_node_array(self, predictor_nodes, grower_node,
                                   bin_thresholds=None, next_free_idx=0):
        node = predictor_nodes[next_free_idx]
        node['count'] = grower_node.sample_indices.shape[0]
        node['depth'] = grower_node.depth
        if grower_node.value is not None:
            # Leaf node
            node['is_leaf'] = True
            node['value'] = grower_node.value
            return next_free_idx + 1
        else:
            # Decision node
            split_info = grower_node.split_info
            feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
            node['feature_idx'] = feature_idx
            node['bin_threshold'] = bin_idx
            if bin_thresholds is not None:
                threshold = bin_thresholds[feature_idx][bin_idx]
                node['threshold'] = threshold
                node['gain'] = split_info.gain
            next_free_idx += 1

            node['left'] = next_free_idx
            next_free_idx = self._fill_predictor_node_array(
                predictor_nodes, grower_node.left_child,
                bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)

            node['right'] = next_free_idx
            return self._fill_predictor_node_array(
                predictor_nodes, grower_node.right_child,
                bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)
