import warnings
from heapq import heappush, heappop
import numpy as np
from time import time

from .splitting import (SplittingContext, split_indices, find_node_split,
                        find_node_split_subtraction)
from .predictor import TreePredictor, PREDICTOR_RECORD_DTYPE


class TreeNode:
    split_info = None  # Result of the split evaluation
    left_child = None  # Link to left node (only for non-leaf nodes)
    right_child = None  # Link to right node (only for non-leaf nodes)
    value = None  # Prediction value (only for leaf nodes)
    histograms = None  # array of histogram shape = (n_features, n_bins)
    sibling = None  # Link to sibling node, None for root
    parent = None  # Link to parent node, None for root
    find_split_time = 0.  # time spent finding the best split
    construction_speed = 0.  # number of samples / find_split_time
    apply_split_time = 0.  # time spent splitting the node
    # wheter the subtraction method was used for histogram computation
    hist_subtraction = False

    def __init__(self, depth, sample_indices, sum_gradients,
                 sum_hessians, parent=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.n_samples = sample_indices.shape[0]
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
                 min_gain_to_split=0., max_bins=256, n_bins_per_feature=None,
                 l2_regularization=0., min_hessian_to_split=1e-3,
                 shrinkage=1.):
        if features_data.dtype != np.uint8:
            raise NotImplementedError(
                "Explicit feature binning required for now")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError(f'max_leaf_nodes={max_leaf_nodes} should not be'
                             f' smaller than 1')
        if max_depth is not None and max_depth < 1:
            raise ValueError(f'max_depth={max_depth} should not be'
                             f' smaller than 1')
        if min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf={min_samples_leaf} should '
                             f'not be smaller than 1')
        if not features_data.flags.f_contiguous:
            warnings.warn("Binned data should be passed as Fortran contiguous"
                          "array for maximum efficiency.")

        if n_bins_per_feature is None:
            n_bins_per_feature = max_bins

        if isinstance(n_bins_per_feature, int):
            n_bins_per_feature = np.array(
                [n_bins_per_feature] * features_data.shape[1],
                dtype=np.uint32)

        self.splitting_context = SplittingContext(
            features_data.shape[1], features_data, max_bins,
            n_bins_per_feature, all_gradients, all_hessians,
            l2_regularization, min_hessian_to_split, min_samples_leaf,
            min_gain_to_split)
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
        self.total_find_split_time = 0.  # time spent finding the best splits
        self.total_apply_split_time = 0.  # time spent splitting nodes
        self._intilialize_root()
        self.n_nodes = 1

    def grow(self):
        while self.can_split_further():
            self.split_next()

    def _intilialize_root(self):
        n_samples = self.features_data.shape[0]
        depth = 0
        if self.splitting_context.constant_hessian:
            hessian = self.splitting_context.all_hessians[0] * n_samples
        else:
            hessian = self.splitting_context.all_hessians.sum()
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitting_context.partition.view(),
            sum_gradients=self.splitting_context.all_gradients.sum(),
            sum_hessians=hessian
        )
        if (self.max_leaf_nodes is not None and self.max_leaf_nodes == 1):
            self._finalize_leaf(self.root)
            return
        if self.root.n_samples < 2 * self.min_samples_leaf:
            # Do not even bother computing any splitting statistics.
            self._finalize_leaf(self.root)
            return

        self._compute_spittability(self.root)

    def _compute_spittability(self, node, only_hist=False):
        """Compute histograms and split_info of a node and either make it a
        leaf or push it on the splittable node heap.

        only_hist is used when _compute_spittability was called by a sibling
        node: we only want to compute the histograms, not finalize or push
        the node. If _compute_spittability is called again by the grower on
        this same node, the histograms won't be computed again.
        """
        # Compute split_info and histograms if not already done
        if node.split_info is None and node.histograms is None:
            # If the sibling has less samples, compute its hist first (with
            # the regular method) and use the subtraction method for the
            # current node
            if node.sibling is not None:  # root has no sibling
                if node.sibling.n_samples < node.n_samples:
                    self._compute_spittability(node.sibling, only_hist=True)
                    # As hist of sibling is now computed we'll use the hist
                    # subtraction method for the current node.
                    node.hist_subtraction = True

            tic = time()
            if node.hist_subtraction:
                split_info, histograms = find_node_split_subtraction(
                    self.splitting_context, node.sample_indices,
                    node.parent.histograms, node.sibling.histograms)
            else:
                split_info, histograms = find_node_split(
                    self.splitting_context, node.sample_indices)
            toc = time()
            node.find_split_time = toc - tic
            self.total_find_split_time += node.find_split_time
            node.construction_speed = node.n_samples / node.find_split_time
            node.split_info = split_info
            node.histograms = histograms

        if only_hist:
            # _compute_spittability was called by a sibling. We only needed to
            # compute the histogram.
            return

        if node.split_info.gain <= 0:  # no valid split
            # Note: this condition is reached if either all the leaves are
            # pure (best gain = 0), or if no split would satisfy the
            # constraints, (min_hessians_to_split, min_gain_to_split,
            # min_samples_leaf)
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

        tic = time()
        (sample_indices_left, sample_indices_right) = split_indices(
            self.splitting_context, node.split_info, node.sample_indices)
        toc = time()
        node.apply_split_time = toc - tic
        self.total_apply_split_time += node.apply_split_time

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(depth,
                                   sample_indices_left,
                                   node.split_info.gradient_left,
                                   node.split_info.hessian_left,
                                   parent=node)
        right_child_node = TreeNode(depth,
                                    sample_indices_right,
                                    node.split_info.gradient_right,
                                    node.split_info.hessian_right,
                                    parent=node)
        left_child_node.sibling = right_child_node
        right_child_node.sibling = left_child_node
        node.right_child = right_child_node
        node.left_child = left_child_node
        self.n_nodes += 2

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            return left_child_node, right_child_node

        if (self.max_leaf_nodes is not None
                and n_leaf_nodes == self.max_leaf_nodes):
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        if left_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(left_child_node)
        else:
            self._compute_spittability(left_child_node)

        if right_child_node.n_samples < self.min_samples_leaf * 2:
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
            node.sum_hessians + self.splitting_context.l2_regularization)
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
        node['count'] = grower_node.n_samples
        node['depth'] = grower_node.depth
        if grower_node.split_info is not None:
            node['gain'] = grower_node.split_info.gain
        else:
            node['gain'] = -1

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
            next_free_idx += 1

            node['left'] = next_free_idx
            next_free_idx = self._fill_predictor_node_array(
                predictor_nodes, grower_node.left_child,
                bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)

            node['right'] = next_free_idx
            return self._fill_predictor_node_array(
                predictor_nodes, grower_node.right_child,
                bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)
