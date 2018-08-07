import warnings
from heapq import heappush, heappop
import numpy as np

from .splitting import SplitContext, find_node_split, split_indices


class TreeNode:
    split_info = None
    left_child = None
    right_child = None

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians):
        self.depth = depth
        self.sample_indices = sample_indices
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians

    def __repr__(self):
        # To help with debugging
        out = f"TreeNode: depth={self.depth}"
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
                 max_leaf_nodes=None, max_depth=None, min_gain_to_split=0.,
                 l2_regularization=0., n_bins=256):
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
        self.context = SplitContext(
            features_data.shape[1], features_data, n_bins,
            all_gradients, all_hessians, l2_regularization)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.features_data = features_data
        self.min_gain_to_split = min_gain_to_split
        self.splittable_nodes = []
        self.finalized_leaves = []
        self._intilialize_root()

    def _intilialize_root(self):
        n_samples = self.features_data.shape[0]
        depth = 0
        self.root = TreeNode(depth, np.arange(n_samples, dtype=np.uint32),
                             self.context.all_gradients.sum(),
                             self.context.all_hessians.sum())
        if self.max_leaf_nodes is not None and self.max_leaf_nodes == 1:
            self._finalize_leaf(self.root)
            return
        self._consider_splittability(self.root)

    def _consider_splittability(self, node):
        split_info = find_node_split(node.sample_indices, self.context)
        node.split_info = split_info
        if split_info.gain < self.min_gain_to_split:
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    def split_next(self):
        """Split the node with highest potential gain.

        Return False if there are no remaining splittable nodes after this
        split.
        """
        if len(self.splittable_nodes) == 0:
            raise StopIteration("No more splittable nodes")

        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)

        sample_indices_left, sample_indices_right = split_indices(
            node.sample_indices, node.split_info, self.context)

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(depth, sample_indices_left,
                                   node.split_info.gradient_left,
                                   node.split_info.hessian_left)
        node.left_child = left_child_node
        right_child_node = TreeNode(depth, sample_indices_right,
                                    node.split_info.gradient_right,
                                    node.split_info.hessian_right)
        node.right_child = right_child_node
        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)

        elif (self.max_leaf_nodes is not None
                and n_leaf_nodes == self.max_leaf_nodes):
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()

        else:
            self._consider_splittability(left_child_node)
            self._consider_splittability(right_child_node)
        return left_child_node, right_child_node

    def can_split_further(self):
        return len(self.splittable_nodes) >= 1

    def _finalize_leaf(self, node):
        # TODO: store preduction score on the node
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)
