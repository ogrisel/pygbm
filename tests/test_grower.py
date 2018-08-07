import numpy as np
import pytest

from pygbm.grower import TreeGrower


@pytest.mark.parametrize('n_bins', [11, 42, 256])
def test_grow_tree(n_bins):
    rng = np.random.RandomState(42)

    # Generate some test data directly binned so as to test the grower code
    # independently of the binning logic.
    features_data = rng.randint(0, n_bins - 1, size=(10000, 2), dtype=np.uint8)
    features_data = np.asfortranarray(features_data)

    def true_decision_function(input_features):
        """Ground truth decision function

        This is a very simple yet asymmetric decision tree. Therefore the
        grower code should have no trouble recovering the decision function
        from 10000 training samples.
        """
        if input_features[0] <= n_bins // 2:
            return -1
        else:
            if input_features[1] <= n_bins // 3:
                return -1
            else:
                return 1

    target = np.array([true_decision_function(x) for x in features_data],
                      dtype=np.float32)

    # Assume a square loss applied to an initial model that always predicts 0
    # (hardcoded for this test):
    all_gradients = target
    all_hessians = np.ones_like(all_gradients)

    for stopping_param in [
                # {'min_gain_to_split': 0.001},
                {'max_leaf_nodes': 3},  # FIXME broken!
            ]:
        grower = TreeGrower(features_data, all_gradients, all_hessians,
                            n_bins=n_bins, **stopping_param)

        # The root node is not yet splitted, but the best possible split has
        # already been evaluated:
        assert grower.root.left_child is None
        assert grower.root.right_child is None

        root_split = grower.root.split_info
        assert root_split.feature_idx == 0
        assert root_split.bin_idx == n_bins // 2
        assert len(grower.splittable_nodes) == 1

        # Calling split next applies the next split and computes the best split
        # for each of the two newly introduced children nodes.
        assert grower.can_split_further()
        left_node, right_node = grower.split_next()
        assert grower.root.left_child is left_node
        assert grower.root.right_child is right_node

        if grower.min_gain_to_split > 0:
            # The left node is too pure: there is no gain to split it further.
            assert left_node.split_info.gain < grower.min_gain_to_split
            assert left_node in grower.finalized_leaves

        # The right node can still be splitted further, this time on feature #1
        split_info = right_node.split_info
        assert split_info.gain > 1.
        assert split_info.feature_idx == 1
        assert split_info.bin_idx == n_bins // 3
        assert right_node.left_child is None
        assert right_node.right_child is None

        # The right split has not been applied yet. Let's do it now:
        assert grower.can_split_further()
        right_left_node, right_right_node = grower.split_next()
        assert right_left_node is right_node.left_child
        assert right_right_node is right_node.right_child

        # All the leafs are pure, it is not possible to split any further:
        assert not grower.can_split_further()

        # TODO: check predicted values on all leafs
