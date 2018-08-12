import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import approx

from pygbm.grower import TreeGrower


def _make_training_data(n_bins=256, constant_hessian=True):
    rng = np.random.RandomState(42)
    n_samples = 10000

    # Generate some test data directly binned so as to test the grower code
    # independently of the binning logic.
    features_data = rng.randint(0, n_bins - 1, size=(n_samples, 2),
                                dtype=np.uint8)
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
    if constant_hessian:
        all_hessians = np.ones(shape=1, dtype=np.float32)
    else:
        all_hessians = np.ones_like(all_gradients)
    return features_data, all_gradients, all_hessians


def _check_children_consistency(parent, left, right):
    assert parent.left_child is left
    assert parent.right_child is right

    # each sample from the parent is propagated to one of the two children
    assert (len(left.sample_indices) + len(right.sample_indices)
            == len(parent.sample_indices))

    assert (set(left.sample_indices).union(set(right.sample_indices))
            == set(parent.sample_indices))

    # samples are sent either to the left or the right node, never to both
    assert (set(left.sample_indices).intersection(set(right.sample_indices))
            == set())


@pytest.mark.parametrize(
    'n_bins, constant_hessian, stopping_param, shrinkage',
    [
        (11, True, "min_gain_to_split", 0.5),
        (11, False, "min_gain_to_split", 1.),
        (11, True, "max_leaf_nodes", 1.),
        (11, False, "max_leaf_nodes", 0.1),
        (42, True, "max_leaf_nodes", 0.01),
        (42, False, "max_leaf_nodes", 1.),
        (256, True, "min_gain_to_split", 1.),
        (256, True, "max_leaf_nodes", 0.1),
    ]
)
def test_grow_tree(n_bins, constant_hessian, stopping_param, shrinkage):
    features_data, all_gradients, all_hessians = _make_training_data(
        n_bins=n_bins, constant_hessian=constant_hessian)
    n_samples = features_data.shape[0]

    if stopping_param == "max_leaf_nodes":
        stopping_param = {"max_leaf_nodes": 3}
    else:
        stopping_param = {"min_gain_to_split": 0.01}

    grower = TreeGrower(features_data, all_gradients, all_hessians,
                        n_bins=n_bins, shrinkage=shrinkage, **stopping_param)

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

    # All training samples have ben splitted in the two nodes, approximately
    # 50%/50%
    _check_children_consistency(grower.root, left_node, right_node)
    assert len(left_node.sample_indices) > 0.4 * n_samples
    assert len(left_node.sample_indices) < 0.6 * n_samples

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
    _check_children_consistency(right_node, right_left_node, right_right_node)
    assert len(right_left_node.sample_indices) > 0.1 * n_samples
    assert len(right_left_node.sample_indices) < 0.2 * n_samples

    assert len(right_right_node.sample_indices) > 0.2 * n_samples
    assert len(right_right_node.sample_indices) < 0.4 * n_samples

    # All the leafs are pure, it is not possible to split any further:
    assert not grower.can_split_further()

    # Check the values of the leaves:
    assert grower.root.left_child.weight == approx(-shrinkage)
    assert grower.root.right_child.left_child.weight == approx(-shrinkage)
    assert grower.root.right_child.right_child.weight == approx(shrinkage)


def test_predictor_from_grower():
    # Build a tree on the toy 3-leaf dataset to extract the predictor.
    n_bins = 256
    features_data, all_gradients, all_hessians = _make_training_data(
        n_bins=n_bins)
    grower = TreeGrower(features_data, all_gradients, all_hessians,
                        n_bins=n_bins, shrinkage=1., max_leaf_nodes=3)
    grower.grow()
    assert grower.n_nodes == 5  # (2 decision nodes + 3 leaves)

    # Check that the node structure can be converted into a predictor
    # object to perform predictions at scale
    predictor = grower.make_predictor()
    assert predictor.nodes.shape[0] == 5
    assert predictor.nodes['is_leaf'].sum() == 3

    def predict(features):
        return predictor.predict_one_binned(np.array(features, dtype=np.uint8))

    # Probe some predictions for each leaf of the tree
    assert predict([0, 0]) == approx(-1)
    assert predict([42, 99]) == approx(-1)
    assert predict([128, 255]) == approx(-1)

    assert predict([129, 0]) == approx(-1)
    assert predict([129, 85]) == approx(-1)
    assert predict([255, 85]) == approx(-1)

    assert predict([129, 86]) == approx(1)
    assert predict([129, 255]) == approx(1)
    assert predict([242, 100]) == approx(1)

    # Check that training set can be recovered exactly:
    predictions = predictor.predict_binned(features_data)
    assert_allclose(predictions, all_gradients)
