"""
This file should be removed before merging... It's just a basic sanity check
to make sure I don't screw up everything when making a change.
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np
from pygbm import GradientBoostingMachine


def test_preds_are_as_expected():
    n_leaf_nodes = 31
    n_trees = 10
    lr = 1.
    max_bins = 255
    X, y = make_classification(n_samples=1000000, n_classes=2, n_features=5,
                               n_informative=5, n_redundant=0, random_state=0)

    data_train, data_test, target_train, target_test = train_test_split(
        X, y, test_size=.2, random_state=0)

    pygbm_model = GradientBoostingMachine(
        learning_rate=lr, max_iter=n_trees, max_bins=max_bins,
        max_leaf_nodes=n_leaf_nodes, random_state=0, scoring=None, verbose=1,
        validation_split=None)

    pygbm_model.fit(data_train, target_train)
    predicted_test = pygbm_model.predict(data_test)
    roc_auc = roc_auc_score(target_test, predicted_test)

    try:
        from pygbm import plotting
        # plotting.plot_tree(pygbm_model, tree_index=5, view=False)
    except:
        pass

    assert np.allclose(roc_auc, 0.9809811751028725)

#test_preds_are_as_expected()