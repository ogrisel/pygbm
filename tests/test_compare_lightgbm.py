from sklearn.model_selection import train_test_split
import numpy as np
import pytest

from pygbm import GradientBoostingMachine
from pygbm.binning import BinMapper


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('n_samples, max_leaf_nodes', [
    (255, 4096),
    (1000, 8),
])
def test_same_predictions_easy_target(seed, n_samples, max_leaf_nodes):
    # Make sure pygbm has the same predictions as LGBM for very easy targets.
    #
    # In particular when the size of the trees are bound and the number of
    # samples is large enough, the structure of the prediction trees found by
    # LightGBM and PyGBM should be exactly identical.
    #
    # Notes:
    # - Several candidate splits may have equal gains when the number of
    #   samples in a node is low (and because of float errors). Therefore the
    #   predictions on the test set might differ if the structure of the tree
    #   is not exactly the same. To avoid this issue we only compare the
    #   predictions on the test set when the number of samples is large enough
    #   and max_leaf_nodes is low enough.
    # - To ignore  discrepancies caused by small differences the binning
    #   strategy, data is pre-binned if n_samples > 255.

    lb = pytest.importorskip("lightgbm")

    rng = np.random.RandomState(seed=seed)
    n_samples = n_samples
    min_samples_leaf = 1  # XXX: changing this breaks the test
    max_iter = 1

    # data = linear target, 5 features, 3 irrelevant.
    X = rng.normal(size=(n_samples, 5))
    y = X[:, 0] - X[:, 1]
    if n_samples > 255:
        X = BinMapper().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

    est_lightgbm = lb.LGBMRegressor(n_estimators=max_iter,
                                    min_data_in_bin=1,
                                    learning_rate=1,
                                    min_data_in_leaf=min_samples_leaf,
                                    num_leaves=max_leaf_nodes)
    est_pygbm = GradientBoostingMachine(max_iter=max_iter,
                                        learning_rate=1,
                                        validation_split=None, scoring=None,
                                        min_samples_leaf=min_samples_leaf,
                                        max_leaf_nodes=max_leaf_nodes)

    est_lightgbm.fit(X_train, y_train)
    est_pygbm.fit(X_train, y_train)

    pred_lgbm = est_lightgbm.predict(X_train)
    pred_pygbm = est_pygbm.predict(X_train)
    np.testing.assert_array_almost_equal(pred_lgbm, pred_pygbm, decimal=3)

    if max_leaf_nodes < 10 and n_samples > 1000:
        pred_lgbm = est_lightgbm.predict(X_test)
        pred_pygbm = est_pygbm.predict(X_test)
        np.testing.assert_array_almost_equal(pred_lgbm, pred_pygbm, decimal=3)
