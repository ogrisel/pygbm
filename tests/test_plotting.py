import numpy as np
from sklearn.datasets import make_classification
import pytest
from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower
from pygbm import GradientBoostingRegressor

X, y = make_classification(n_samples=150, n_classes=2, n_features=5,
                           n_informative=3, n_redundant=0,
                           random_state=0)


def test_plot_grower(tmpdir):
    pytest.importorskip('graphviz')
    from pygbm.plotting import plot_tree

    X_binned = BinMapper().fit_transform(X)
    gradients = np.asarray(y, dtype=np.float32).copy()
    hessians = np.ones(1, dtype=np.float32)
    grower = TreeGrower(X_binned, gradients, hessians, max_leaf_nodes=5)
    grower.grow()
    filename = tmpdir.join('plot_grower.pdf')
    plot_tree(grower, view=False, filename=filename)
    assert filename.exists()


def test_plot_estimator(tmpdir):
    pytest.importorskip('graphviz')
    from pygbm.plotting import plot_tree

    n_trees = 3
    est = GradientBoostingRegressor(max_iter=n_trees)
    est.fit(X, y)
    for i in range(n_trees):
        filename = tmpdir.join('plot_predictor.pdf')
        plot_tree(est, tree_index=i, view=False, filename=filename)
        assert filename.exists()


def test_plot_estimator_and_lightgbm(tmpdir):
    pytest.importorskip('graphviz')
    lightgbm = pytest.importorskip('lightgbm')
    from pygbm.plotting import plot_tree

    n_trees = 3
    est_pygbm = GradientBoostingRegressor(max_iter=n_trees)
    est_pygbm.fit(X, y)
    est_lightgbm = lightgbm.LGBMRegressor(n_estimators=n_trees)
    est_lightgbm.fit(X, y)

    for i in range(n_trees):
        filename = tmpdir.join('plot_mixed_predictors.pdf')
        plot_tree(est_pygbm, est_lightgbm=est_lightgbm, tree_index=i,
                  view=False, filename=filename)
        assert filename.exists()
