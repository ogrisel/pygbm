"""
This example illustrates how to display the tree of a single TreeGrower for
debugging purpose.
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np

from pygbm import GradientBoostingMachine
from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower 
from pygbm import plotting


rng = np.random.RandomState(0)

n_samples = int(1e6)
n_leaf_nodes = 10
X, y = make_classification(n_samples=n_samples, n_classes=2, n_features=5,
                           n_informative=5, n_redundant=0, random_state=rng)

bin_mapper_ = BinMapper(random_state=rng)
X_binned = bin_mapper_.fit_transform(X)

gradients = np.asarray(y, dtype=np.float32).copy()
hessians = np.ones(1, dtype=np.float32)

grower = TreeGrower(X_binned, gradients, hessians, max_leaf_nodes=n_leaf_nodes)
grower.grow()

plotting.plot_tree(grower)
