"""
This example illustrates how to display the tree of a single TreeGrower for
debugging purpose.
"""
from sklearn.datasets import make_classification
import numpy as np

from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower
from pygbm import plotting


rng = np.random.RandomState(0)

n_samples = int(1e7)
n_leaf_nodes = 5
X, y = make_classification(n_samples=n_samples, n_classes=2, n_features=5,
                           n_informative=3, n_redundant=0, random_state=rng)

bin_mapper_ = BinMapper(random_state=rng)
X_binned = bin_mapper_.fit_transform(X)

gradients = np.asarray(y, dtype=np.float32).copy()
hessians = np.ones(1, dtype=np.float32)

# First run to trigger the compilation of numba jit methods to avoid recording
# the compiler overhead in the profile report.
TreeGrower(X_binned, gradients, hessians, max_leaf_nodes=n_leaf_nodes).grow()

# New run with to collect timing statistics that will be included in the plot.
grower = TreeGrower(X_binned, gradients, hessians, max_leaf_nodes=n_leaf_nodes)
grower.grow()
plotting.plot_tree(grower)
