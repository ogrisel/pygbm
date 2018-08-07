from time import time
import numpy as np
from joblib import Memory
from pygbm.grower import TreeGrower


m = Memory(location='/tmp')


@m.cache
def make_data(n_bins=256, n_samples=int(1e6), n_features=100,
              loss_dtype=np.float32, seed=42):
    rng = np.random.RandomState(seed)

    gradients = rng.randn(n_samples).astype(loss_dtype)
    hessians = np.ones(n_samples, dtype=loss_dtype)

    binned_features = rng.randint(0, n_bins - 1, dtype=np.uint8,
                                  size=(n_features, n_samples)).T
    assert binned_features.shape == (n_samples, n_features)
    assert binned_features.flags.f_contiguous
    return binned_features, gradients, hessians


n_bins = 256
n_samples, n_features = int(1e8), 10
binned_features, gradients, hessians = make_data(
    n_bins=n_bins, n_samples=int(1e7), n_features=200)

print(f"Growing one tree on {binned_features.nbytes / 1e9:0.1f} GB of "
      f"random data ({n_samples:.0e} samples, {n_features} features).")
print("Finding the best split on the root node...")
tic = time()
grower = TreeGrower(binned_features, gradients, hessians, n_bins=n_bins,
                    max_leaf_nodes=5)
toc = time()
print(f"done in {toc - tic:0.3f}s")

while grower.can_split_further():
    print("Splitting next node...")
    tic = time()
    grower.split_next()
    toc = time()
    print(f"done in {toc - tic:0.3f}s")
