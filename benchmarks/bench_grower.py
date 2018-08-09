from time import time
import numpy as np
from joblib import Memory
from pygbm.grower import TreeGrower


m = Memory(location='/tmp', mmap_mode='r')


@m.cache
def make_data(n_bins=256, n_samples=int(1e6), n_features=100,
              loss_dtype=np.float32, seed=42):
    rng = np.random.RandomState(seed)

    gradients = rng.randn(n_samples).astype(loss_dtype)

    binned_features = rng.randint(0, n_bins - 1, dtype=np.uint8,
                                  size=(n_features, n_samples)).T
    assert binned_features.shape == (n_samples, n_features)
    assert binned_features.flags.f_contiguous
    return binned_features, gradients


n_bins = 256
n_samples, n_features = int(1e7), 200
binned_features, gradients = make_data(
    n_bins=n_bins, n_samples=n_samples, n_features=n_features)
hessians = np.ones(shape=1, dtype=gradients.dtype)

print(f"Growing one tree on {binned_features.nbytes / 1e9:0.1f} GB of "
      f"random data ({n_samples:.0e} samples, {n_features} features).")
print("Finding the best split on the root node...")
tree_start = tic = time()
grower = TreeGrower(binned_features, gradients, hessians, n_bins=n_bins,
                    max_leaf_nodes=255)
toc = time()
print(f"done in {toc - tic:0.3f}s")

while grower.can_split_further():
    print("Splitting next node...")
    tic = time()
    left, right = grower.split_next()
    toc = time()
    print("left node: ", left)
    print("right node: ", right)
    print(f"done in {toc - tic:0.3f}s")

print(f"{len(grower.finalized_leaves)} leaves in {time() - tree_start:0.3f}s")
