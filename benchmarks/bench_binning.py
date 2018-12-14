from time import time
import numpy as np
from joblib import Memory
from pygbm.binning import _find_binning_thresholds, _map_to_bins


m = Memory(location='/tmp')


@m.cache
def make_data(n_samples=int(1e6), n_features=5, seed=42, dtype=np.float32):
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features).astype(dtype)


print("Generating random data...")
data = make_data(n_samples=int(1e6), n_features=5, seed=42, dtype=np.float32)
print("Extracting bins from subsample of data...")
bins = _find_binning_thresholds(data, random_state=0)

print("Compiling map_to_bins...")
tic = time()
binned = _map_to_bins(np.asfortranarray(data[:5]), bins)
toc = time()
duration = toc - tic
print(f"done in {duration:0.3f}s")

print("Mapping data to integer bins...")
tic = time()
binned = _map_to_bins(data, bins)
toc = time()
duration = toc - tic
print(f"Processed {data.nbytes/1e9:0.3f} GB in {duration:0.3f}s"
      f" ({data.nbytes / 1e6 / duration:0.1f} MB/s)")
print(f"Output size: {binned.nbytes / 1e9:0.3f} GB")
