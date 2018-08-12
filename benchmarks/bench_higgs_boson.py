from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import numpy as np
import pandas as pd
from joblib import Memory
from pygbm.binning import find_bins, map_to_bins
from pygbm.grower import TreeGrower


HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')


@m.cache
def load_data(n_bins):
    filename = os.path.join(HERE, URL.rsplit('/', 1)[-1])
    if not os.path.exists(filename):
        print(f"Downloading {URL} to {filename} (2.6 GB)...")
        urlretrieve(URL, filename)
        print("done.")

    print(f"Parsing {filename}...")
    tic = time()
    with GzipFile(filename) as f:
        df = pd.read_csv(f, header=None, dtype=np.float32)
    toc = time()
    print(f"Loaded {df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")

    target = df.values[:, 0]
    data = np.ascontiguousarray(df.values[:, 1:])  # TODO: lift requirement
    print("Binning features...")
    tic = time()
    bin_thresholds = find_bins(data, n_bins=n_bins)
    binned_features = map_to_bins(data, bin_thresholds)
    toc = time()
    print(f"Binned {data.nbytes / 1e9:.3f} GB in {toc - tic:0.3f}s")
    print(f"Resulting data size: {binned_features.nbytes / 1e9} GB")
    return df, binned_features, target


n_bins = 256
df, binned_features, target = load_data(n_bins)
n_samples, n_features = binned_features.shape
gradients = target
hessians = np.ones(1, dtype=np.float32)

print("Compiling grower code...")
tic = time()
TreeGrower(np.asfortranarray(binned_features[:5]), gradients[:5], hessians[:5],
           n_bins=n_bins, max_leaf_nodes=3).grow()
toc = time()
print(f"done in {toc - tic:0.3f}s")

print(f"Growing one tree on {binned_features.nbytes / 1e9:0.1f} GB of "
      f"binned data ({n_samples:.0e} samples, {n_features} features).")
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


predictor = grower.make_predictor()
binned_features_c = np.ascontiguousarray(binned_features)
print("Compiling predictor code...")
tic = time()
predictor.predict_binned(np.asfortranarray(binned_features[:10]))
predictor.predict_binned(binned_features_c[:10])
toc = time()
print(f"done in {toc - tic:0.3f}s")

print("Computing predictions (F-contiguous binned data)...")
tic = time()
scores = predictor.predict_binned(binned_features)
toc = time()
print(f"done in {toc - tic:0.3f}s")

print("Computing predictions (C-contiguous binned data)...")
tic = time()
scores_c = predictor.predict_binned(binned_features_c)
toc = time()
print(f"done in {toc - tic:0.3f}s")
