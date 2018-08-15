from time import time
import numpy as np
from numpy.testing import assert_allclose
from pygbm.grower import TreeGrower


# TODO: make some random numerical data, bin it, and fit a grower with many
# leaves in a joblib'ed function.


predictor = grower.make_predictor(bin_thresholds=bin_thresholds)
binned_features_c = np.ascontiguousarray(binned_features)
print("Compiling predictor code...")
tic = time()
predictor.predict_binned(np.asfortranarray(binned_features[:10]))
predictor.predict_binned(binned_features_c[:10])
predictor.predict(data[:10])
toc = time()
print(f"done in {toc - tic:0.3f}s")

data_size = binned_features.nbytes
print("Computing predictions (F-contiguous binned data)...")
tic = time()
scores_binned_f = predictor.predict_binned(binned_features)
toc = time()
duration = toc - tic
print(f"done in {duration:.3f}s ({data_size / duration / 1e9:.3} GB/s)")

print("Computing predictions (C-contiguous binned data)...")
tic = time()
scores_binned_c = predictor.predict_binned(binned_features_c)
toc = time()
duration = toc - tic
print(f"done in {duration:.3f}s ({data_size / duration / 1e9:.3} GB/s)")

assert_allclose(scores_binned_f, scores_binned_f)

print("Computing predictions (C-contiguous numerical data)...")
data_size = data.nbytes
tic = time()
scores_data = predictor.predict(data)
toc = time()
duration = toc - tic
print(f"done in {duration:.3f}s ({data_size / duration / 1e9:.3} GB/s)")

assert_allclose(scores_data, scores_binned_c)
