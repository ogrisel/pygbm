from time import time

import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from pygbm.binning import BinMapper
from pygbm import GradientBoostingRegressor

n_samples = int(5e6)

X, y = make_regression(n_samples=n_samples, n_features=5)
est = GradientBoostingRegressor(max_iter=1, n_iter_no_change=None,
                                random_state=0)
est.fit(X, y)
predictor = est.predictors_[0][0]

bin_mapper = BinMapper(random_state=0)
X_binned = bin_mapper.fit_transform(X)

X_binned_c = np.ascontiguousarray(X_binned)
print("Compiling predictor code...")
tic = time()
predictor.predict_binned(np.asfortranarray(X_binned[:100]))
predictor.predict_binned(X_binned_c[:100])
predictor.predict(np.asfortranarray(X[:100]))
predictor.predict(X[:100])
toc = time()
print(f"done in {toc - tic:0.3f}s")

data_size = X_binned.nbytes
print("Computing predictions (F-contiguous binned data)...")
tic = time()
scores_binned_f = predictor.predict_binned(X_binned)
toc = time()
duration = toc - tic
speed = n_samples / duration
print(f"done in {duration:.4f}s    {data_size / duration / 1e6:.3f} MB/s    "
      f"{speed:.3e} sample/s")

print("Computing predictions (C-contiguous binned data)...")
tic = time()
scores_binned_c = predictor.predict_binned(X_binned_c)
toc = time()
duration = toc - tic
speed = n_samples / duration
print(f"done in {duration:.4f}s    {data_size / duration / 1e6:.3f} MB/s    "
      f"{speed:.3e} sample/s")

assert_allclose(scores_binned_f, scores_binned_c)

data_size = X.nbytes
print("Computing predictions (F-contiguous numerical data)...")
X_f = np.asfortranarray(X)
tic = time()
scores_f = predictor.predict(X_f)
toc = time()
duration = toc - tic
speed = n_samples / duration
print(f"done in {duration:.4f}s    {data_size / duration / 1e6:.3f} MB/s    "
      f"{speed:.3e} sample/s")

assert_allclose(scores_binned_f, scores_f)

print("Computing predictions (C-contiguous numerical data)...")
data_size = X.nbytes
tic = time()
scores_c = predictor.predict(X)
toc = time()
duration = toc - tic
speed = n_samples / duration
print(f"done in {duration:.4f}s    {data_size / duration / 1e6:.3f} MB/s    "
      f"{speed:.3e} sample/s")

assert_allclose(scores_binned_f, scores_c)
