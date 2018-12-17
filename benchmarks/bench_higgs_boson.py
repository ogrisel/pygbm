from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Memory
from pygbm import GradientBoostingClassifier
from pygbm.utils import get_lightgbm_estimator
import numba


parser = argparse.ArgumentParser()
parser.add_argument('--n-leaf-nodes', type=int, default=31)
parser.add_argument('--n-trees', type=int, default=10)
parser.add_argument('--no-lightgbm', action="store_true", default=False)
parser.add_argument('--learning-rate', type=float, default=1.)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--max-bins', type=int, default=255)
args = parser.parse_args()

HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')

n_leaf_nodes = args.n_leaf_nodes
n_trees = args.n_trees
subsample = args.subsample
lr = args.learning_rate
max_bins = args.max_bins


@m.cache
def load_data():
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
    return df


df = load_data()
target = df.values[:, 0]
data = np.ascontiguousarray(df.values[:, 1:])
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=50000, random_state=0)

if subsample is not None:
    data_train, target_train = data_train[:subsample], target_train[:subsample]

n_samples, n_features = data_train.shape
print(f"Training set with {n_samples} records with {n_features} features.")

print("JIT compiling code for the pygbm model...")
tic = time()
pygbm_model = GradientBoostingClassifier(learning_rate=lr, max_iter=1,
                                         max_bins=max_bins,
                                         max_leaf_nodes=n_leaf_nodes,
                                         n_iter_no_change=None,
                                         random_state=0,
                                         verbose=0)
pygbm_model.fit(data_train[:100], target_train[:100])
pygbm_model.predict(data_train[:100])  # prediction code is also jitted
toc = time()
print(f"done in {toc - tic:.3f}s")

print("Fitting a pygbm model...")
tic = time()
pygbm_model = GradientBoostingClassifier(loss='binary_crossentropy',
                                         learning_rate=lr, max_iter=n_trees,
                                         max_bins=max_bins,
                                         max_leaf_nodes=n_leaf_nodes,
                                         n_iter_no_change=None,
                                         random_state=0,
                                         verbose=1)
pygbm_model.fit(data_train, target_train)
toc = time()
predicted_test = pygbm_model.predict(data_test)
roc_auc = roc_auc_score(target_test, predicted_test)
acc = accuracy_score(target_test, predicted_test)
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if hasattr(numba, 'threading_layer'):
    print("Threading layer chosen: %s" % numba.threading_layer())

if not args.no_lightgbm:
    print("Fitting a LightGBM model...")
    tic = time()
    lightgbm_model = get_lightgbm_estimator(pygbm_model)
    lightgbm_model.fit(data_train, target_train)
    toc = time()
    predicted_test = lightgbm_model.predict(data_test)
    roc_auc = roc_auc_score(target_test, predicted_test)
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
