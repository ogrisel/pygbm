from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import Memory
from pygbm import GradientBoostingMachine
# from lightgbm import LGBMClassifier
# for now as pygbm does not have classifier loss yet:
from lightgbm import LGBMRegressor

HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')
n_leaf_nodes = 31
n_trees = 10
# subsample = 25000
subsample = None


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

print("Fitting a LightGBM model...")
tic = time()
lightgbm_model = LGBMRegressor(n_estimators=n_trees, num_leaves=n_leaf_nodes,
                               learning_rate=1., verbose=10)
lightgbm_model.fit(data_train, target_train)
toc = time()
predicted_test = lightgbm_model.predict(data_test)
roc_auc = roc_auc_score(target_test, predicted_test)
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}")

# print(lightgbm_model._Booster._save_model_to_string())

print("JIT compiling code for the pygbm model...")
tic = time()
pygbm_model = GradientBoostingMachine(learning_rate=0.1, max_iter=1,
                                      max_bins=255,
                                      max_leaf_nodes=n_leaf_nodes,
                                      random_state=0, scoring=None,
                                      verbose=0, validation_split=None)
pygbm_model.fit(data_train[:100], target_train[:100])
toc = time()
print(f"done in {toc - tic:.3f}s")


print("Fitting a pygbm model...")
tic = time()
pygbm_model = GradientBoostingMachine(learning_rate=1, max_iter=n_trees,
                                      max_bins=255,
                                      max_leaf_nodes=n_leaf_nodes,
                                      random_state=0, scoring=None,
                                      verbose=1, validation_split=None)
pygbm_model.fit(data_train, target_train)
toc = time()
predicted_test = pygbm_model.predict(data_test)
roc_auc = roc_auc_score(target_test, predicted_test)
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}")

# for predictor in pygbm_model.predictors_:
#     print(predictor.nodes)