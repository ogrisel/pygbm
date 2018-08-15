from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import numpy as np
import pandas as pd
from joblib import Memory
from pygbm.gradient_boosting import GradientBoostingMachine


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
    return df


n_bins = 256
df = load_data(n_bins)
target = df.values[:, 0]
data = np.ascontiguousarray(df.values[:, 1:])

n_samples, n_features = data.shape
gradients = target
hessians = np.ones(1, dtype=np.float32)

model = GradientBoostingMachine(learning_rate=0.5, max_iter=100, n_bins=n_bins,
                                random_state=42, scoring='roc_auc', verbose=1)
model.fit(data, target)
