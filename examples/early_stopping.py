"""This example illustrates early-stopping to avoid overfitting.

Early stopping is performed on some held-out validation data, or on the
training data if validation_split is None.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pygbm import GradientBoostingClassifier

rng = np.random.RandomState(0)

n_samples = int(1e6)
X, y = make_classification(n_samples, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

print('Early stopping on held-out validation data')
clf = GradientBoostingClassifier(max_iter=100,
                                 scoring='neg_log_loss',
                                 validation_split=.1,
                                 n_iter_no_change=5,
                                 tol=1e-4,
                                 verbose=1,
                                 random_state=rng)
clf.fit(X_train, y_train)
print(f'Early stopped at iteration {clf.n_iter_}')
print(f'Mean accuracy: {clf.score(X_test, y_test)}')

print('Early stopping on training data')
clf = GradientBoostingClassifier(max_iter=100,
                                 scoring='neg_log_loss',
                                 validation_split=None,
                                 n_iter_no_change=5,
                                 tol=1e-4,
                                 verbose=1,
                                 random_state=rng)
clf.fit(X_train, y_train)
print(f'Early stopped at iteration {clf.n_iter_}')
print(f'Mean accuracy: {clf.score(X_test, y_test)}')
