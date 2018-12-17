"""This example illustrates the use of scikit-learn's GridSearchCV.

The grid search is used to determine the best learning rate."""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from pygbm import GradientBoostingRegressor

rng = np.random.RandomState(0)

n_samples = int(1e6)
X, y = make_regression(n_samples, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

clf = GradientBoostingRegressor(max_iter=10,
                                n_iter_no_change=None,
                                verbose=1,
                                random_state=rng)
param_grid = {'learning_rate': [1, .1, .01, .001]}
cv = KFold(n_splits=3, random_state=rng)
gs = GridSearchCV(clf, param_grid=param_grid, cv=cv)
gs.fit(X_train, y_train)

print(f'Best param: {gs.best_params_}')
print(f'R2 coefficient: {gs.score(X_test, y_test)}')
