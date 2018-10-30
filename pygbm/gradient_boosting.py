import numpy as np
from time import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_random_state
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split

from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower, update_y_pred


class GradientBoostingMachine(BaseEstimator, RegressorMixin):

    def __init__(self, learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
                 max_depth=None, min_samples_leaf=20,
                 l2_regularization=0., max_bins=255,
                 max_no_improvement=5, validation_split=0.1,
                 scoring='neg_mean_squared_error',
                 tol=1e-7, verbose=0, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.max_no_improvement = max_no_improvement
        self.validation_split = validation_split
        self.scoring = scoring
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        fit_start_time = time()
        acc_find_split_time = 0.  # time spent finding the best splits
        acc_apply_split_time = 0.  # time spent splitting nodes
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.
        # TODO: add support for mixed-typed (numerical + categorical) data
        # TODO: add support for missing data
        # TODO: add support for pre-binned data (pass-through)?
        X, y = check_X_y(X, y, dtype=[np.float32, np.float32])
        rng = check_random_state(self.random_state)
        if self.verbose:
            print(f"Binning {X.nbytes / 1e9:.3f} GB of data: ", end="",
                  flush=True)
        tic = time()
        self.bin_mapper_ = BinMapper(max_bins=self.max_bins, random_state=rng)
        X_binned = self.bin_mapper_.fit_transform(X)
        toc = time()
        if self.verbose:
            duration = toc - tic
            troughput = X.nbytes / duration
            print(f"{duration:.3f} s ({troughput / 1e6:.3f} MB/s)")
        if self.validation_split is not None:
            X_binned_train, X_binned_val, y_train, y_val = train_test_split(
                X_binned, y, test_size=self.validation_split, stratify=y,
                random_state=rng)
            # Histogram computation is faster on feature-aligned data.
            X_binned_train = np.asfortranarray(X_binned_train)
        else:
            X_binned_train, y_train = X_binned, y
            X_binned_val, y_val = None, None

        # Subsample the training set for score-based monitoring.
        subsample_size = 10000
        if X_binned_train.shape[0] < subsample_size:
            X_binned_small_train = np.ascontiguousarray(X_binned_train)
            y_small_train = y_train
        else:
            indices = rng.choice(
                np.arange(X_binned_train.shape[0]), subsample_size)
            X_binned_small_train = X_binned_train[indices]
            y_small_train = y_train[indices]

        if self.verbose:
            print("Fitting gradient boosted rounds:")
        # TODO: plug custom loss functions
        y_pred = np.zeros_like(y_train, dtype=np.float32)
        gradients = np.asarray(y_train, dtype=np.float32).copy()
        hessians = np.ones(1, dtype=np.float32)
        self.predictors_ = predictors = []
        self.train_scores_ = []
        if self.validation_split is not None:
            self.validation_scores_ = []
        scorer = check_scoring(self, self.scoring)
        gb_start_time = time()
        # TODO: compute training loss and use it for early stopping if no
        # validation data is provided?
        self.n_iter_ = 0
        while True:
            should_stop = self._stopping_criterion(
                gb_start_time, scorer, X_binned_small_train, y_small_train,
                X_binned_val, y_val)
            if should_stop or self.n_iter_ == self.max_iter:
                break
            shrinkage = 1. if self.n_iter_ == 0 else self.learning_rate
            grower = TreeGrower(
                X_binned_train, gradients, hessians, n_bins=self.max_bins,
                max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                shrinkage=shrinkage)
            grower.grow()
            predictor = grower.make_predictor(
                bin_thresholds=self.bin_mapper_.bin_thresholds_)
            predictors.append(predictor)
            self.n_iter_ += 1
            tic_pred = time()
            update_y_pred(grower.finalized_leaves, y_pred)
            gradients = (y_train - y_pred).astype(np.float32)
            toc_pred = time()
            acc_prediction_time += toc_pred - tic_pred

            acc_apply_split_time += grower.total_apply_split_time
            acc_find_split_time += grower.total_find_split_time
        if self.verbose:
            duration = time() - fit_start_time
            n_leaf_nodes = sum(p.get_n_leaf_nodes() for p in self.predictors_)
            print(f"Fit {len(self.predictors_)} trees in {duration:.3f} s, "
                  f"({n_leaf_nodes} total leaf nodes)")
            print('{:<32} {:.3f}s'.format('Time spent finding best splits:',
                                          acc_find_split_time))
            print('{:<32} {:.3f}s'.format('Time spent applying splits:',
                                          acc_apply_split_time))
            print('{:<32} {:.3f}s'.format('Time spent predicting:',
                                          acc_prediction_time))
        self.train_scores_ = np.asarray(self.train_scores_)
        if self.validation_split is not None:
            self.validation_scores_ = np.asarray(self.validation_scores_)
        return self

    def predict(self, X):
        # TODO: check input / check_fitted
        # TODO: make predictor behave correctly on pre-binned data
        # TODO: handle classification and output class labels in this case
        predicted = np.zeros(X.shape[0], dtype=np.float32)
        for predictor in self.predictors_:
            predicted += predictor.predict(X)
        return predicted

    def _predict_binned(self, X_binned):
        predicted = np.zeros(X_binned.shape[0], dtype=np.float32)
        for predictor in self.predictors_:
            predicted += predictor.predict_binned(X_binned)
        return predicted

    def _stopping_criterion(self, start_time, scorer, X_binned_train, y_train,
                            X_binned_val, y_val):
        log_msg = f"[{self.n_iter_}/{self.max_iter}]"

        if self.scoring is not None:
            # TODO: make sure that self.predict can work on binned data and
            # then only use the public scorer.__call__.
            predicted_train = self._predict_binned(X_binned_train)
            score_train = scorer._score_func(y_train, predicted_train)
            self.train_scores_.append(score_train)
            log_msg += f" {self.scoring} train: {score_train:.5f},"

            if self.validation_split is not None:
                predicted_val = self._predict_binned(X_binned_val)
                score_val = scorer._score_func(y_val, predicted_val)
                self.validation_scores_.append(score_val)
                log_msg += f", {self.scoring} val: {score_val:.5f},"

        if self.n_iter_ > 0:
            iteration_time = (time() - start_time) / self.n_iter_
            predictor_nodes = self.predictors_[-1].nodes
            max_depth = predictor_nodes['depth'].max()
            n_leaf_nodes = predictor_nodes['is_leaf'].sum()
            log_msg += (f" {n_leaf_nodes} leaf nodes, max depth {max_depth}"
                        f" in {iteration_time:0.3f}s")

        if self.verbose:
            print(log_msg)

        if self.validation_split is not None:
            return self._should_stop(self.validation_scores_)
        else:
            return self._should_stop(self.train_scores_)

    def _should_stop(self, scores):
        if (len(scores) == 0 or
                (self.max_no_improvement
                 and len(scores) < self.max_no_improvement)):
            return False
        context_scores = scores[-self.max_no_improvement:]
        candidate = scores[-self.max_no_improvement]
        tol = 0. if self.tol is None else self.tol
        # sklearn scores: higher is always better.
        best_with_tol = max(context_scores) * (1 - tol)
        return candidate >= best_with_tol


# TODO: shall we split between GBMClassifier and GBMRegressor instead
# of using a single class?
