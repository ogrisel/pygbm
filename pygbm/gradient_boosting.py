from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange
from time import time
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_X_y, check_random_state
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split

from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower
from pygbm.loss import _LOSSES


class BaseGradientBoostingMachine(BaseEstimator, ABC):

    @abstractmethod
    def __init__(self, loss, learning_rate, max_iter, max_leaf_nodes,
                 max_depth, min_samples_leaf, l2_regularization, max_bins,
                 max_no_improvement, validation_split, scoring, tol, verbose,
                 random_state):
        self.loss = loss
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

    def _validate_parameters(self):

        if self.loss not in _LOSSES:
            raise ValueError("Invalid loss {}. Accepted losses are {}.".format(
                self.loss, ', '.join(self._VALID_LOSSES)))
        if self.loss not in self._VALID_LOSSES:
            raise ValueError(
                "Loss {} is not supported for {}. Accepted losses"
                "are {}.".format(self.loss, self.__class__.__name__,
                                 ', '.join(self._VALID_LOSSES)))
        self.loss_ = _LOSSES[self.loss]()

        if self.learning_rate <= 0:
            raise ValueError(f'learning_rate={self.learning_rate} must '
                             f'be strictly positive')
        if self.max_iter < 1:
            raise ValueError(f'max_iter={self.max_iter} must '
                             f'not be smaller than 1.')
        if self.max_no_improvement < 1:
            raise ValueError(f'max_no_improvement={self.max_no_improvement} '
                             f'must not be smaller than 1.')
        if self.validation_split is not None and self.validation_split <= 0:
            raise ValueError(f'validation_split={self.validation_split} '
                             f'must be strictly positive, or None.')
        if self.tol <= 0:
            raise ValueError(f'tol={self.tol} '
                             f'must be strictly positive.')

    def fit(self, X, y):

        fit_start_time = time()
        acc_find_split_time = 0.  # time spent finding the best splits
        acc_apply_split_time = 0.  # time spent splitting nodes
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.
        # TODO: add support for mixed-typed (numerical + categorical) data
        # TODO: add support for missing data
        # TODO: add support for pre-binned data (pass-through)?
        # TODO: test input checking
        X, y = check_X_y(X, y, dtype=[np.float32, np.float64])
        y = y.astype(np.float32, copy=False)
        rng = check_random_state(self.random_state)

        self._validate_parameters()

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
            # stratify for classification
            stratify = y if hasattr(self.loss_, 'predict_proba') else None

            X_binned_train, X_binned_val, y_train, y_val = train_test_split(
                X_binned, y, test_size=self.validation_split,
                stratify=stratify, random_state=rng)
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

        # TODO: is the initial prediction always 0? What about classif?
        y_pred = np.zeros_like(y_train)
        gradients, hessians = self.loss_.init_gradients_and_hessians(
            n_samples=y_train.shape[0])
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
            # Update gradients and hessians inplace
            self.loss_.update_gradients_and_hessians(gradients, hessians,
                                                     y_train, y_pred)
            grower = TreeGrower(
                X_binned_train, gradients, hessians, max_bins=self.max_bins,
                n_bins_per_feature=self.bin_mapper_.n_bins_per_feature_,
                max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                shrinkage=shrinkage)
            grower.grow()
            predictor = grower.make_predictor(
                bin_thresholds=self.bin_mapper_.bin_thresholds_)
            predictors.append(predictor)
            self.n_iter_ += 1
            tic_pred = time()
            # prepare leaves_data so that _update_y_pred can be @njitted
            leaves_data = [(l.value, l.sample_indices)
                           for l in grower.finalized_leaves]
            _update_y_pred(leaves_data, y_pred)
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

    def _raw_predict(self, X):
        """Return the sum of the leaves values"""
        # TODO: check input / check_fitted
        # TODO: make predictor behave correctly on pre-binned data
        raw_predictions = np.zeros(X.shape[0], dtype=np.float32)
        for predictor in self.predictors_:
            raw_predictions += predictor.predict(X)

        return raw_predictions

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


class GradientBoostingRegressor(BaseGradientBoostingMachine, RegressorMixin):

    _VALID_LOSSES = ('least_squares',)

    def __init__(self, loss='least_squares', learning_rate=0.1, max_iter=100,
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
                 l2_regularization=0., max_bins=256, max_no_improvement=5,
                 validation_split=0.1, scoring='neg_mean_squared_error',
                 tol=1e-7, verbose=0, random_state=None):
        super(GradientBoostingRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_bins=max_bins,
            max_no_improvement=max_no_improvement,
            validation_split=validation_split, scoring=scoring, tol=tol,
            verbose=verbose, random_state=random_state)

    def predict(self, X):
        return self._raw_predict(X)


class GradientBoostingClassifier(BaseGradientBoostingMachine, ClassifierMixin):

    _VALID_LOSSES = ('binary_crossentropy',)

    def __init__(self, loss='binary_crossentropy', learning_rate=0.1,
                 max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0., max_bins=256,
                 max_no_improvement=5, validation_split=0.1,
                 scoring='neg_mean_squared_error', tol=1e-7, verbose=0,
                 random_state=None):
        super(GradientBoostingClassifier, self).__init__(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_bins=max_bins,
            max_no_improvement=max_no_improvement,
            validation_split=validation_split, scoring=scoring, tol=tol,
            verbose=verbose, random_state=random_state)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        raw_predictions = self._raw_predict(X)
        return self.loss_.predict_proba(raw_predictions)


@njit(parallel=True)
def _update_y_pred(leaves_data, y_pred):
    """Read prediction data on the training set from the grower leaves"""
    for leaf_idx in prange(len(leaves_data)):
        leaf_value, sample_indices = leaves_data[leaf_idx]
        for sample_idx in sample_indices:
            y_pred[sample_idx] += leaf_value
