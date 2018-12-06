from abc import ABC, abstractmethod

from scipy.special import expit, logsumexp
import numpy as np
from numba import njit, prange
import numba


# TODO: Write proper docstrings

@njit
def _logsumexp(a):
    # custom logsumexp function with numerical stability, based on scipy's
    # logsumexp which is unfortunately not supported (neither is
    # np.logaddexp.reduce). Only supports 1d arrays.

    a_max = np.amax(a)
    if not np.isfinite(a_max):
        a_max = 0

    s = np.sum(np.exp(a - a_max))
    return np.log(s) + a_max


@njit
def _get_threads_chunks(total_size):
    # Divide [0, total_size - 1] into n_threads contiguous regions, and
    # returns the starts and ends of each region. Used to simulate a 'static'
    # scheduling.
    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, total_size // n_threads, dtype=np.int32)
    sizes[:total_size % n_threads] += 1
    starts = np.zeros(n_threads, dtype=np.int32)
    starts[1:] = np.cumsum(sizes[:-1])
    ends = starts + sizes

    return starts, ends, n_threads


@njit(fastmath=True)
def _expit(x):
    # custom sigmoid because we cannot use that of scipy with numba
    return 1 / (1 + np.exp(-x))


class Loss(ABC):

    def init_gradients_and_hessians(self, n_samples, n_trees_per_iteration):
        shape = n_samples * n_trees_per_iteration
        gradients = np.empty(shape=shape, dtype=np.float32)
        if self.hessian_is_constant:
            hessians = np.ones(shape=1, dtype=np.float32)
        else:
            hessians = np.empty(shape=shape, dtype=np.float32)

        return gradients, hessians

    @abstractmethod
    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        pass


class LeastSquares(Loss):

    hessian_is_constant = True

    def __call__(self, y_true, raw_predictions, average=True):
        # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        loss = np.power(y_true - raw_predictions, 2)
        return loss.mean() if average else loss

    def inverse_link_function(self, raw_predictions):
        return raw_predictions

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        return _update_gradients_least_squares(gradients, y_true,
                                               raw_predictions)


@njit(parallel=True, fastmath=True)
def _update_gradients_least_squares(gradients, y_true, raw_predictions):
    # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
    # return a view.
    raw_predictions = raw_predictions.reshape(-1)
    n_samples = raw_predictions.shape[0]
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            # Note: a more correct exp is 2 * (raw_predictions - y_true) but
            # since we use 1 for the constant hessian value (and not 2) this
            # is strictly equivalent for the leaves values.
            gradients[i] = raw_predictions[i] - y_true[i]


class BinaryCrossEntropy(Loss):

    hessian_is_constant = False
    inverse_link_function = staticmethod(expit)

    def __call__(self, y_true, raw_predictions, average=True):
        # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        # logaddexp(0, x) = log(1 + exp(x))
        loss = np.logaddexp(0, raw_predictions) - y_true * raw_predictions
        return loss.mean() if average else loss

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        return _update_gradients_hessians_binary_crossentropy(
            gradients, hessians, y_true, raw_predictions)

    def predict_proba(self, raw_predictions):
        # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        proba = np.empty((raw_predictions.shape[0], 2), dtype=np.float32)
        proba[:, 1] = expit(raw_predictions)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


@njit(parallel=True, fastmath=True)
def _update_gradients_hessians_binary_crossentropy(gradients, hessians,
                                                   y_true, raw_predictions):
    # Note: using LightGBM version (first mapping {0, 1} into {-1, 1})
    # will cause overflow issues in the exponential as we're using float32
    # precision.

    # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
    # return a view.
    raw_predictions = raw_predictions.reshape(-1)
    n_samples = raw_predictions.shape[0]
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            gradients[i] = _expit(raw_predictions[i]) - y_true[i]
            gradient_abs = np.abs(gradients[i])
            hessians[i] = gradient_abs * (1. - gradient_abs)


class CategoricalCrossEntropy(Loss):

    hessian_is_constant = False

    def __call__(self, y_true, raw_predictions, average=True):
        one_hot_true = np.zeros_like(raw_predictions)
        n_trees_per_iteration = raw_predictions.shape[1]
        for k in range(n_trees_per_iteration):
            one_hot_true[:, k] = (y_true == k)

        return (logsumexp(raw_predictions, axis=1) -
                (one_hot_true * raw_predictions).sum(axis=1))

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        return _update_gradients_hessians_categorical_crossentropy(
            gradients, hessians, y_true, raw_predictions)

    def predict_proba(self, raw_predictions):
        # TODO: This could be done in parallel
        # compute softmax (using exp(log(softmax)))
        return np.exp(raw_predictions -
                      logsumexp(raw_predictions, axis=1)[:, np.newaxis])


@njit(parallel=True)
def _update_gradients_hessians_categorical_crossentropy(
        gradients, hessians, y_true, raw_predictions):
    # Here gradients and hessians are of shape
    # (n_samples * n_trees_per_iteration,).
    # y_true is of shape (n_samples,).
    # raw_predictions is of shape (n_samples, raw_predictions)
    #
    # Instead of passing the whole gradients and hessians arrays and slicing
    # them here, we could instead do the update in the 'for k in ...' loop of
    # fit(), by passing gradients_at_k and hessians_at_k which are of size
    # (n_samples,).
    # That would however require to pass a copy of raw_predictions, so it does
    # not get partially overwritten at the end of the loop when
    # _update_y_pred() is called (see sklearn PR 12715)
    n_samples, n_trees_per_iteration = raw_predictions.shape
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for k in range(n_trees_per_iteration):
        gradients_at_k = gradients[n_samples * k:n_samples * (k + 1)]
        hessians_at_k = hessians[n_samples * k:n_samples * (k + 1)]
        for thread_idx in prange(n_threads):
            for i in range(starts[thread_idx], ends[thread_idx]):
                # p_k is the probability that class(ith sample) == k.
                # This is a regular softmax.
                p_k = np.exp(raw_predictions[i, k] -
                             _logsumexp(raw_predictions[i, :]))
                gradients_at_k[i] = p_k - (y_true[i] == k)
                hessians_at_k[i] = p_k * (1. - p_k)
                # LightGBM uses 2 * p_k * (1 - p_k) which is not stricly
                # correct but equivalent to using half the learning rate.


_LOSSES = {'least_squares': LeastSquares,
           'binary_crossentropy': BinaryCrossEntropy,
           'categorical_crossentropy': CategoricalCrossEntropy}
