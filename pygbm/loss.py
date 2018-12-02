from abc import ABC

from scipy.special import expit
import numpy as np
from numba import njit, prange
import numba


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

    def init_gradients_and_hessians(self, n_samples):
        gradients = np.empty(shape=n_samples, dtype=np.float32)
        if self.hessian_is_constant:
            hessians = np.ones(shape=1, dtype=np.float32)
        else:
            hessians = np.empty(shape=n_samples, dtype=np.float32)

        return gradients, hessians


class LeastSquares(Loss):

    hessian_is_constant = True

    def __call__(self, y_true, y_pred, average=True):
        loss = np.power(y_true - y_pred, 2)
        return loss.mean() if average else loss

    def inverse_link_function(self, raw_predictions):
        return raw_predictions

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      y_pred):
        return _update_gradients_least_squares(gradients, y_true, y_pred)


@njit(parallel=True, fastmath=True)
def _update_gradients_least_squares(gradients, y_true, y_pred):
    n_samples = gradients.shape[0]
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            # Note: a more correct exp is 2 * (y_pred - y_true) but since we
            # use 1 for the constant hessian value (and not 2) this is
            # strictly equivalent for the leaves values.
            gradients[i] = y_pred[i] - y_true[i]


class BinaryCrossEntropy(Loss):

    hessian_is_constant = False
    inverse_link_function = staticmethod(expit)

    def __call__(self, y_true, y_pred, average=True):
        # logaddexp(0, x) = log(1 + exp(x))
        loss = np.logaddexp(0, y_pred) - y_true * y_pred
        return loss.mean() if average else loss

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      y_pred):
        return _update_gradients_hessians_logistic(gradients, hessians,
                                                   y_true, y_pred)

    def predict_proba(self, raw_predictions):
        proba = np.empty((raw_predictions.shape[0], 2), dtype=np.float32)
        proba[:, 1] = expit(raw_predictions)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


@njit(parallel=True, fastmath=True)
def _update_gradients_hessians_logistic(gradients, hessians, y_true, y_pred):
    # Note: using LightGBM version (first mapping {0, 1} into {-1, 1})
    # will cause overflow issues in the exponential as we're using float32
    # precision.

    n_samples = gradients.shape[0]
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            gradients[i] = _expit(y_pred[i]) - y_true[i]
            gradient_abs = np.abs(gradients[i])
            hessians[i] = gradient_abs * (1. - gradient_abs)


_LOSSES = {'least_squares': LeastSquares,
           'binary_crossentropy': BinaryCrossEntropy}
