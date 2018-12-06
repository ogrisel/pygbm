import numpy as np
from numpy.testing import assert_almost_equal
from scipy.optimize import newton
from scipy.special import logsumexp
import pytest

from pygbm.loss import _LOSSES
from pygbm.loss import _logsumexp


def get_derivatives_helper(loss):
    """Helper that returns get_gradients() and get_hessians() functions for
    a given loss. Loss classes used to have get_gradients() and
    get_hessians() methods, but now the update is done inplace in
    update_gradient_and_hessians(). This helper is used to keep the tests
    almost unchanged."""

    def get_gradients(y_true, raw_predictions):
        # create gradients and hessians array, update inplace, and return
        shape = raw_predictions.shape[0] * raw_predictions.shape[1]
        gradients = np.empty(shape=shape, dtype=raw_predictions.dtype)
        hessians = np.empty(shape=shape, dtype=raw_predictions.dtype)
        loss.update_gradients_and_hessians(gradients, hessians, y_true,
                                           raw_predictions)

        if loss.__class__ is _LOSSES['least_squares']:
            gradients *= 2  # ommitted a factor of 2 to be consistent with LGBM

        return gradients

    def get_hessians(y_true, raw_predictions):
        # create gradients and hessians array, update inplace, and return
        shape = raw_predictions.shape[0] * raw_predictions.shape[1]
        gradients = np.empty(shape=shape, dtype=raw_predictions.dtype)
        hessians = np.empty(shape=shape, dtype=raw_predictions.dtype)
        loss.update_gradients_and_hessians(gradients, hessians, y_true,
                                           raw_predictions)

        if loss.__class__ is _LOSSES['least_squares']:
            # hessians aren't updated because they're constant
            hessians = np.full_like(y_true, fill_value=2)

        return hessians

    return get_gradients, get_hessians


@pytest.mark.parametrize('loss, x0, y_true', [
    ('least_squares', -2., 42),
    ('least_squares', 117., 1.05),
    ('least_squares', 0., 0.),
    ('binary_crossentropy', 0.3, 0),
    ('binary_crossentropy', -12, 1),
    ('binary_crossentropy', 30, 1),
])
def test_derivatives(loss, x0, y_true):
    # Check that gradients are zero when the loss is minimized on 1D array
    # using the Newton-Raphson and the first and second order derivatives
    # computed by the Loss instance.

    loss = _LOSSES[loss]()
    y_true = np.array([y_true], dtype=np.float32)
    x0 = np.array([x0], dtype=np.float32).reshape(1, 1)
    get_gradients, get_hessians = get_derivatives_helper(loss)

    def func(x):
        return loss(y_true, x)

    def fprime(x):
        return get_gradients(y_true, x)

    def fprime2(x):
        return get_hessians(y_true, x)

    optimum = newton(func, x0=x0, fprime=fprime, fprime2=fprime2)
    assert np.allclose(loss.inverse_link_function(optimum), y_true)
    assert np.allclose(loss(y_true, optimum), 0)
    assert np.allclose(get_gradients(y_true, optimum), 0)


@pytest.mark.parametrize('loss, n_classes, n_trees_per_iteration', [
    ('least_squares', 0, 1),
    ('binary_crossentropy', 2, 1),
    ('categorical_crossentropy', 3, 3),
])
def test_numerical_gradients(loss, n_classes, n_trees_per_iteration):
    # Make sure gradients and hessians computed in the loss are correct, by
    # comparing with their approximations computed with finite central
    # differences.
    # See https://en.wikipedia.org/wiki/Finite_difference.

    rng = np.random.RandomState(0)
    n_samples = 100
    if loss == 'least_squares':
        y_true = rng.normal(size=n_samples).astype(np.float64)
    else:
        y_true = rng.randint(0, n_classes, size=n_samples).astype(np.float64)
    raw_predictions = rng.normal(
        size=(n_samples, n_trees_per_iteration)
    ).astype(np.float64)
    loss = _LOSSES[loss]()
    get_gradients, get_hessians = get_derivatives_helper(loss)

    # [:n_samples] to only take gradients and hessians of first tree.
    gradients = get_gradients(y_true, raw_predictions)[:n_samples]
    hessians = get_hessians(y_true, raw_predictions)[:n_samples]

    # Approximate gradients
    # For multiclass loss, we should only change the predictions of one tree
    # (here the first), hence the use of offset[:n_samples] += eps
    # As a softmax is computed, offsetting the whole array by a constant would
    # have no effect on the probabilities, and thus on the loss
    eps = 1e-9
    offset = np.zeros_like(raw_predictions)
    offset[:, 0] = eps
    f_plus_eps = loss(y_true, raw_predictions + offset / 2, average=False)
    f_minus_eps = loss(y_true, raw_predictions - offset / 2, average=False)
    numerical_gradient = (f_plus_eps - f_minus_eps) / eps
    numerical_gradient = numerical_gradient

    # Approximate hessians
    eps = 1e-4  # need big enough eps as we divide by its square
    offset[:, 0] = eps
    f_plus_eps = loss(y_true, raw_predictions + offset, average=False)
    f_minus_eps = loss(y_true, raw_predictions - offset, average=False)
    f = loss(y_true, raw_predictions, average=False)
    numerical_hessians = (f_plus_eps + f_minus_eps - 2 * f) / eps**2
    numerical_hessians = numerical_hessians

    def relative_error(a, b):
        return np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))

    assert np.all(relative_error(numerical_gradient, gradients) < 1e-5)
    assert np.all(relative_error(numerical_hessians, hessians) < 1e-5)


def test_logsumexp():
    # check consistency with scipy's version
    rng = np.random.RandomState(0)
    for _ in range(100):
        a = rng.uniform(0, 1000, size=1000)
        assert_almost_equal(logsumexp(a), _logsumexp(a))

    # Test whether logsumexp() function correctly handles large inputs.
    # (from scipy tests)

    b = np.array([1000, 1000])
    desired = 1000.0 + np.log(2.0)
    assert_almost_equal(_logsumexp(b), desired)

    n = 1000
    b = np.full(n, 10000, dtype='float64')
    desired = 10000.0 + np.log(n)
    assert_almost_equal(_logsumexp(b), desired)
