import numpy as np
from scipy.optimize import newton
import pytest

from pygbm.loss import _LOSSES


def get_derivatives_helper(loss):
    """Helper that returns get_gradients() and get_hessians() functions for
    a given loss. Loss classes used to have get_gradients() and
    get_hessians() methods, but now the update is done inplace in
    update_gradient_and_hessians(). This helper is used to keep the tests
    almost unchanged."""

    def get_gradients(y_true, y_pred):
        # create gradients and hessians array, update inplace, and return
        gradients = np.empty_like(y_true)
        hessians = np.empty_like(y_true)
        loss.update_gradients_and_hessians(gradients, hessians, y_true,
                                           y_pred)

        if loss.__class__ is _LOSSES['least_squares']:
            gradients *= 2  # ommitted a factor of 2 to be consistent with LGBM

        return gradients

    def get_hessians(y_true, y_pred):
        # create gradients and hessians array, update inplace, and return
        gradients = np.empty_like(y_true)
        hessians = np.empty_like(y_true)
        loss.update_gradients_and_hessians(gradients, hessians, y_true,
                                           y_pred)

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
    x0 = np.array([x0], dtype=np.float32)
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


@pytest.mark.parametrize('loss', ('least_squares', 'binary_crossentropy'))
def test_gradients_and_hessians_values(loss):
    # Make sure gradients and hessians computed in the loss are correct, by
    # comparing with their approximations computed with finite central
    # differences.
    # See https://en.wikipedia.org/wiki/Finite_difference.

    rng = np.random.RandomState(0)
    n_samples = 100
    if loss == 'least_squares':
        y_true = rng.normal(size=n_samples).astype(np.float64)
    else:
        y_true = rng.randint(0, 2, size=n_samples).astype(np.float64)
    y_pred = rng.normal(size=(n_samples)).astype(np.float64)
    loss = _LOSSES[loss]()
    get_gradients, get_hessians = get_derivatives_helper(loss)

    gradients = get_gradients(y_true, y_pred)
    hessians = get_hessians(y_true, y_pred)

    # Approximate gradients
    eps = 1e-9
    f_plus_eps = loss(y_true, y_pred + eps / 2, average=False)
    f_minus_eps = loss(y_true, y_pred - eps / 2, average=False)
    numerical_gradient = (f_plus_eps - f_minus_eps) / eps

    # Approximate hessians
    eps = 1e-4  # need big enough eps as we divide by its square
    f_plus_eps = loss(y_true, y_pred + eps, average=False)
    f_minus_eps = loss(y_true, y_pred - eps, average=False)
    f = loss(y_true, y_pred, average=False)
    numerical_hessians = (f_plus_eps + f_minus_eps - 2 * f) / eps**2

    def relative_error(a, b):
        return np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))

    assert np.all(relative_error(numerical_gradient, gradients) < 1e-5)
    assert np.all(relative_error(numerical_hessians, hessians) < 1e-5)
