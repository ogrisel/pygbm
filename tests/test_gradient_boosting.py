import os
import warnings

import pytest
from sklearn.utils.testing import assert_raises_regex
from sklearn.datasets import make_classification, make_regression
from numpy.testing import assert_array_equal

from pygbm import GradientBoostingClassifier
from pygbm import GradientBoostingRegressor


X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(random_state=0)


@pytest.mark.parametrize('GradientBoosting, X, y', [
    (GradientBoostingClassifier, X_classification, y_classification),
    (GradientBoostingRegressor, X_regression, y_regression)
])
def test_init_parameters_validation(GradientBoosting, X, y):

    assert_raises_regex(
        ValueError,
        "Loss blah is not supported for",
        GradientBoosting(loss='blah').fit, X, y
    )

    for learning_rate in (-1, 0):
        assert_raises_regex(
            ValueError,
            f"learning_rate={learning_rate} must be strictly positive",
            GradientBoosting(learning_rate=learning_rate).fit, X, y
        )

    assert_raises_regex(
        ValueError,
        f"max_iter=0 must not be smaller than 1",
        GradientBoosting(max_iter=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        f"max_leaf_nodes=0 should not be smaller than 1",
        GradientBoosting(max_leaf_nodes=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        f"max_depth=0 should not be smaller than 1",
        GradientBoosting(max_depth=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        f"min_samples_leaf=0 should not be smaller than 1",
        GradientBoosting(min_samples_leaf=0).fit, X, y
    )

    assert_raises_regex(
        ValueError,
        f"l2_regularization=-1 must be positive",
        GradientBoosting(l2_regularization=-1).fit, X, y
    )

    for max_bins in (1, 257):
        assert_raises_regex(
            ValueError,
            f"max_bins={max_bins} should be no smaller than 2 and no larger",
            GradientBoosting(max_bins=max_bins).fit, X, y
        )

    assert_raises_regex(
        ValueError,
        f"n_iter_no_change=1 must not be smaller than 2",
        GradientBoosting(n_iter_no_change=1).fit, X, y
    )

    for validation_split in (-1, 0):
        assert_raises_regex(
            ValueError,
            f"validation_split={validation_split} must be strictly positive",
            GradientBoosting(validation_split=validation_split).fit, X, y
        )

    assert_raises_regex(
        ValueError,
        f"tol=-1 must not be smaller than 0",
        GradientBoosting(tol=-1).fit, X, y
    )


def test_one_sample_one_feature():
    # Until numba issue #3569 is fixed, we raise an informative error message
    # when X is only one sample or one feature in fit (it's OK in predict).
    # The array is both F and C contiguous, and numba can't compile.
    gb = GradientBoostingClassifier()
    for X, y in (([[1, 2]], [0]), ([[1], [2]], [0, 1])):
        assert_raises_regex(
            ValueError,
            'Passing only one sample or one feature is not supported yet.',
            gb.fit, X, y
        )


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Travis times out without numba")
@pytest.mark.parametrize('scoring, validation_split, tol', [
    ('neg_mean_squared_error', .1, 1e-7),
    ('neg_mean_squared_error', None, 1e-1),  # need much higher tol on trainset
    (None, None, None),  # no early stopping
])
def test_early_stopping_regression(scoring, validation_split, tol):

    max_iter = 500
    n_iter_no_change = 5

    X, y = make_regression(random_state=0)

    gb = GradientBoostingRegressor(verbose=1,  # just for coverage
                                   scoring=scoring,
                                   tol=tol,
                                   validation_split=validation_split,
                                   max_iter=max_iter,
                                   n_iter_no_change=n_iter_no_change,
                                   random_state=0)
    gb.fit(X, y)

    if scoring is not None:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Travis times out without numba")
@pytest.mark.parametrize('data', (
    make_classification(random_state=0),
    make_classification(n_classes=3, n_clusters_per_class=1, random_state=0)
))
@pytest.mark.parametrize('scoring, validation_split, tol', [
    ('accuracy', .1, .01),
    ('accuracy', None, .1),
    (None, None, None),  # no early stopping
])
def test_early_stopping_classification(data, scoring, validation_split, tol):

    max_iter = 500
    n_iter_no_change = 5

    X, y = data

    gb = GradientBoostingClassifier(verbose=1,  # just for coverage
                                    scoring=scoring,
                                    tol=tol,
                                    validation_split=validation_split,
                                    max_iter=max_iter,
                                    n_iter_no_change=n_iter_no_change,
                                    random_state=0)
    gb.fit(X, y)

    if scoring is not None:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


# TODO: Remove if / when numba issue 3569 is fixed and check_classifiers_train
# is less strict
def custom_check_estimator(Estimator):
    # Same as sklearn.check_estimator, skipping tests that can't succeed.

    from sklearn.utils.estimator_checks import _yield_all_checks
    from sklearn.utils.testing import SkipTest
    from sklearn.exceptions import SkipTestWarning
    from sklearn.utils import estimator_checks

    estimator = Estimator
    name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        if (check is estimator_checks.check_fit2d_1feature or
                check is estimator_checks.check_fit2d_1sample):
            # X is both Fortran and C aligned and numba can't compile.
            # Opened numba issue 3569
            continue
        if check is estimator_checks.check_classifiers_train:
            continue # probas don't exactly sum to 1 (very close though)
        if (hasattr(check, 'func') and
                check.func is estimator_checks.check_classifiers_train):
            continue # same, wrapped in a functools.partial object.

        try:
            check(name, estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


@pytest.mark.skipif(
    int(os.environ.get("NUMBA_DISABLE_JIT", 0)) == 1,
    reason="Potentially long")
@pytest.mark.parametrize('Estimator', (
    GradientBoostingRegressor(),
    GradientBoostingClassifier(scoring=None, min_samples_leaf=5),))
def test_estimator_checks(Estimator):
    # Run the check_estimator() test suite on GBRegressor and GBClassifier.

    # Notes:
    # - Can't do early stopping with classifier because often
    #   validation_split=.1 leads to test_size=2 < n_classes and
    #   train_test_split raises an error.
    # - Also, need to set a low min_samples_leaf for
    #   check_classifiers_classes() to pass: with only 30 samples on the
    #   dataset, the root is never split with min_samples_leaf=20 and only the
    #   majority class is predicted.
    custom_check_estimator(Estimator)
