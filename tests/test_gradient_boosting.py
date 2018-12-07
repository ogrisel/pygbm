import os

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


@pytest.mark.parametrize('n_classes', (2, 3))
@pytest.mark.parametrize('mapping', (lambda x: str(x),
                                     lambda x: 2 * x))
def test_classification_input(n_classes, mapping):
    # check that GradientBoostingClassifier supports classes encoded as
    # strings or non contiguous ints.
    # TODO: maybe remove once we run the sklearn test suite?
    X, y = make_classification(n_classes=n_classes, n_clusters_per_class=1)

    y_mapped = [mapping(target) for target in y]

    gbc = GradientBoostingClassifier(scoring=None, random_state=0)
    pred = gbc.fit(X, y).predict(X)
    pred_mapped = gbc.fit(X, y_mapped).predict(X)

    pred_mapped_after = [mapping(target) for target in pred]

    assert_array_equal(pred_mapped, pred_mapped_after)


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
