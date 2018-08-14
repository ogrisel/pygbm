import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from pygbm.binning import BinMapper
from pygbm.grower import TreeGrower


def test_boston_dataset():
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=42)

    mapper = BinMapper(random_state=42)
    X_train_binned = mapper.fit_transform(X_train)
    X_test_binned = mapper.transform(X_test)

    gradients = y_train.astype(np.float32)
    hessians = np.ones(1, dtype=np.float32)

    grower = TreeGrower(X_train_binned, gradients, hessians, max_leaf_nodes=31)
    grower.grow()
    predictor = grower.make_predictor(bin_thresholds=mapper.bin_thresholds_)

    assert r2_score(y_train, predictor.predict_binned(X_train_binned)) > 0.9
    assert r2_score(y_test, predictor.predict_binned(X_test_binned)) > 0.65

    assert_allclose(predictor.predict(X_train),
                    predictor.predict_binned(X_train_binned))

    assert_allclose(predictor.predict(X_test),
                    predictor.predict_binned(X_test_binned))

    assert r2_score(y_train, predictor.predict(X_train)) > 0.9
    assert r2_score(y_test, predictor.predict(X_test)) > 0.65
