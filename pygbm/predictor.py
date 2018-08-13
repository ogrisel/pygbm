import numpy as np
from numba import njit, from_dtype, prange


PREDICTOR_RECORD_DTYPE = np.dtype([
    ('is_leaf', np.uint8),
    ('weight', np.float32),
    ('feature_idx', np.uint32),
    ('bin_threshold', np.uint8),
    ('num_feature_threshold', np.float32),
    ('left', np.uint32),
    ('right', np.uint32),
    # TODO: gain for feature importance?
    # TODO: fraction of training set in leaf for feature importance error bar?
    # TODO: shrinkage in leaf for feature importance error bar?
])
PREDICTOR_NUMBA_TYPE = from_dtype(PREDICTOR_RECORD_DTYPE)[::1]


class TreePredictor:
    def __init__(self, nodes):
        self.nodes = nodes

    def predict_binned(self, binned_features, out=None):
        if out is None:
            out = np.empty(binned_features.shape[0], dtype=np.float32)
        _predict_binned(self.nodes, binned_features, out)
        return out

    def predict(self, X):
        # TODO: introspect X to dispatch to numerical or categorical data
        # (dense or sparse) on a feature by feature basis.
        out = np.empty(X.shape[0], dtype=np.float32)
        _predict_from_num_features(self.nodes, X, out)
        return out


@njit
def _predict_one_binned(nodes, binned_features):
    node = nodes[0]
    while True:
        if node.is_leaf:
            return node.weight
        if binned_features[node['feature_idx']] <= node['bin_threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


@njit(parallel=True)
def _predict_binned(nodes, binned_features, out):
    for i in prange(binned_features.shape[0]):
        out[i] = _predict_one_binned(nodes, binned_features[i])


@njit
def _predict_one_from_num_features(nodes, num_features):
    node = nodes[0]
    while True:
        if node.is_leaf:
            return node.weight
        if num_features[node['feature_idx']] <= node['num_feature_threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


@njit(parallel=True)
def _predict_from_num_features(nodes, num_features, out):
    for i in prange(num_features.shape[0]):
        out[i] = _predict_one_from_num_features(nodes, num_features[i])
