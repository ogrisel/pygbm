import numpy as np
from numba import jitclass, njit
from numba import uint32, float32


@jitclass([
    ('target', float32[::1]),
    ('count', uint32[::1]),
])
class Histogram:
    """Summarize the distribution of the target for a given binned feature"""

    def __init__(self, n_bins=256):
        self.target = np.empty(n_bins, dtype=np.float32)
        self.count = np.empty(n_bins, dtype=np.uint32)

    def build(self, sample_indices, binned_feature, target):
        self.target.fill(0)
        self.count.fill(0)
        for sample_idx in sample_indices:
            bin_idx = binned_feature[sample_idx]
            self.count[bin_idx] += 1
            self.target[bin_idx] += target[sample_idx]
        return self


@njit
def build_histogram(n_bins, sample_indices, binned_feature, target):
    return Histogram(n_bins).build(sample_indices, binned_feature, target)