import numpy as np
from numba import jitclass, njit
from numba import uint32, float32


@jitclass([
    ('sum_gradients', float32[::1]),
    ('sum_hessians', float32[::1]),
    ('count', uint32[::1]),
])
class Histogram:
    """Summarize the distribution of the target for a given binned feature"""

    def __init__(self, n_bins=256):
        self.sum_gradients = np.empty(n_bins, dtype=np.float32)
        self.sum_hessians = np.empty(n_bins, dtype=np.float32)
        self.count = np.empty(n_bins, dtype=np.uint32)

    def build(self, sample_indices, binned_feature,
              ordered_gradients, ordered_hessians):
        self.sum_gradients.fill(0)
        self.sum_hessians.fill(0)
        self.count.fill(0)
        for i, sample_idx in enumerate(sample_indices):
            bin_idx = binned_feature[sample_idx]
            self.count[bin_idx] += 1
            self.sum_gradients[bin_idx] += ordered_gradients[i]
            self.sum_hessians[bin_idx] += ordered_hessians[i]
        return self
