from __future__ import annotations

import numpy as np


class ScalerFromStats:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / (self.scale_ + 1e-12)
