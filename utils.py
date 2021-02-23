import numpy as np

class Smoothie:
    def __init__(self, smoothing=0.99):
        self._values = {}
        self._smoothing = 0.99

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None or not np.isfinite(v):
                continue
            if k in self._values:
                self._values[k] = self._smoothing * \
                    self._values[k] + (1 - self._smoothing) * v
            else:
                self._values[k] = v

    def value(self):
        return self._values.copy()
