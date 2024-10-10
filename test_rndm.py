import numpy as np

idx = 0
rng = np.random.default_rng(idx)
print(rng.integers(low=0, high=10, size=3))
