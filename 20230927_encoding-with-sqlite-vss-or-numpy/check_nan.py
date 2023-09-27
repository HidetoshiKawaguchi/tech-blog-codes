import numpy as np
np.random.seed(3939)
n_samples, x_dim = 100000, 1000
X = np.random.uniform(-2, 2, (n_samples, x_dim))
X = X.astype(np.float32)
print(any(any(np.isnan(x)) for x in X)) # Xに一つでもnanが含まれていればTrueになる
