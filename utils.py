import numpy as np


def clip_for_log(X):
    eps = 1e-30
    return np.clip(X, eps, 1-eps)


def clip(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)

