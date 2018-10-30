import numpy as np


def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)


def clip(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)

