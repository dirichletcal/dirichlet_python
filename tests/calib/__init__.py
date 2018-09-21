import numpy as np

def get_simple_binary_example():
    S = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]).T
    S = np.hstack((np.flip(S, axis=0), S))
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return S, y
