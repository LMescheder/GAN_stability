import numpy as np

def sigmoid(x):
    m = np.minimum(0, x)
    return np.exp(m)/(np.exp(m) + np.exp(-x + m))


def clip(x, clipval=0.3):
    x = np.clip(x, -clipval, clipval)
    return x
